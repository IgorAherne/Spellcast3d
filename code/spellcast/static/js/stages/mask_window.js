// Smart Mask's frame (marks.js): the part of the picture a masked change takes, shown to the artist while they brush.
// A picture that is not tileable is drawn anew whole by the image server, and of a large one a PATCH of that round
// what was brushed is laid onto the picture (patchOf; no frame where the patch is all of the picture). A tileable one
// is drawn in a WINDOW of it only - a box round what was brushed, cut out at the picture's own size -, and a mask too
// large for a window is refused. Everything here is in the pixels of the picture asked for (the request's), and
// mirrors launchers/qwen_server.py - patch_of, window_of, to_the_middle and same_shape: the two change together.

import { strokePart, strokeTail } from './strokes.js';

export const PATCH_SIDE = 1024;            // a patch's sides at the least; a picture of up to that squared is all its patch's
export const PATCH_MARGIN = 0.05;          // a patch stays this share of the picture inside its sides, from afar ...
export const PATCH_SHY = 1.5;              // ... and half of it where what was brushed lies this many margins off
export const WINDOW_PIXELS = 1024 * 1024;  // the most a window holds
export const WINDOW_SIDE = 768;            // its sides at the least, where WINDOW_PIXELS allows both
export const KEEP_ROOM = 0.12;             // it reaches beyond what was brushed, each way, the room a change may be
export const WINDOW_RIM = 32;              // drawn in (this share of the picture's longer side) and a rim ...
export const WINDOW_BUFFER_MIN = 96;       // ... and at the least
export const WINDOW_STEP = 32;             // its sides in steps of this (qwen_server REFERENCE_STEP)

/** `length` up to the next multiple of WINDOW_STEP. */
const stepped = (length) => Math.ceil(length / WINDOW_STEP) * WINDOW_STEP;

/** The sides of the least box round spans: what was brushed and WINDOW_BUFFER_MIN each way, in steps, inside `size`. */
const leastSides = (spans, size) => spans.map((span, i) => Math.min(size[i], stepped(span + 2 * WINDOW_BUFFER_MIN)));

const spansOf = (brushed) => [brushed[2] - brushed[0], brushed[3] - brushed[1]];

/**
 * The window of the box `brushed` ([left, top, right, bottom], right and bottom not in it) in a picture of `size`
 * ([width, height]): [left, top, right, bottom], or null where what was brushed is too large for one (window_of).
 */
export function windowOf(brushed, size) {
  const spans = spansOf(brushed);
  const least = leastSides(spans, size);
  if (least[0] * least[1] > WINDOW_PIXELS) return null;
  const buffer = Math.round(KEEP_ROOM * Math.max(...size)) + WINDOW_RIM;
  const sides = spans.map((span, i) => Math.min(size[i], Math.max(stepped(span + 2 * buffer), WINDOW_SIDE)));
  while (sides[0] * sides[1] > WINDOW_PIXELS) {
    sides[sides[0] - least[0] >= sides[1] - least[1] ? 0 : 1] -= WINDOW_STEP;   // the side with more to give
  }
  const [left, top] = [0, 1].map((i) => Math.min(Math.max(Math.floor((brushed[i] + brushed[i + 2] - sides[i]) / 2), 0),
    size[i] - sides[i]));
  return [left, top, left + sides[0], top + sides[1]];
}

/**
 * The patch of the box `brushed` in a picture of `size`: what was brushed with its room each way, PATCH_SIDE a side at
 * the least, round the middle of what was brushed; all of a picture of up to PATCH_SIDE squared pixels (the model sees
 * that one at its own size), and all of one along a side no longer than PATCH_SIDE. It stays off the picture's sides -
 * PATCH_MARGIN where what was brushed lies far from one, less and less the nearer (keptOff) -, so it nears a side
 * without a jump and is on it only where what was brushed is (patch_of: the same sums in the same order).
 */
export function patchOf(brushed, size) {
  if (size[0] * size[1] <= PATCH_SIDE ** 2) return [0, 0, ...size];
  const room = Math.round(KEEP_ROOM * Math.max(...size));
  // how far it stays from a side that what was brushed lies `gap` from, whole pixels: a third of the way at the most
  const keptOff = (gap, margin) => {
    const half = PATCH_SHY * margin;
    return Math.floor((margin * gap * gap) / (gap * gap + half * half));
  };
  const along = (low, high, full) => {
    if (full <= PATCH_SIDE) return [0, full];
    const margin = PATCH_MARGIN * full;
    const [near, far] = [keptOff(low, margin), keptOff(full - high, margin)];
    const side = Math.max(PATCH_SIDE, high - low + 2 * room);
    if (side >= full - near - far) return [near, full - far];
    const left = Math.min(Math.max(Math.floor((low + high - side) / 2), near), full - far - side);
    return [left, left + side];
  };
  const [[left, right], [top, bottom]] = [0, 1].map((i) => along(brushed[i], brushed[i + 2], size[i]));
  return [left, top, right, bottom];
}

/** A picture of size `a` ([w, h]), resized to the width of `b`, is as high as it to a pixel or two (same_shape). */
export const sameShape = (a, b) => Math.abs(a[0] * b[1] - b[0] * a[1]) <= 2 * a[0];

/**
 * Along one side `length` long, round its edge, where the stretches `runs` ([[from, to], ...], whole pixels inside it)
 * brushed it -> {start, span}: what lies between the ends of the longest stretch nothing was brushed in, the one
 * to_the_middle splits half to either side - the first such going twice round from 0, as it finds it -; the whole
 * side when nothing is left unbrushed. Null without runs.
 */
export function roundTheEdge(runs, length) {
  if (!runs.length) return null;
  const merged = joined(runs);
  const [first, last] = [merged[0][0], merged.at(-1)[1]];
  const inner = merged.slice(1).map(([from], i) => [merged[i][1], from - merged[i][1]]);
  const round = [last % length, length - last + first];   // [begins, long]: across the edge
  // in the order to_the_middle comes to their ends: one that begins at 0 first, one across the edge last
  let longest = 0;
  let begins = 0;
  for (const [at, long] of last === length ? [round, ...inner] : [...inner, round]) {
    if (long > longest) [longest, begins] = [long, at];
  }
  return { start: (begins + longest) % length, span: length - longest };
}

/** The stretches `runs` ([[from, to], ...]) as few: in order, those that meet or overlap as one. */
function joined(runs) {
  const out = [];
  for (const [from, to] of [...runs].sort((a, b) => a[0] - b[0])) {
    const last = out.at(-1);
    if (last && from <= last[1]) last[1] = Math.max(last[1], to);
    else out.push([from, to]);
  }
  return out;
}

/** Where a part of a stroke (strokes.js strokePart) reaches along x and along y, its brush's width with it. */
function reachOf(part) {
  const r = part.width / 2;
  return ['x', 'y'].map((k) => {
    if (part.dot) return [part.to[k] - r, part.to[k] + r];
    const [a, c, b] = [part.from[k], part.ctrl[k], part.to[k]];
    const ends = [a, b];
    const t = (a - c) / (a - 2 * c + b);                  // where the curve turns back, if it does
    if (t > 0 && t < 1) ends.push((1 - t) ** 2 * a + 2 * t * (1 - t) * c + t ** 2 * b);
    return [Math.min(...ends) - r, Math.max(...ends) + r];
  });
}

/** `part`'s reach added to `runs` (per axis), in whole pixels of `size` for a picture of `shown`, if it lies on it. */
function addReach(runs, part, shown, size) {
  const [across, down] = reachOf(part).map(([lo, hi], i) => {
    const k = size[i] / shown[i];
    return [Math.max(0, Math.floor(lo * k)), Math.min(size[i], Math.ceil(hi * k))];
  });
  if (across[1] <= across[0] || down[1] <= down[0]) return;
  runs[0] = joined([...runs[0], across]);
  runs[1] = joined([...runs[1], down]);
}

// A stroke's points -> the reach of its parts that no point to come changes (strokes.js strokePart: all but its tail),
// so a long stroke under way adds only its new parts.
const settled = new WeakMap();

/**
 * Per axis, the stretches the parts of `strokes`, drawn on a picture of `shown` pixels, reach in whole pixels of one of
 * `size` (what the mask has white there, a pixel more at most), inside it.
 */
function reaches(strokes, shown, size) {
  const out = [[], []];
  const scale = `${shown} ${size}`;
  for (const { points } of strokes) {
    if (!points.length) continue;
    let kept = settled.get(points);
    if (kept?.scale !== scale) settled.set(points, kept = { scale, parts: 0, runs: [[], []] });
    for (; kept.parts < Math.max(1, points.length - 1); kept.parts++) {
      addReach(kept.runs, strokePart(points, kept.parts), shown, size);
    }
    const runs = kept.runs.map((axis) => [...axis]);
    const tail = strokeTail(points);
    if (tail) addReach(runs, tail, shown, size);
    out[0].push(...runs[0]);
    out[1].push(...runs[1]);
  }
  return out;
}

/**
 * Smart Mask's frame for the mask `strokes` drawn on a picture of `shown` pixels ([w, h]), asked for as `request`
 * ({width, height, tileable}) -> null where all of the picture is the change's (no request, one of another shape,
 * nothing brushed on it, a patch that is the whole picture); else {box: [left, top, right, bottom] in the request's
 * pixels, size: [width, height], fits}: the patch, which always fits; of a tileable picture the window - or, where
 * the mask is too large for one (fits: false), the least box round what was brushed. That box may reach past the
 * right and the bottom edge: what does, comes back at the other side.
 */
export function maskWindow(strokes, shown, request) {
  const size = request && [request.width, request.height];
  if (!size || !sameShape(shown, size)) return null;
  const runs = reaches(strokes, shown, size);
  if (!runs[0].length) return null;
  if (!request.tileable) {
    const brushed = [0, 1].map((i) => Math.min(...runs[i].map((r) => r[0])))
      .concat([0, 1].map((i) => Math.max(...runs[i].map((r) => r[1]))));
    const box = patchOf(brushed, size);
    if (box[2] - box[0] === size[0] && box[3] - box[1] === size[1]) return null;
    return { box, size, fits: true };
  }
  // A tileable picture is moved first, what was brushed to its middle (to_the_middle): its window is found there and
  // laid back where the picture shows.
  const middle = runs.map((axis, i) => {
    const { start, span } = roundTheEdge(axis, size[i]);
    const low = Math.ceil((size[i] - span) / 2);
    return { low, high: low + span, back: start - low };
  });
  return framed([middle[0].low, middle[1].low, middle[0].high, middle[1].high], size, middle.map((m) => m.back));
}

/** The window of `brushed` in `size` (or its least box), laid `back` along each axis, wrapping round. */
function framed(brushed, size, back) {
  const window = windowOf(brushed, size);
  const box = window || [0, 1].map((i) => Math.max(0, brushed[i] - WINDOW_BUFFER_MIN))
    .concat([0, 1].map((i) => Math.min(size[i], brushed[i + 2] + WINDOW_BUFFER_MIN)));
  const [left, top] = [0, 1].map((i) => (((box[i] + back[i]) % size[i]) + size[i]) % size[i]);
  return { box: [left, top, left + box[2] - box[0], top + box[3] - box[1]], size, fits: !!window };
}

/**
 * The box `box` as it shows on a picture of `size`, in parts inside it - what reaches past the right or the bottom edge
 * comes back at the other side - each {box, open}: the sides that are the picture's edge cutting it ('l', 't', 'r',
 * 'b'), not the box's own. A box as long as a side lies along all of it.
 */
export function windowParts(box, size) {
  const along = (i) => {
    const [low, high] = [box[i], box[i + 2]];
    if (high - low >= size[i]) return [[0, size[i], []]];
    if (high <= size[i]) return [[low, high, []]];
    return [[low, size[i], [['r', 'b'][i]]], [0, high - size[i], [['l', 't'][i]]]];
  };
  return along(0).flatMap(([x0, x1, xOpen]) => along(1).map(([y0, y1, yOpen]) => ({
    box: [x0, y0, x1, y1], open: [...xOpen, ...yOpen],
  })));
}
