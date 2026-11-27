// The Omni-Texture viewer's tools, on the prop's result - or on its 3D model, which the first tool taken up makes a
// result of its own (bind's `begin`: paintbar.js) -: Projection's paint bar (stages/paintbar.js) - the marker and the
// clone stamp, whose strokes are saved into the step shown as hand paint (the texels painted alone: the hub keeps
// them as a layer over the steps it makes again) - with a tool of its own at the column's foot, the mask marker. It
// paints magenta over what the artist wants made again (the model's overlay, paint3d.js, on the page alone; the x to
// its left discards the mask). Once something is masked, a box beside the mask - in the room the model leaves, near
// the mask, never on it (askPlace) - takes words of the artist's own, if they have any, and holds "Render Masked".
// That sends the hub what is masked (the overlay's texels covered at least half way), the viewport's camera in the
// model's own space and the words; the hub's job has Qwen-Image paint the masked surface again as that camera sees it
// (adapters/omnitexture.py retouch) and the prop gets a new result, one more of its omnitexture.steps. While the mask
// marker is up, the viewport outside the square that camera sees is covered in black; the view moves as with any
// tool (Alt + drag, the middle and right buttons, the wheel), and the box goes to its new place once it rests. The
// box shows only then; with the marker put down, the side panel's "Render Masked" renders. Either sends the view as
// it is at the press, with the words, and puts the marker down. In the box, left of its button, "No Facets": the
// depth map Qwen-Image is shown is then blurred until the model's flat faces are gone from it (`smoothing`).
// Under the mask marker stands "Inverse perspective": a button that bends the viewport's view round the model
// (viewer3d.js LensCamera: the lines of sight meet behind it), so that what turns away from a camera - a face's
// cheeks - shows, is masked and is repainted from there; a slider to its left sets how far. "Render Masked" sends
// that lens with the view, and the hub's worker looks through the same one. Gone when the artist leaves the stage's
// view (no other stage shows it).
// Ctrl+Z takes back the newest change first: a stroke (paint3d.js keeps them per model), or a repaint, a picture
// taken off the model or the whole result taken off it (their × on the right: stages/omnitexture.js), each a step of
// its own - the prop then shows the step before it again with its mask as it was, stroke by stroke (the page keeps a
// model's mask when it leaves the model, and puts it on again when that model is loaded once more). Ctrl+Y goes
// forward the same way. The magenta is only ever on the page: nothing is saved of it but the mask a repaint was made
// from.

import { api, propApi, putBytes } from '../api.js';
import { replaceProp, startJob } from '../actions.js';
import { spinner } from '../controls.js';
import { h } from '../dom.js';
import { icon } from '../icons.js';
import { projectId } from '../store.js';
import { toast, toastError } from '../ui.js';
import { isNewest, undoList, undoStamp } from '../undo.js';
import { createPaintBar } from './paintbar.js';
import { TAKEN_OFF } from './rules.js';

export const MASK_COLOUR = '#ff00ff';
const MASKED = 128;                // an overlay texel covered this far or more is masked (half way)
const HINT = 'Mask marker: paint over what to make again, then “Render Masked” beside it';
const LOST = 'The mask went with it: paint it again then.';
// The box beside the mask (askPlace), in CSS pixels: how far it keeps from the model, the mask and the viewport's
// buttons, and from the viewport's edge; the places tried for it, this far apart.
const ASK_GAP = 16;
const ASK_EDGE = 10;
const ASK_STEP = 8;
const SIGHT_WIDE = 160;            // the model in sight is looked at this many pixels wide to find room (paint3d.js sight)
// The viewport's buttons the box keeps clear of: the paint bar's glasses, the stage's own.
const KEEP_CLEAR = '.pt-tools, .pt-row, .mk-lens, .pt-wipe, .sv-viewer-tools, .ot-go, .btn.donate, .btn.discord';
const WORDS_MOST = 400;            // (the hub takes no more: adapters/omnitexture.py retouch_prompt)
const WORDS_HINT = 'e.g. a smooth armour plate with a few scratches';
const REST_MS = 200;               // the view at rest this long: the box goes to its place in it
const FACETS = 'No Facets: Qwen-Image is shown the depth map blurred, without the model’s flat faces and hard '
  + 'creases, so it paints them rounded';

const diverged = new WeakSet();    // sessions masked again after a Ctrl+Z: Ctrl+Y no longer leads on from them
const stamps = new Map();          // "project/prop/glb" of a step's model -> the stamp of the change that made it
const kept = new Map();            // "project/prop/glb" -> the mask on that model as the page left it (paint3d keep)
const KEPT = 6;                    // ... of that many models at most: each holds its strokes' tiles
const lastShown = new Map();       // "project/prop" -> the model last shown of it: tells a repaint's result as it arrives
// "Inverse perspective" as the artist set it, kept from prop to prop while the page is open: on or off, and how far
// along its slider (0 to 1). The slider's whole way takes the view's perspective (viewer3d.js LensCamera) from a
// camera's own, 1, down by LENS_SPAN: a quarter of the way there is none left, from there on it is inverse.
const lens = { on: false, amount: 0.5 };
const LENS_SPAN = 4;
const BEND = {
  off: 'Inverse perspective: bends the view round the model, so that what turns away from you - the cheeks of a '
    + 'face - shows and can be masked and repainted',
  on: 'Inverse perspective is on: click for the usual view again',
  amount: 'How far the view is bent: a quarter of the way there is no perspective left; beyond that, what is farther '
    + 'shows larger and the sides turn towards you',
};
// "No Facets" as the artist set it, kept like the lens: on, a repaint's depth map goes without the model's facets.
const smoothing = { on: false };

/** The view's perspective (LensCamera.perspective) for a setting of "Inverse perspective": {on, amount}. */
export const perspectiveOf = ({ on, amount }) => (on ? 1 - amount * LENS_SPAN : 1);

/** The texels masked, from the overlay's coverage (paint3d.js readOverlay) -> {mask: a byte per texel, 255 masked; count}. */
export function maskOf(cover) {
  const mask = new Uint8Array(cover.length);
  let count = 0;
  for (let t = 0; t < cover.length; t++) {
    if (cover[t] < MASKED) continue;
    mask[t] = 255;
    count++;
  }
  return { mask, count };
}

/** The largest centred square of a `width` x `height` viewport: its side as a share of the viewport's height. */
export const squareShare = (width, height) => Math.min(width, height) / (height || 1);

/** How much of rect `b` ({x0, y0, x1, y1}) lies within `gap` of rect `a`: the area. */
const within = (a, b, gap) => Math.max(0, Math.min(a.x1 + gap, b.x1) - Math.max(a.x0 - gap, b.x0))
  * Math.max(0, Math.min(a.y1 + gap, b.y1) - Math.max(a.y0 - gap, b.y0));

/** Where a side of `len` may start along a viewport side of `room`: ASK_STEP apart, ASK_EDGE inside its ends. */
function startsAlong(len, room) {
  const last = room - len - ASK_EDGE;
  if (last < ASK_EDGE) return [Math.max(0, (room - len) / 2)];
  const starts = [];
  for (let v = ASK_EDGE; v < last; v += ASK_STEP) starts.push(v);
  starts.push(last);
  return starts;
}

/**
 * Where the box for the mask's words goes: its top left {x, y} in the viewport (CSS pixels), or null with nothing
 * masked in sight. ASK_EDGE inside the viewport, ASK_GAP from the mask, from the `clear` rects ({x0, y0, x1, y1}: the
 * viewport's buttons) and from the model in sight, as near the mask as that leaves room; where there is none, it
 * covers as little as can be of the model first, then of the buttons, and last of the mask. sight: paint3d.js
 * Session.sight's, of a viewport `view` ({w, h}); box: {w, h}; square: the part of the viewport not covered in black
 * ({x0, y0, x1, y1}: the model outside it is out of sight).
 */
export function askPlace(sight, view, box, { clear = [], square = null } = {}) {
  const { width: gw, height: gh } = sight;
  const sx = view.w / gw;
  const sy = view.h / gh;
  const seen = (x, y) => !square || ((x + 0.5) * sx >= square.x0 && (x + 0.5) * sx <= square.x1
    && (y + 0.5) * sy >= square.y0 && (y + 0.5) * sy <= square.y1);
  // Summed counts of the model's and of the mask's pixels: how many lie in any rect, at once.
  const model = new Int32Array((gw + 1) * (gh + 1));
  const mask = new Int32Array((gw + 1) * (gh + 1));
  const masked = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
  for (let y = 0; y < gh; y++) {
    let inRow = 0;
    let maskInRow = 0;
    for (let x = 0; x < gw; x++) {
      const i = y * gw + x;
      inRow += sight.model[i] && seen(x, y) ? 1 : 0;
      if (sight.cover[i] >= MASKED) {
        maskInRow++;
        masked.x0 = Math.min(masked.x0, x * sx);
        masked.y0 = Math.min(masked.y0, y * sy);
        masked.x1 = Math.max(masked.x1, (x + 1) * sx);
        masked.y1 = Math.max(masked.y1, (y + 1) * sy);
      }
      const at = (y + 1) * (gw + 1) + x + 1;
      model[at] = model[at - gw - 1] + inRow;
      mask[at] = mask[at - gw - 1] + maskInRow;
    }
  }
  if (masked.x1 < 0) return null;
  /** The share of the pixels within ASK_GAP of `r` that `sums` counts. */
  const share = (sums, r) => {
    const x0 = Math.max(0, Math.floor((r.x0 - ASK_GAP) / sx));
    const x1 = Math.min(gw, Math.ceil((r.x1 + ASK_GAP) / sx));
    const y0 = Math.max(0, Math.floor((r.y0 - ASK_GAP) / sy));
    const y1 = Math.min(gh, Math.ceil((r.y1 + ASK_GAP) / sy));
    if (x1 <= x0 || y1 <= y0) return 0;
    const w = gw + 1;
    return (sums[y1 * w + x1] - sums[y0 * w + x1] - sums[y1 * w + x0] + sums[y0 * w + x0]) / ((x1 - x0) * (y1 - y0));
  };
  const middle = { x: (masked.x0 + masked.x1) / 2, y: (masked.y0 + masked.y1) / 2 };
  // What a place covers counts before how far it is, the mask before the buttons before the model: each step of
  // `far` outweighs any distance in the viewport, and every share of what is covered after it.
  const far = view.w + view.h;
  const cost = (part) => (part > 0 ? far * (1 + part) : 0);
  let best = null;
  for (const y of startsAlong(box.h, view.h)) {
    for (const x of startsAlong(box.w, view.w)) {
      const r = { x0: x, y0: y, x1: x + box.w, y1: y + box.h };
      const buttons = clear.reduce((sum, c) => sum + within(r, c, ASK_GAP), 0) / (box.w * box.h || 1);
      const gap = Math.hypot(Math.max(0, masked.x0 - r.x1, r.x0 - masked.x1), Math.max(0, masked.y0 - r.y1, r.y0 - masked.y1));
      const off = Math.hypot(x + box.w / 2 - middle.x, y + box.h / 2 - middle.y);
      const score = 16 * cost(share(mask, r)) + 4 * cost(buttons) + cost(share(model, r)) + gap + 0.1 * off;
      if (!best || score < best.score) best = { x, y, score };
    }
  }
  return { x: best.x, y: best.y };
}

/** Where `glb` is among the prop's steps, and the ones before and after it -> {index, before, after} (null: none). */
export function stepsAround(steps, glb) {
  const list = Array.isArray(steps) ? steps : [];
  const index = list.indexOf(glb);
  return { index, before: index > 0 ? list[index - 1] : null, after: index >= 0 && index < list.length - 1 ? list[index + 1] : null };
}

/**
 * A prop's models were made again under other names (a repaint's fade, stages/omnitexture.js: `renamed`, step as
 * it was -> step as it is): what the page keeps per model - the repaints' places in Ctrl+Z's order, the masks - follows.
 */
export function renameModels(project, propId, renamed) {
  const prop = `${project}/${propId}`;
  for (const [was, now] of Object.entries(renamed || {})) {
    for (const map of [stamps, kept]) {
      if (!map.has(`${prop}/${was}`)) continue;
      map.set(`${prop}/${now}`, map.get(`${prop}/${was}`));
      map.delete(`${prop}/${was}`);
    }
    if (lastShown.get(prop) === was) lastShown.set(prop, now);
  }
}

/** The page took a prop's result off as a whole (stages/omnitexture.js): the newest change, Ctrl+Z's first. */
export function tookOff(project, propId) {
  stamps.set(`${project}/${propId}/${TAKEN_OFF}`, undoStamp());
}

/**
 * Saves the hand paint on the step `glb` of the prop's Omni-Texture: the texels painted alone (readAll's patch, alpha
 * 255 where painted, 0 elsewhere) -> the prop. Nothing is sent when no texel differs from the files (a stroke taken
 * back before it was saved).
 */
export function saveHandPaint({ project, propId, glb }, image) {
  if (!image.patch) return null;
  const query = `glb=${encodeURIComponent(glb)}&width=${image.width}&height=${image.height}`;
  return putBytes(`${propApi(project, propId)}/omnitexture/paint?${query}`, image.patch);
}

/** A mask (a byte per texel) as a PNG's bytes. */
async function maskPng(mask, size) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const context = canvas.getContext('2d');
  const image = context.createImageData(size, size);
  for (let t = 0, i = 0; t < mask.length; t++, i += 4) {
    image.data[i] = image.data[i + 1] = image.data[i + 2] = mask[t];
    image.data[i + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
  if (!blob) throw new Error('The mask could not be made into a picture.');
  return new Uint8Array(await blob.arrayBuffer());
}

/**
 * The viewer's camera in the space of the model on screen, as the viewport's largest centred square sees through it:
 * {eye, forward, up, tan}, and `wide` when its lens is bent. Half that square's side at depth z from the eye is
 * wide + tan * z (viewer3d.js LensCamera.lens): a camera's own is the tangent of half its angle, and no `wide`.
 */
export function cameraOf(viewer) {
  const object = viewer.current.children[0];
  object.updateWorldMatrix(true, false);
  viewer.camera.updateMatrixWorld();
  const inverse = object.matrixWorld.clone().invert();
  const v = viewer.camera.position.clone();
  const eye = viewer.camera.getWorldPosition(v).applyMatrix4(inverse).toArray();
  const forward = viewer.camera.getWorldDirection(v).transformDirection(inverse).toArray();
  const up = v.set(0, 1, 0).transformDirection(viewer.camera.matrixWorld).transformDirection(inverse).toArray();
  const canvas = viewer.renderer.domElement;
  const square = squareShare(canvas.clientWidth || canvas.width, canvas.clientHeight || canvas.height);
  const { tan, wide } = viewer.camera.lens();
  const scale = object.getWorldScale(v).x;       // (`wide` is a length: sent in the model's own units)
  return { eye, forward, up, tan: tan * square, ...(wide ? { wide: (wide * square) / scale } : {}) };
}

/**
 * The tools of one Omni-Texture view: {el, cover (nodes for the viewport), bind({viewer, project, propId, glb, steps,
 * step?, select(view), begin?}), unbind(), settle(), flush(), destroy()}. step: the prop's step the model stands for
 * when it is not the model itself (TAKEN_OFF: the 3D model, shown for a result taken off). cover: black outside the square "Render Masked" sends,
 * while the mask marker is up; select: moves the model to a map view on the artist's behalf (the map bar's select,
 * `byArtist`); busy(): the stage works on the prop
 * (its job starting, a fade, a picture being taken off): nothing paints and Ctrl+Z waits. settle(): see below.
 */
export function createMaskBar({ busy: stageBusy = () => false, changed: told = () => {} } = {}) {
  let binding = null;       // the step shown: {viewer, project, propId, glb, steps, step, select}
  let token = 0;
  let working = false;      // the mask is on its way to the hub, or a step is being taken

  const key = (glb) => `${binding.project}/${binding.propId}/${glb}`;
  const stepShown = () => binding.step ?? binding.glb;
  const around = () => (binding ? stepsAround(binding.steps, stepShown()) : { index: -1, before: null, after: null });
  /** The stamp of the change that made the step shown (0: it is the first step, or one from before this visit). */
  const renderStamp = () => (binding && around().before ? stamps.get(key(stepShown())) || 0 : 0);

  const marker = h('button', {
    type: 'button', class: 'pt-tool', 'aria-label': 'Mask marker',
    title: 'Mask marker: paint over what should be made again, then “Render Masked” (Esc puts it down)',
  }, icon('mask'));
  // While the marker is up, the viewport outside the square "Render Masked" sends (squareShare) is covered in
  // black, so the artist frames what is to be painted inside it; the box for the mask's words lies on it too.
  const square = h('div', { class: 'mk-square' });
  const cover = h('div', { class: 'mk-cover', hidden: true }, square);

  // ---- the box beside the mask: the artist's own words for the repaint, if any, and "Render Masked" ------------
  let words = '';           // what the box holds: Qwen-Image hears them after "remove the magenta-masked blob."
  let wordsSent = false;    // ... rendered as they stand: clicked into, the box selects them, to type over
  let placing = 0;          // the box's places asked for (placeAsk): an older one's answer is dropped
  let resting = 0;          // the timer that puts the box in its place once the view rests (moved)
  const glyph = h('span', { class: 'mk-glyph' }, icon('sparkles'));
  const render = h('button', {
    type: 'button', class: 'btn primary small mk-render',
    title: 'Qwen-Image repaints what you masked, as you see it from here, and it goes onto the model',
    onclick: () => renderMasked(),
  }, glyph, 'Render Masked');
  const noFacets = h('button', {
    type: 'button', class: 'pt-flag', title: FACETS, 'aria-pressed': String(smoothing.on),
    onclick: () => {
      smoothing.on = !smoothing.on;
      noFacets.blur();
      sync();
    },
  }, 'No Facets');
  const field = h('textarea', {
    class: 'input mk-ask-input', rows: 3, maxlength: WORDS_MOST, placeholder: WORDS_HINT, spellcheck: 'true',
    'aria-label': 'Anything to add about the masked part (optional)',
  });
  field.addEventListener('input', () => {
    words = field.value;
    wordsSent = false;
  });
  field.addEventListener('focus', () => { if (wordsSent) field.select(); });
  // Enter renders (Shift+Enter: a new line), Escape gives the keys back to the viewer.
  field.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      renderMasked();
    } else if (e.key === 'Escape') {
      field.blur();
    }
  });
  const ask = h('div', { class: 'mk-ask mk-ask-mask ot-ask', hidden: true },
    h('span', { class: 'mk-ask-label' }, 'Anything to add? (optional)'), field,
    h('div', { class: 'mk-ask-row' }, noFacets, render));
  cover.append(ask);
  const resized = globalThis.ResizeObserver ? new ResizeObserver(() => { if (!ask.hidden) placeAsk(); }) : null;
  resized?.observe(cover);
  /** The view moved (the camera, the lens): the box is out of the way until it rests, then goes to its place in it. */
  function moved() {
    if (cover.hidden) return;
    placing++;
    ask.hidden = true;
    clearTimeout(resting);
    resting = setTimeout(placeAsk, REST_MS);
  }
  // Under the marker: "Inverse perspective" (the module's `lens`), and to its left while it is on, how far.
  let seen = null;          // the viewer this view bent the lens of: set straight again when the view goes
  const aim = () => {
    (binding?.viewer || seen)?.setPerspective(perspectiveOf(lens));
    moved();
  };
  const bend = h('button', {
    type: 'button', class: 'pt-tool mk-bend', 'aria-label': 'Inverse perspective',
    onclick: () => {
      lens.on = !lens.on;
      aim();
      sync();
    },
  }, icon('lens'));
  const amount = h('input', {
    type: 'range', class: 'range', min: 0, max: 100, value: Math.round(lens.amount * 100), title: BEND.amount,
    'aria-label': 'How far the view is bent',
  });
  amount.addEventListener('input', () => {
    lens.amount = amount.value / 100;
    amount.style.setProperty('--fill', `${amount.value}%`);
    aim();
  });
  amount.addEventListener('pointerup', () => amount.blur());    // (the keys belong to the viewer again)
  const amountGlass = h('span', { class: 'mk-lens', hidden: true }, amount);
  const bar = createPaintBar({
    save: saveHandPaint,
    patches: true,
    overlay: {
      button: marker, colour: MASK_COLOUR, hint: HINT, lost: LOST,
      wipe: 'Discard the mask (Ctrl+Z brings it back)',
      up: () => {
        cover.hidden = false;
        sync();
        placeAsk();
      },
      down: () => { cover.hidden = true; sync(); },
      stroke: () => {                  // the box out of the way while the mask grows; placed anew after the stroke
        placing++;
        ask.hidden = true;
      },
    },
    foot: h('span', { class: 'pt-own' }, amountGlass, bend),
    busy: () => working || stageBusy(),
    undo: walk,
    changed: (layer) => {
      const s = bar.session();
      if (layer === 'overlay' && s && around().after) diverged.add(s);
      sync();
      if (layer === 'overlay') placeAsk();
    },
  });

  function sync() {
    const busy = working || stageBusy();
    if (!askable()) ask.hidden = true;
    render.disabled = busy;
    marker.disabled = busy;
    noFacets.setAttribute('aria-pressed', String(smoothing.on));
    bend.disabled = amount.disabled = busy;
    bend.title = lens.on ? BEND.on : BEND.off;
    bend.setAttribute('aria-pressed', String(lens.on));
    amountGlass.hidden = !lens.on;
    amount.value = Math.round(lens.amount * 100);
    amount.style.setProperty('--fill', `${amount.value}%`);
    const spinning = working && !ask.hidden;
    if (glyph.dataset.busy !== String(spinning)) {
      glyph.dataset.busy = String(spinning);
      glyph.replaceChildren(spinning ? spinner() : icon('sparkles'));
    }
    told();
  }
  /** The artist is masking: the marker is up, or something on the model is masked. */
  const masking = () => !cover.hidden || !!bar.session()?.painted('overlay');
  /** The box beside the mask may show: the marker is up and something is masked. */
  const askable = () => !cover.hidden && !!binding && !!bar.session()?.painted('overlay') && !bar.painting();

  /**
   * The box goes where askPlace finds room for it beside the mask, in the view as it is now: the model and the mask
   * as the viewport shows them (paint3d.js sight), the viewport's buttons as they stand (KEEP_CLEAR).
   */
  async function placeAsk() {
    const mine = ++placing;
    const s = askable() && bar.session();
    const view = s && binding.viewer.renderer.domElement.getBoundingClientRect();
    if (!view?.width || !view.height) {
      ask.hidden = true;
      return;
    }
    let sight;
    try {
      sight = await s.sight(SIGHT_WIDE, Math.max(1, Math.round((SIGHT_WIDE * view.height) / view.width)));
    } catch (e) {
      console.warn(e);           // the graphics card was reset: the box waits for the next change
      return;
    }
    if (mine !== placing || !askable()) return;
    const inView = (r) => ({ x0: r.left - view.left, y0: r.top - view.top, x1: r.right - view.left, y1: r.bottom - view.top });
    const clear = [...(cover.parentElement?.querySelectorAll(KEEP_CLEAR) || [])]
      .map((node) => node.getBoundingClientRect()).filter((r) => r.width && r.height).map(inView);
    const measuring = ask.hidden;
    if (measuring) {
      ask.style.visibility = 'hidden';
      ask.hidden = false;
    }
    const seen = square.getBoundingClientRect();
    const at = askPlace(sight, { w: view.width, h: view.height }, { w: ask.offsetWidth, h: ask.offsetHeight },
      { clear, square: seen.width ? inView(seen) : null });
    ask.style.visibility = '';
    const origin = cover.getBoundingClientRect();
    ask.hidden = !at;
    if (!at) return;
    ask.style.left = `${view.left - origin.left + at.x}px`;
    ask.style.top = `${view.top - origin.top + at.y}px`;
  }

  // ---- "Render Masked" ------------------------------------------------------------------------------------------
  async function renderMasked() {
    const s = bar.session();
    if (working || stageBusy() || !s || !binding || bar.painting()) return;
    const { viewer, project, propId, glb } = binding;
    const mine = token;
    working = true;
    sync();
    try {
      await bar.flush();         // the hand paint into the files first: the hub takes none while the job is under way
      const { data, width } = await s.readOverlay();
      const { mask, count } = maskOf(data);
      if (!count) {
        toast('Nothing is masked yet: paint over what should be made again first.', 'info');
        return;
      }
      const camera = cameraOf(viewer);
      const said = words.trim();
      const png = await maskPng(mask, width);
      if (mine !== token) return;
      const sent = await putBytes(`${propApi(project, propId)}/omnitexture/mask?glb=${encodeURIComponent(glb)}`, png);
      bar.pick(null);
      diverged.delete(s);          // what follows this mask from now on is the repaint made of it
      keepMask();
      await startJob('omnitexture', [propId], {
        params: {
          retouch: { mask: sent.mask, ...camera, ...(said ? { words: said } : {}), ...(smoothing.on ? { smooth: true } : {}) },
        },
      });
      wordsSent = true;
    } catch (e) {
      kept.delete(`${project}/${propId}/${glb}`);
      console.warn(e);
      toastError(e);
    } finally {
      working = false;
      sync();
    }
  }

  // ---- Ctrl+Z / Ctrl+Y: after the strokes, the prop's steps ------------------------------------------------------
  /** The mask on the model shown, kept for when the page shows that model again (a step retraced, a model made again). */
  function keepMask() {
    if (!binding) return;
    const mark = key(binding.glb);
    kept.delete(mark);
    const s = bar.session();
    const mask = s?.keep('overlay');
    if (!mask || !(mask.steps.length || mask.undone.length || mask.strokes.overlay)) return;
    kept.set(mark, { ...mask, diverged: diverged.has(s) });
    for (const oldest of kept.keys()) {
      if (kept.size <= KEPT) break;
      kept.delete(oldest);
    }
  }

  /** The prop shows another of its steps (`forward`: a repaint again, the newest change once more). */
  async function step(to, forward = false) {
    const { project, propId } = binding;
    working = true;
    sync();
    keepMask();
    try {
      if (forward) stamps.set(`${project}/${propId}/${to}`, undoStamp());
      const prop = await api('POST', `${propApi(project, propId)}/omnitexture/step`, { glb: to });
      if (projectId() === project) replaceProp(prop);
    } catch (e) {
      toastError(e);
    } finally {
      working = false;
      sync();
    }
  }

  /** Ctrl+Z / Ctrl+Y once no stroke on the model is the change to take back or bring back (paintbar.js asks). */
  function walk(which) {
    if (!binding) return false;
    const { before, after } = around();
    if (which === 'undo') {
      const repaint = renderStamp();
      if (!repaint || !isNewest(repaint)) return false;
      step(before);
      return true;
    }
    const s = bar.session();
    if (after == null || (s && diverged.has(s))) return false;
    step(after, true);
    return true;
  }
  const dropUndo = undoList(renderStamp);

  // ---- binding to a model -----------------------------------------------------------------------------------
  async function bind(next) {
    if (binding && next && binding.glb === next.glb && binding.propId === next.propId && binding.viewer === next.viewer) {
      binding = { ...binding, ...next };
      sync();
      return;
    }
    unbind();
    if (!next?.viewer) return;
    const mine = token;
    await bar.bind(next);
    if (mine !== token) return;
    binding = { ...next };
    seen = binding.viewer;
    seen.controls.addEventListener?.('change', moved);
    aim();
    // A step seen for the first time after the one before it (a repaint, a picture taken off): the newest change
    // on the page, so Ctrl+Z takes it back first.
    const shown = key(stepShown());
    const prop = `${binding.project}/${binding.propId}`;
    const { before } = around();
    if (before && lastShown.get(prop) === before && !stamps.has(shown)) stamps.set(shown, undoStamp());
    lastShown.set(prop, stepShown());
    // A mask kept of this model goes onto it again, strokes and all, when the viewer loaded the model anew (while
    // the viewer held it, the mask is still on: put() then refuses).
    const mark = key(binding.glb);
    const mask = kept.get(mark);
    kept.delete(mark);
    const s = mask && bar.session(true);
    if (s?.put(mask) && mask.diverged) diverged.add(s);
    sync();
    placeAsk();
  }

  function unbind() {
    token++;
    clearTimeout(resting);
    binding?.viewer.controls.removeEventListener?.('change', moved);
    bar.unbind();
    binding = null;
    sync();
  }

  return {
    el: bar.el,
    cover,
    bind,
    unbind,
    // For the side panel's button, which stands in for "Render Masked" while the artist is masking (`changed`
    // tells when any of this may have changed).
    masking,
    working: () => working,
    render: renderMasked,
    /**
     * The page is about to have the hub make the prop's steps again (a fade, a picture taken off, a job): the hand
     * paint goes into the files first - the hub takes none while that is under way - and the mask on the model is
     * kept for when it comes back -> once the paint is in the files.
     */
    async settle() {
      await bar.flush();
      keepMask();
    },
    flush: () => bar.flush(),
    destroy() {
      if (!cover.hidden) bar.pick(null);       // the marker is this view's: no other arrives with its view locked
      seen?.setPerspective(1);                 // ... nor with its lens bent
      resized?.disconnect();
      unbind();
      dropUndo();
      bar.destroy();
    },
  };
}
