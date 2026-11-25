// The slicer over the Images workbench's picture: the artist marks which part of a picture shows which
// side of the object - a turnaround sheet cut into its views, or pictures of the object from several
// sides - so the 3D generator builds the model from all of them (Pixal3D's multi-view models).
//
// Its buttons are one row above the picture, in the workbench's header (the `toolbar` createSlicer is given),
// while a picture shows and none is being made: "4 quadrants", "4 in a row", "Slice into views", then "Save" and "Copy"
// (the picture, onto the clipboard). While slicing is on, the four views show before them (Front, Back, ← Left,
// Right →: SHEET_ORDER): a drag draws the slice of the view picked there (Shift: a square), and the next view without
// a slice is picked. The drag may begin beside the picture, in the empty room around it: the slice then starts at
// the picture's edge, and it is never smaller than MIN_BOX_PX on screen nor outside the picture. "4 quadrants" and
// "4 in a row" cut the whole picture at once, a little inside its edges and apart from each other (PRESET_EDGE,
// PRESET_GAP), past the lines an image generator likes to draw between the views. Only the slices lie over the picture,
// which has the whole of its box (and, while slicing, a small line over its top edge: what the views lack).
// Each slice has a chip with its view's name in its bottom-left corner: a click on it changes its view or
// removes it. While slicing, a square button above the chip moves the slice (press and drag) and one to the
// chip's right removes it. Ctrl+Z undoes (an import of several pictures too), Escape stops slicing (a move
// in progress first). The views are the prop's images.views (rules.js),
// saved to the hub as they change (actions.queueViews); a picture shows the slices made of it. They are
// used while the chosen picture is one of theirs (rules.effectiveViews): the slicer always shows the chosen
// one, so a picture without slices makes the 3D model alone, until it gets a slice too.

import { h, isTyping } from '../dom.js';
import { icon } from '../icons.js';
import { getProp, store } from '../store.js';
import { queueViews, selectImage } from '../actions.js';
import { attempt, dialogOpen } from '../ui.js';
import { isNewest, isUndoKey, undoList, undoStamp } from '../undo.js';
import { closeMenuIn, menuOpen, popupMenu } from '../shell/popups.js';
import { VIEW_ORDER, effectiveViews, viewsSignature } from './rules.js';

/**
 * The four views: their names on the buttons and chips, the strip's short one, the tooltip and colour.
 * "← Left" and "Right →" say which way the object faces in the picture: the arrow points where it looks.
 */
export const VIEWS = {
  front: { label: 'Front', short: 'Front', title: 'Front view: the 3D model faces it', color: '#ffb547' },
  left: { label: '← Left', short: '← Left', title: 'Left view: the object faces left', color: '#4cc3ff' },
  back: { label: 'Back', short: 'Back', title: 'Back view', color: '#ff7eb6' },
  right: { label: 'Right →', short: 'Right →', title: 'Right view: the object faces right', color: '#62dc8e' },
};

/** The smallest slice the hub takes, in the picture's pixels. */
export const MIN_SIDE = 16;

const GLOW_MS = 750;     // after a new picture shows, "Slice into views" glows once
const DRAG_PX = 4;       // a press that moves less is a click, not a slice
const MIN_BOX_PX = 10;   // a slice being drawn is at least this big on screen, in each direction
// The presets keep clear of the picture's edges and of each other, as shares of its width and height: a sheet's
// views often come with a frame around the sheet and a cross (or bars) between them.
export const PRESET_EDGE = 0.02;
export const PRESET_GAP = 0.02;

// Icons of the slicer only (icons.js has the shared ones).
const ICONS = {
  slice: '<path d="M6.5 2.5v12a3 3 0 0 0 3 3h12"/><path d="M2.5 6.5h12a3 3 0 0 1 3 3v12"/>',
  quad: '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M12 4v16M4 12h16"/>',
  row: '<rect x="2.5" y="7" width="19" height="10" rx="2"/><path d="M7.3 7v10M12 7v10M16.7 7v10"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>',
  move: '<path d="M12 3v18M3 12h18M9.5 5.5L12 3l2.5 2.5M9.5 18.5L12 21l2.5-2.5M5.5 9.5L3 12l2.5 2.5M18.5 9.5L21 12l-2.5 2.5"/>',
};

function ownIcon(name) {
  const tpl = document.createElement('template');
  tpl.innerHTML = '<svg class="icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" '
    + `stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;
  return tpl.content.firstChild;
}

const SVG = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

// ---- the rules of slicing (plain functions: code/tests/web/slicer.test.mjs) ---------------------------

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/**
 * Where a picture of `natural` size ({w, h}) shows in `box` ({left, top, width, height}, the <img>'s box
 * on screen) with object-fit: contain: {left, top, width, height, scale}.
 */
export function fitRect(box, natural) {
  const scale = Math.min(box.width / natural.w, box.height / natural.h);
  const width = natural.w * scale;
  const height = natural.h * scale;
  return { left: box.left + (box.width - width) / 2, top: box.top + (box.height - height) / 2, width, height, scale };
}

/** The picture's point (in its pixels, top-left origin) under the screen point (x, y), kept inside the picture. */
export function toImage(x, y, box, natural) {
  const f = fitRect(box, natural);
  return { x: clamp((x - f.left) / f.scale, 0, natural.w), y: clamp((y - f.top) / f.scale, 0, natural.h) };
}

/**
 * The slice dragged from `a` to `b` (points in the picture): [x, y, w, h] in whole pixels, inside the
 * picture. `square` (Shift): a square as big as the drag's longer side, as far as the picture lets it
 * grow from `a` in the drag's direction. `least`: no side is shorter (a drag that has barely begun, or one that
 * runs along the picture's edge); the slice then grows away from `a` the drag's way, and back from the edge it meets.
 */
export function dragRect(a, b, natural, square = false, least = 0) {
  if (square) {
    const right = b.x >= a.x;
    const down = b.y >= a.y;
    const room = Math.min(right ? natural.w - a.x : a.x, down ? natural.h - a.y : a.y);
    const dragged = Math.min(Math.max(Math.abs(b.x - a.x), Math.abs(b.y - a.y)), room);
    const side = Math.floor(Math.max(dragged, Math.min(least, natural.w, natural.h)));
    const x = clamp(Math.round(right ? a.x : a.x - side), 0, natural.w - side);
    const y = clamp(Math.round(down ? a.y : a.y - side), 0, natural.h - side);
    return [x, y, side, side];
  }
  const span = (from, to, size) => {
    let [lo, hi] = [clamp(Math.min(from, to), 0, size), clamp(Math.max(from, to), 0, size)];
    if (hi - lo < least) [lo, hi] = to >= from ? [lo, lo + least] : [hi - least, hi];
    if (hi > size) [lo, hi] = [lo - (hi - size), size];
    if (lo < 0) [lo, hi] = [0, Math.min(size, hi - lo)];
    return [Math.round(lo), Math.round(hi)];
  };
  const [x0, x1] = span(a.x, b.x, natural.w);
  const [y0, y1] = span(a.y, b.y, natural.h);
  return [x0, y0, x1 - x0, y1 - y0];
}

/**
 * The slice `rect` ([x, y, w, h]) carried (dx, dy) of the picture's pixels away: the same size, in whole
 * pixels, as far as the picture's edges let it go.
 */
export function moveRect(rect, dx, dy, natural) {
  const [x, y, w, hh] = rect;
  return [clamp(Math.round(x + dx), 0, Math.max(0, natural.w - w)), clamp(Math.round(y + dy), 0, Math.max(0, natural.h - hh)), w, hh];
}

/** The hub takes the slice: both sides at least MIN_SIDE pixels. */
export const usable = (rect) => !!rect && rect[2] >= MIN_SIDE && rect[3] >= MIN_SIDE;

/**
 * A preset's four slices, cell by cell in reading order: 'quad' cuts the picture into 2 × 2 cells, 'row' into
 * 1 × 4, each PRESET_EDGE of the picture away from its edges and PRESET_GAP from its neighbours (a quadrant so
 * ends 1 % short of the picture's middle).
 */
export function presetRects(kind, natural) {
  const [cols, rows] = kind === 'row' ? [4, 1] : [2, 2];
  const cut = (size, n, i) => {
    const cell = (1 - 2 * PRESET_EDGE - (n - 1) * PRESET_GAP) / n;
    const from = PRESET_EDGE + i * (cell + PRESET_GAP);
    return [Math.round(size * from), Math.round(size * (from + cell))];
  };
  const rects = [];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const [x0, x1] = cut(natural.w, cols, c);
      const [y0, y1] = cut(natural.h, rows, r);
      rects.push([x0, y0, x1 - x0, y1 - y0]);
    }
  }
  return rects;
}

// A sheet's views in reading order: the front and the back first (the quadrants' top row), ← left and right →
// after them - how the picture generator is asked for a sheet (the prompt library's "4 views"). Also the order of
// the view buttons, of a preset's cells, of the views a drag draws one after the other, and of a slice's menu.
export const SHEET_ORDER = ['front', 'back', 'left', 'right'];

/** The views a preset makes of the picture `src`: its cells in SHEET_ORDER, kept in the views' order. */
export function presetViews(kind, src, natural) {
  return sortViews(presetRects(kind, natural).map((rect, i) => ({ view: SHEET_ORDER[i], src, rect })));
}

/** `views` in the order they are kept in (rules.VIEW_ORDER, the hub's): front, ← left, back, right →. */
export const sortViews = (views) => [...views].sort((a, b) => VIEW_ORDER.indexOf(a.view) - VIEW_ORDER.indexOf(b.view));

/** `views` with `entry` in it, instead of the view of its kind (on whichever picture that was). */
export const putView = (views, entry) => sortViews([...views.filter((v) => v.view !== entry.view), entry]);

/** `views` without the view `view`. */
export const dropView = (views, view) => views.filter((v) => v.view !== view);

/** `views` without the slices of the picture `src`. */
export const dropPicture = (views, src) => views.filter((v) => v.src !== src);

/** The slice of the view `from` becomes the view `to`; the slice that was `to`, if any, becomes `from` (they swap). */
export function relabel(views, from, to) {
  if (from === to) return views;
  return sortViews(views.map((v) => (v.view === from ? { ...v, view: to } : v.view === to ? { ...v, view: from } : v)));
}

/** `view` has no slice on the picture `src` (on any picture, without one). */
const unsliced = (views, view, src) => !views.some((x) => x.view === view && (src == null || x.src === src));

/** `view` is the whole of another picture: that picture was brought in as that side (no slice of a sheet). */
const wholeElsewhere = (views, view, src) => views.some((x) => x.view === view && x.src !== src && !x.rect);

// A picture is sliced in its own order, whatever slices the prop's other pictures hold: the first slice drawn on a
// new sheet is its Front, the next its Back, and a view that had its slice on another sheet takes the new one
// (putView). Only a side that is another picture's whole is left to that picture while a side is still free:
// pictures brought in one per side keep their Front when one of them is sliced.

/** The views a slice on the picture `src` may take, starting after `from`, the ones to leave alone last. */
function free(views, src, from = -1) {
  const order = SHEET_ORDER.map((_, k) => SHEET_ORDER[(from + 1 + k) % SHEET_ORDER.length]);
  const open = order.filter((v) => unsliced(views, v, src));
  return [...open.filter((v) => !wholeElsewhere(views, v, src)), ...open.filter((v) => wholeElsewhere(views, v, src))];
}

/** The view to draw first on the picture `src`; Front when every view has a slice there. */
export const firstFree = (views, src) => free(views, src)[0] || SHEET_ORDER[0];

/** The view to draw after `current` on the picture `src`; `current` when every view has a slice there. */
export const nextView = (views, current, src) => free(views, src, SHEET_ORDER.indexOf(current))
  .find((v) => v !== current) || current;

/** Several views and none is the front: the 3D generator needs one to face. */
export const needsFront = (views) => views.length >= 2 && !views.some((v) => v.view === 'front');

/** The same views, whatever order they are in. */
export const sameViews = (a, b) => viewsSignature(a) === viewsSignature(b);

// ---- pictures imported together: the side each one shows -------------------------------------------

const SIDE_WORDS = { front: 'front', back: 'back', rear: 'back', left: 'left', right: 'right', side: 'side' };
const ANGLE_SIDES = { 0: 'front', 90: 'left', 180: 'back', 270: 'right' };
const byName = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' }).compare;

/** The words and numbers of a file name, without its folder and extension: "C:\art\Bear_Left02.png" -> ['bear', 'left', '02']. */
const nameTokens = (name) => String(name || '').replace(/^.*[\\/]/, '').replace(/\.[^.]*$/, '').toLowerCase()
  .split(/[^a-z0-9]+|(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])/).filter(Boolean);

/** The side a file name says in words ("bear_back.png" -> 'back'; "side" -> 'side': either side), or null. */
const wordSide = (tokens) => tokens.map((t) => SIDE_WORDS[t]).find(Boolean) || null;

/** The angles around the object (0, 90, 180, 270) a file name has among its numbers. */
const angles = (tokens) => tokens.filter((t) => /^\d{1,3}$/.test(t)).map(Number).filter((n) => n in ANGLE_SIDES);

/**
 * Pictures imported together (their file names, as the artist's file picker or drag gave them, in no
 * order the artist chose): the order to upload them in and the side each one shows. The side a name says
 * comes first - "front", "back" (or "rear"), "left", "right", "side" (whichever side is free), or an
 * angle around the object when the names give several (0 front, 90 ← left, 180 back, 270 right →). The
 * others take the sides still free in name order ("2" before "10"): front, ← left, back, right →. A side
 * two names say goes to the first of them in name order; the other, like any picture after the fourth, is
 * just a picture. One picture alone is no view.
 * Returns [{index, side | null, told}] in upload order - the views from front to right →, then the other
 * pictures in name order; index: into `names`; told: the name said the side (false: taken in name order).
 */
export function importSides(names) {
  const tokens = names.map(nameTokens);
  const order = names.map((_, i) => i).sort((a, b) => byName(names[a].replace(/^.*[\\/]/, ''), names[b].replace(/^.*[\\/]/, '')) || a - b);
  if (names.length < 2) return order.map((index) => ({ index, side: null, told: false }));
  // Numbers are angles only when the names give several of them (not "shot 0", "shot 1"...).
  const found = new Set(tokens.flatMap(angles));
  const useAngles = found.size >= 2 && [...found].some((n) => n !== 0);
  const said = tokens.map((t) => wordSide(t) || (useAngles ? ANGLE_SIDES[angles(t)[0]] ?? null : null));
  const side = new Map();   // index -> its side (null: just a picture)
  const taken = new Set();
  const give = (i, s) => {
    side.set(i, s);
    if (s) taken.add(s);
  };
  for (const i of order) if (said[i] && said[i] !== 'side') give(i, taken.has(said[i]) ? null : said[i]);
  for (const i of order) if (said[i] === 'side') give(i, ['left', 'right'].find((s) => !taken.has(s)) || null);
  for (const i of order) if (!side.has(i)) give(i, VIEW_ORDER.find((s) => !taken.has(s)) || null);
  const rank = (i) => (side.get(i) ? VIEW_ORDER.indexOf(side.get(i)) : VIEW_ORDER.length);
  return [...order].sort((a, b) => rank(a) - rank(b))
    .map((index) => ({ index, side: side.get(index), told: !!side.get(index) && !!said[index] }));
}

/**
 * The undo list: record(views) before each change (or {views, chosen} before an import of several pictures:
 * undone, it picks that picture again too); undo() gives what to go back to (null: none); newest: the undo stamp
 * of the last change (undo.js).
 */
export function history(limit = 100) {
  const past = [];
  return {
    record(views) {
      past.push({ views, at: undoStamp() });
      if (past.length > limit) past.shift();
    },
    undo: () => past.pop()?.views ?? null,
    get size() { return past.length; },
    get newest() { return past.at(-1)?.at; },
  };
}

// ---- on the page ------------------------------------------------------------------------------------

/** The picture at `url` as a PNG, the one kind of picture a clipboard takes: as it is, or drawn again as one. */
async function pngOf(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('Could not read the picture to copy it.');
  const blob = await res.blob();
  if (blob.type === 'image/png') return blob;
  const bitmap = await createImageBitmap(blob);
  const canvas = Object.assign(document.createElement('canvas'), { width: bitmap.width, height: bitmap.height });
  canvas.getContext('2d').drawImage(bitmap, 0, 0);
  bitmap.close?.();
  return new Promise((resolve, reject) => {
    canvas.toBlob((png) => (png ? resolve(png) : reject(new Error('Could not copy the picture.'))), 'image/png');
  });
}

/**
 * The tiny mark of a picture in the candidates strip that has views: the view's short name ("Front",
 * "← Left", "Back", "Right →"), or "4 views" for a sheet; null for a picture without views.
 */
export function viewChip(views, src) {
  const mine = sortViews((views || []).filter((v) => v.src === src));
  if (!mine.length) return null;
  const one = mine.length === 1 ? VIEWS[mine[0].view] : null;
  return h('span', {
    class: 'sl-cand-chip', style: one ? `--view: ${one.color}` : null,
    title: mine.map((v) => VIEWS[v.view]?.label || v.view).join(' · '),
  }, one ? one.short : `${mine.length} views`);
}

/**
 * The slicer of the prop `propId`, a layer over its picture (a sibling of .sv-picture in the stage box,
 * so re-renders of the picture keep it). picture(img, src) whenever the workbench shows another <img>
 * (null: no picture), update({busy}) on every render, destroy() when the workbench goes.
 * `toolbar`: an empty element above the picture (the workbench's header), the slicer's own: it holds the row
 * of buttons while a picture shows and none is being made, nothing otherwise. Without one the row lies over
 * the picture's top. `onSave` (-> true once saved): a "Save" button before "Copy".
 */
export function createSlicer(propId, { toolbar = null, onSave = null } = {}) {
  let img = null;           // the <img> the picture shows in
  let src = null;           // its path in the prop ("images/0003.png"), as images.views has it
  let natural = null;       // {w, h}: its size, once loaded
  let on = false;           // slicing
  let active = SHEET_ORDER[0];
  let busy = false;         // a generation runs: the layer over the picture says it, the slicer hides
  let inToolbar = false;    // the row of buttons is in `toolbar`
  let glowedFor = null;     // the picture "Slice into views" glowed for
  let glowTimer = null;
  let regenerated = false;  // a generation just finished: its picture counts as a new one
  let drag = null;          // {id, a, b, x, y, far, square} while a slice is drawn
  let moving = null;        // {id, x, y, rect, btn} while a slice is moved by its move button
  let drawn = null;
  let copiedTimer = null;
  const past = history();
  const dropUndo = undoList(() => past.newest);

  const views = () => getProp(propId)?.images?.views || [];

  // ---- the elements ------------------------------------------------------------------------------
  const shapes = svgEl('g');
  const draft = svgEl('rect', { class: 'sl-draft', display: 'none' });
  const svg = svgEl('svg', { class: 'sl-canvas', preserveAspectRatio: 'xMidYMid meet', 'aria-hidden': 'true' });
  svg.append(shapes, draft);
  const chips = h('div', { class: 'sl-fit' });
  const area = h('div', { class: 'sl-area' }, chips);   // the picture's own box: the slices' buttons sit on it

  const toggle = h('button', {
    type: 'button', class: 'btn small sl-toggle', 'aria-pressed': 'false',
    title: 'Mark which part of the picture shows which side of the object',
    onclick: () => setOn(!on),
  }, ownIcon('slice'), h('span', null, 'Slice into views'));
  const clear = h('button', {
    type: 'button', class: `icon-btn small${toolbar ? '' : ' glass'} sl-clear`, title: 'Remove the slices of this picture',
    onclick: () => {                               // and the views to draw next go with them
      change(dropPicture(views(), src));
      setOn(false);
    },
  }, icon('trash'));
  const viewBtns = SHEET_ORDER.map((v) => h('button', {
    type: 'button', class: 'sl-view', role: 'radio', style: `--view: ${VIEWS[v].color}`,
    onclick: () => { active = v; paint(); },
  }, h('span', { class: 'sl-swatch' }), h('span', { class: 'sl-view-label' }, VIEWS[v].label), icon('check', 'sl-check')));
  const stack = h('div', { class: 'sl-views', role: 'radiogroup', 'aria-label': 'The view you draw next' }, viewBtns);
  // What the views still lack, while slicing: a line over the picture's top edge, which takes no room (the
  // picture stays where it is) and no pointer (a slice may start under it).
  const hint = h('p', { class: 'sl-hint' }, 'Mark one slice as Front.');
  // Several views, but Pixal3D's multi-view models are not downloaded yet (the 3D stage offers them).
  const mvHint = h('p', { class: 'sl-note' }, 'Front only: multi-view isn’t downloaded');
  const hints = h('div', { class: 'sl-hints' }, hint, mvHint);
  const copyLabel = h('span', null, 'Copy');
  const copy = h('button', {
    type: 'button', class: 'btn small sl-copy', title: 'Copy this picture, to paste it somewhere else',
    onclick: () => attempt(copyPicture),
  }, ownIcon('copy'), copyLabel);
  const saveLabel = h('span', null, 'Save');
  const save = onSave && h('button', {
    type: 'button', class: 'btn small sl-save', title: 'Save this picture into the project’s export folder',
    onclick: () => attempt(async () => { if (await onSave()) flash(save, saveLabel, 'Save', 'Saved'); }),
  }, icon('save'), saveLabel);
  const presetBtn = (kind, label, title) => h('button', {
    type: 'button', class: 'btn small sl-preset', title, onclick: () => preset(kind),
  }, ownIcon(kind), h('span', null, label));
  // The row of buttons, centred: the view the next drag draws (while slicing), then the presets, "Slice into
  // views", the bin of this picture's slices, Save and Copy. Where it is narrow the two groups go on a row each (the
  // views over the buttons), never one button alone on a second row.
  const acts = h('div', { class: 'sl-acts' },
    presetBtn('quad', '4 quadrants', 'Cut the picture into 2 × 2 views: front and back on top, ← left and right → under them'),
    presetBtn('row', '4 in a row', 'Cut the picture into 4 views next to each other: front, back, ← left, right →'),
    toggle, clear, save, copy);
  const bar = h('div', { class: 'sl-bar' }, stack, acts);
  // The row is put into `toolbar` as a picture shows (paint); without one it is a part of the layer.
  const el = h('div', { class: 'sl', hidden: true }, svg, area, hints, !toolbar && bar);

  // ---- Copy: the picture shown, onto the clipboard -----------------------------------------------------
  async function copyPicture() {
    if (!img) return;
    if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') throw new Error('This browser can’t copy pictures.');
    // The PNG as a promise: the browser takes the copy at the click, the picture when it is read.
    try {
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': pngOf(img.currentSrc || img.src) })]);
    } catch (e) {   // the browser's own words ("Document is not focused") say little to an artist
      throw new Error('The browser did not let Spellcast3D copy the picture. Click the page and try again.', { cause: e });
    }
    flash(copy, copyLabel, 'Copy', 'Copied');
  }

  /** A button says it did it ("Copied", "Saved") for a moment. */
  function flash(btn, label, word, done) {
    label.textContent = done;
    btn.classList.add('done');
    clearTimeout(copiedTimer);
    copiedTimer = setTimeout(() => {
      label.textContent = word;
      btn.classList.remove('done');
    }, 1600);
  }

  // ---- changes, undo --------------------------------------------------------------------------------
  function change(next, advance = false) {
    const before = views();
    if (sameViews(before, next)) return;
    past.record(before);
    queueViews(propId, next);
    if (advance) active = nextView(next, active, src);
    paint();
  }

  function undo() {
    const back = past.undo();
    if (!back) return false;
    if (Array.isArray(back)) {
      queueViews(propId, back);
    } else {   // an import of several pictures: its views and the picture chosen before it
      queueViews(propId, back.views);
      if (back.chosen && back.chosen !== getProp(propId)?.images?.chosen) attempt(() => selectImage(propId, back.chosen));
    }
    paint();
    return true;
  }

  function setOn(v) {
    if (v && !on) active = firstFree(views(), src);
    on = v;
    drag = null;
    stopMove();
    paint();
  }

  function preset(kind) {
    if (!natural) return;
    change(presetViews(kind, src, natural));
    on = true;
    active = firstFree(views(), src);
    paint();
  }

  // ---- drawing a slice ------------------------------------------------------------------------------
  // The whole layer takes the press while slicing, so a slice may begin in the empty room beside the picture: the
  // point is brought to the picture's nearest edge (toImage). The slices' own buttons keep their presses.
  const at = (e) => toImage(e.clientX, e.clientY, svg.getBoundingClientRect(), natural);
  /** The shortest side a slice is drawn with, in the picture's pixels: MIN_BOX_PX on screen, and what the hub takes. */
  const least = () => Math.max(MIN_SIDE, Math.ceil(MIN_BOX_PX / fitRect(svg.getBoundingClientRect(), natural).scale));

  el.addEventListener('pointerdown', (e) => {
    if (!on || !natural || e.button !== 0 || moving) return;   // a slice being moved: no new one meanwhile
    if (e.target !== el && !svg.contains(e.target)) return;    // a chip, a move button
    e.preventDefault();
    el.setPointerCapture(e.pointerId);
    const a = at(e);
    drag = { id: e.pointerId, a, b: a, x: e.clientX, y: e.clientY, far: false, square: e.shiftKey };
  });
  el.addEventListener('pointermove', (e) => {
    if (drag?.id !== e.pointerId) return;
    drag.b = at(e);
    drag.square = e.shiftKey;
    drag.far ||= Math.hypot(e.clientX - drag.x, e.clientY - drag.y) > DRAG_PX;
    paintDraft();
  });
  el.addEventListener('pointerup', (e) => {
    if (drag?.id !== e.pointerId) return;
    const d = drag;
    drag = null;
    paintDraft();
    const rect = d.far ? dragRect(d.a, at(e), natural, e.shiftKey, least()) : null;
    if (usable(rect)) change(putView(views(), { view: active, src, rect }), true);
  });
  const stopDrag = (e) => {
    if (drag?.id !== e.pointerId) return;
    drag = null;
    paintDraft();
  };
  el.addEventListener('pointercancel', stopDrag);
  el.addEventListener('lostpointercapture', stopDrag);

  function paintDraft() {
    const rect = drag?.far ? dragRect(drag.a, drag.b, natural, drag.square, least()) : null;
    draft.setAttribute('display', rect ? 'inline' : 'none');
    if (!rect) return;
    const [x, y, w, hh] = rect;
    Object.entries({ x, y, width: w, height: hh }).forEach(([k, v]) => draft.setAttribute(k, v));
    draft.style.setProperty('--view', VIEWS[active].color);
  }

  // ---- a slice's own buttons: its chip, move, remove -----------------------------------------------------
  /**
   * The slice `entry` as the views have it now (an undo meanwhile may have changed it), or undefined. Of two
   * slices at one place (one moved onto the other), the one of its view.
   */
  const current = (entry) => {
    const there = views().filter((v) => v.src === entry.src && String(v.rect) === String(entry.rect));
    return there.find((v) => v.view === entry.view) || there[0];
  };

  function removeSlice(entry) {
    const now = current(entry);
    if (now) change(dropView(views(), now.view));
  }

  /** Where a slice's buttons and outline are: at `rect` ([x, y, w, h]). */
  function place(shape, ctl, [x, y, w, hh]) {
    Object.entries({ x, y, width: w, height: hh }).forEach(([k, v]) => shape.setAttribute(k, v));
    ctl.style.setProperty('--x', `${(x / natural.w) * 100}%`);
    ctl.style.setProperty('--y', `${((y + hh) / natural.h) * 100}%`);
  }

  /**
   * The move button of the slice `entry`: pressed and dragged, it carries the whole slice over the picture
   * (moveRect: the same size, inside the picture); released, the slice is saved there, one step for Ctrl+Z.
   */
  function moveButton(entry, shape, ctl) {
    const btn = h('button', { type: 'button', class: 'sl-move', title: 'Move this slice: press and drag' }, ownIcon('move'));
    btn.addEventListener('pointerdown', (e) => {
      if (!natural || e.button !== 0 || moving || drag) return;
      e.preventDefault();
      btn.setPointerCapture(e.pointerId);
      moving = { id: e.pointerId, x: e.clientX, y: e.clientY, rect: entry.rect, btn };
      btn.classList.add('moving');
    });
    btn.addEventListener('pointermove', (e) => {
      if (moving?.id !== e.pointerId || moving.btn !== btn) return;
      const { scale } = fitRect(svg.getBoundingClientRect(), natural);
      moving.rect = moveRect(entry.rect, (e.clientX - moving.x) / scale, (e.clientY - moving.y) / scale, natural);
      place(shape, ctl, moving.rect);
    });
    btn.addEventListener('pointerup', (e) => {
      if (moving?.id !== e.pointerId || moving.btn !== btn) return;
      const { rect } = moving;
      moving = null;
      btn.classList.remove('moving');
      const now = current(entry);
      if (now) change(views().map((v) => (v === now ? { ...v, rect } : v)));
    });
    const lost = (e) => {
      if (moving?.id === e.pointerId && moving.btn === btn) stopMove();
    };
    btn.addEventListener('pointercancel', lost);
    btn.addEventListener('lostpointercapture', lost);
    return btn;
  }

  /** The move in progress is given up: the slices are drawn again where they are. */
  function stopMove() {
    if (!moving) return;
    moving = null;
    drawn = null;
    paint();
  }

  // ---- keys: Ctrl+Z undoes, Escape stops slicing (before the shell, which would deselect the prop) ---
  // Not while typing (a field's own Escape and Ctrl+Z) or while a menu is open (it has its own keys).
  function onKey(e) {
    if (!el.isConnected || el.hidden) return;
    if (e.key === 'Shift' && drag) {
      drag.square = e.type === 'keydown';
      paintDraft();
      return;
    }
    if (e.type !== 'keydown' || dialogOpen() || menuOpen() || isTyping()) return;
    if (e.key === 'Escape' && on) {
      e.preventDefault();
      if (drag) {
        drag = null;
        paintDraft();
      } else if (moving) {
        stopMove();
      } else {
        setOn(false);
      }
    } else if (isUndoKey(e) && isNewest(past.newest)) {
      if (undo()) e.preventDefault();
    }
  }
  document.addEventListener('keydown', onKey, true);
  document.addEventListener('keyup', onKey, true);

  // ---- painting -----------------------------------------------------------------------------------------
  /**
   * A slice's chip was clicked: its view can be changed (swapping with the slice that had it) or removed.
   * The slice is looked up again when an item is picked: an undo meanwhile may have moved it.
   */
  function chipMenu(chip, entry) {
    const act = (fn) => () => {
      const now = current(entry);
      if (now) change(fn(now));
    };
    popupMenu(chip, [
      ...SHEET_ORDER.map((v) => ({
        label: VIEWS[v].label, title: VIEWS[v].title, current: v === entry.view,
        lead: h('span', { class: 'sl-swatch', style: `--view: ${VIEWS[v].color}` }),
        onClick: act((now) => relabel(views(), now.view, v)),
      })),
      'sep',
      { label: 'Remove this slice', icon: 'trash', danger: true, onClick: act((now) => dropView(views(), now.view)) },
    ], { className: 'sl-menu' });
  }

  function paintShapes(mine) {
    const key = natural ? `${natural.w}x${natural.h}|${viewsSignature(mine)}` : '';
    if (key === drawn) return;
    drawn = key;
    closeMenuIn(chips);   // a chip's menu goes with its chip
    moving = null;        // and a move with its button
    shapes.replaceChildren();
    chips.replaceChildren();
    if (!natural) return;
    svg.setAttribute('viewBox', `0 0 ${natural.w} ${natural.h}`);
    chips.style.setProperty('--ar', `${natural.w} / ${natural.h}`);
    for (const v of mine) {
      const info = VIEWS[v.view];
      if (!info) continue;
      const shape = svgEl('rect', { class: `sl-rect${v.rect ? '' : ' whole'}` });
      shape.style.setProperty('--view', info.color);
      shapes.append(shape);
      // In the slice's bottom-left corner: the chip; above it the move button (a slice that is the whole
      // picture has nowhere to go: none); to its right the one that removes the slice. The last two show
      // while slicing (stages-images.css).
      const chip = h('button', {
        type: 'button', class: 'sl-chip', title: `${info.title}. Click to change it or remove it.`,
        onclick: () => chipMenu(chip, v),
      }, info.label);
      const ctl = h('div', { class: 'sl-ctl', style: `--view: ${info.color}` });
      const del = h('button', {
        type: 'button', class: 'sl-del', title: 'Remove this slice', onclick: () => removeSlice(v),
      }, icon('trash'));
      ctl.append(...[v.rect && moveButton(v, shape, ctl), chip, del].filter(Boolean));
      place(shape, ctl, v.rect || [0, 0, natural.w, natural.h]);
      chips.append(ctl);
    }
  }

  function paint() {
    const all = views();
    const mine = src ? all.filter((v) => v.src === src) : [];
    el.hidden = !natural || busy;
    // The row is in the header while a picture shows and none is being made; it stays while the next picture
    // loads, so the header keeps its height.
    if (toolbar && inToolbar !== (!!img && !busy)) {
      inToolbar = !inToolbar;
      toolbar.replaceChildren(...(inToolbar ? [bar] : []));
    }
    el.classList.toggle('on', on);
    toggle.classList.toggle('on', on);
    toggle.setAttribute('aria-pressed', String(on));
    stack.hidden = !on;
    clear.hidden = !mine.length;
    // The views count while this picture (the chosen one) has a slice: else it is made into 3D alone.
    const used = mine.length > 0;
    const counting = effectiveViews(getProp(propId)?.images);
    hint.hidden = !on || !needsFront(counting);
    const mv = store.status?.multiview;
    mvHint.hidden = !on || needsFront(counting) || counting.length < 2 || !mv || !!mv.available;
    for (const [i, b] of viewBtns.entries()) {
      const v = SHEET_ORDER[i];
      const has = all.find((x) => x.view === v);
      b.classList.toggle('on', v === active);
      b.setAttribute('aria-checked', String(v === active));
      b.classList.toggle('has', !!has);
      b.classList.toggle('elsewhere', !!has && has.src !== src);
      b.title = `${VIEWS[v].title}${has && has.src !== src
        ? ` · its slice is on another picture${used ? '' : ', used once this picture has a slice too'}` : ''}`;
    }
    paintShapes(mine);
    paintDraft();
  }

  // ---- "Slice into views" glows once after a new picture shows, so it is noticed ----------------------
  function glow() {
    glowedFor = src;
    regenerated = false;
    clearTimeout(glowTimer);
    if (on) return;   // slicing goes on over the next picture
    // No movement, so gentle motion keeps it too.
    glowTimer = setTimeout(() => toggle.animate([
      { boxShadow: '0 0 0 0 rgba(139, 125, 255, 0)' },
      { boxShadow: '0 0 0 5px rgba(139, 125, 255, 0.45), 0 0 22px rgba(139, 125, 255, 0.55)', offset: 0.3 },
      { boxShadow: '0 0 0 0 rgba(139, 125, 255, 0)' },
    ], { duration: 1400, easing: 'ease-out' }), GLOW_MS);
  }

  function loaded(which, path) {
    if (which !== img || !which.naturalWidth) return;
    natural = { w: which.naturalWidth, h: which.naturalHeight };
    paint();
    if (path !== glowedFor || regenerated) glow();
  }

  return {
    el,

    /** The workbench shows the <img> `picture` (null: no picture), of the prop's picture `path`. */
    picture(picture, path) {
      img = picture;
      if (!picture) {
        src = natural = glowedFor = null;
        clearTimeout(glowTimer);
        paint();
        return;
      }
      if (path !== src) {
        natural = null;                   // (the same picture again, under a new URL, keeps its size meanwhile)
        if (on) active = firstFree(views(), path);   // another picture while slicing: its own first view
      }
      src = path;
      paint();
      if (picture.complete && picture.naturalWidth) loaded(picture, path);
      else picture.addEventListener('load', () => loaded(picture, path), { once: true });
    },

    /**
     * Before an import of several pictures replaces the views: `views` and the `chosen` picture it had, so
     * Ctrl+Z brings them back.
     */
    remember(before, chosen) {
      past.record({ views: before, chosen });
    },

    /** On every render: `busy` while a generation of the prop's pictures is queued or runs. */
    update({ busy: working = false } = {}) {
      if (busy && !working) regenerated = true;
      busy = working;
      if (busy && drag) drag = null;
      if (busy) stopMove();
      paint();
    },

    destroy() {
      clearTimeout(glowTimer);
      clearTimeout(copiedTimer);
      if (inToolbar) toolbar.replaceChildren();
      document.removeEventListener('keydown', onKey, true);
      document.removeEventListener('keyup', onKey, true);
      dropUndo();
    },
  };
}
