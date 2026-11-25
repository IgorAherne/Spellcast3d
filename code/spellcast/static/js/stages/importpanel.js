// The Import tab of the Images workbench: one picture of the object (single view), or a picture per side
// (multi-view: Front, Back, "← Left" and "Right →": the object faces left in one, right in the other).
//
// Two areas, one under the other. "Single view": a drop zone and "Paste"; the picture becomes one of the prop's
// pictures and its chosen one, so the 3D model is made from it alone. "Multi-view": four optional slots, each
// with its own "Paste"; a filled slot is that picture as one of the prop's pictures plus a view of the whole of
// it for that side (images.views: {view, src, rect: null}, the same views the slicer draws), and the chosen
// picture is then one of the views' (the front one when there is one), so the views are used
// (rules.effectiveViews). A slot also shows a view made another way (a slice of a sheet): cropped to it.
// The area the 3D model is made from now says "In use". Ctrl+V does nothing here (which slot would it fill?):
// every zone and slot has its own "Paste".

import { dropTarget, h, pickFile } from '../dom.js';
import { icon } from '../icons.js';
import { getProp, jobFor } from '../store.js';
import { queueViews, selectImage, uploadImages } from '../actions.js';
import { spinner } from '../controls.js';
import { attempt, toast } from '../ui.js';
import { boxFile } from './common.js';
import { VIEW_ORDER, effectiveViews } from './rules.js';

const IMAGE_TYPES = 'image/png,image/jpeg,image/webp';
const NOT_AN_IMAGE = 'That is not an image. Use a PNG, JPG or WebP file.';
const NOTHING_COPIED = 'There is no picture on the clipboard: copy one first.';
const NO_CLIPBOARD = 'The browser did not let Spellcast3D read the clipboard. Drop the picture here or click to choose its file.';
const WHERE = 'Drop the picture on Single view, or on the Multi-view slot of the side it shows.';
const UPLOADING = 'Wait a moment: the last picture is still being added.';
const MAKING = 'This prop’s pictures are being made. Import yours once they are here.';
const SHOW_SOON_MS = 300;   // a quick upload shows nothing; a slow one its progress, after a moment

// ---- the rules of the panel (plain functions: code/tests/web/importpanel.test.mjs) ---------------------

/**
 * Multi-view's four slots, in the order they show. `view` is the view's own name (rules.VIEW_ORDER; the
 * slicer calls left "← Left" and right "Right →" too: the arrow points where the object looks in the picture).
 */
export const SLOTS = [
  { view: 'front', label: 'Front', hint: 'The 3D model faces it' },
  { view: 'back', label: 'Back', hint: 'The object from behind' },
  { view: 'left', label: '← Left', hint: 'The object faces left' },
  { view: 'right', label: 'Right →', hint: 'The object faces right' },
];

// An empty Left or Right slot shows which way the object looks in its picture: a head with its nose that way
// over an arrow (drawn facing left; mirrored for Right), so the two are not mixed up.
const FACING = '<circle cx="13.5" cy="8.5" r="5.5"/><path d="M8.3 6.9L4.5 9.6l4.2 1"/><path d="M11.4 7.2v.1"/>'
  + '<path d="M19.5 19.5h-15M8 16l-3.5 3.5L8 23"/>';

function facingIcon(view) {
  const tpl = document.createElement('template');
  tpl.innerHTML = '<svg class="icon ip-facing" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" '
    + 'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    + `<g${view === 'right' ? ' transform="translate(24 0) scale(-1 1)"' : ''}>${FACING}</g></svg>`;
  return tpl.content.firstChild;
}

const inOrder = (views) => [...views].sort((a, b) => VIEW_ORDER.indexOf(a.view) - VIEW_ORDER.indexOf(b.view));

/** The prop's views that can show: one of the four sides, on a picture the prop still has. */
function viewsOf(images) {
  const kept = Array.isArray(images?.candidates) ? images.candidates : null;
  return (images?.views || []).filter((v) => v && typeof v === 'object' && VIEW_ORDER.includes(v.view)
    && typeof v.src === 'string' && (!kept || kept.includes(v.src)));
}

/**
 * What each slot shows for a prop's pictures (`images`): [{view, label, hint, src, rect}] in SLOTS' order.
 * src: the picture of that side's view (null: the slot is empty); rect: the part of it the view is
 * ([x, y, w, h], a slice made in the slicer), null for the whole picture.
 */
export function slotsOf(images) {
  const views = viewsOf(images);
  return SLOTS.map((slot) => {
    const v = views.find((x) => x.view === slot.view);
    return { ...slot, src: v?.src ?? null, rect: Array.isArray(v?.rect) ? v.rect : null };
  });
}

/**
 * The area the prop's 3D model is made from now: 'multi' while two or more views are used
 * (rules.effectiveViews), 'single' for one picture, null while the prop has no picture.
 */
export const inUse = (images) => (!images?.chosen ? null : effectiveViews(images).length >= 2 ? 'multi' : 'single');

/**
 * The one picture the 3D model is made from, for Single view's zone: {src, rect} (rect: the part of it a
 * lone view marks, else null), or null while there is none or several views are used.
 */
export function singleOf(images) {
  const views = effectiveViews(images);
  if (!images?.chosen || views.length >= 2) return null;
  return { src: images.chosen, rect: Array.isArray(views[0]?.rect) ? views[0].rect : null };
}

/** The picture to choose so that `views` are used: the front one's, else `chosen` if it is one of theirs, else the first one's. */
const chosenFor = (views, chosen) => views.find((v) => v.view === 'front')?.src
  ?? (views.some((v) => v.src === chosen) ? chosen : views[0]?.src ?? chosen ?? null);

/**
 * The slot `view` filled with the whole picture `src` (one of the prop's pictures): {views, chosen} - the
 * prop's views with it instead of that side's earlier one, and the picture to choose so they are used.
 */
export function fillSlot(images, view, src) {
  const views = inOrder([...viewsOf(images).filter((v) => v.view !== view), { view, src, rect: null }]);
  return { views, chosen: chosenFor(views, src) };
}

/**
 * The slot `view` emptied: {views, chosen} - the prop's views without that side's (its picture stays one of
 * the prop's), and the picture to choose: while the views were used and some are left, one of theirs, so
 * they stay used; else the one chosen now.
 */
export function emptySlot(images, view) {
  const views = viewsOf(images).filter((v) => v.view !== view);
  const used = effectiveViews(images).length > 0;
  return { views, chosen: used && views.length ? chosenFor(views, images.chosen) : images?.chosen ?? null };
}

/**
 * The views wait: the prop has `count` of them (2 or more) but the chosen picture is none of theirs, so it
 * is made into 3D alone. {count, chosen: the picture to choose to use them}, or null.
 */
export function waiting(images) {
  const views = viewsOf(images);
  if (views.length < 2 || !images?.chosen || effectiveViews(images).length) return null;
  return { count: views.length, chosen: chosenFor(views, images.chosen) };
}

/** Several slots are filled and Front is not: the 3D generator needs a front view to face. */
export const needsFront = (images) => viewsOf(images).length >= 2 && !viewsOf(images).some((v) => v.view === 'front');

/** The picture an upload added: the one of `images.candidates` that `before` (the pictures until then) lacks, else the chosen one. */
export function newPicture(before, images) {
  const added = (images?.candidates || []).filter((c) => !before.includes(c));
  return added[added.length - 1] ?? images?.chosen ?? null;
}

/**
 * Where a whole picture of `natural` size ({w, h}) goes in a box that shows its part `rect` ([x, y, w, h])
 * alone: {width, height, left, top} in % of the box.
 */
export function cropBox(rect, natural) {
  const [x, y, w, hh] = rect;
  return { width: (natural.w / w) * 100, height: (natural.h / hh) * 100, left: (-x / w) * 100, top: (-y / hh) * 100 };
}

// ---- the panel -------------------------------------------------------------------------------------------

/**
 * navigator.clipboard.read(), asked once more when the browser's own "allow pasting?" question got in the way:
 * some browsers turn down the read that raised the question although the artist allowed it (Edge: the first
 * "Paste" said no, the second worked). While the question is open this waits for its answer; allowed, it asks
 * again. Anything else, or not allowed, fails as before.
 */
export async function readClipboardItems() {
  try {
    return await navigator.clipboard.read();
  } catch (e) {
    if (e?.name !== 'NotAllowedError') throw e;
    const status = await clipboardStatus();
    if (!status) throw e;
    if (status.state === 'prompt') await answered(status);
    if (status.state !== 'granted') throw e;
    return navigator.clipboard.read();
  }
}

/** The page's clipboard-read permission (a PermissionStatus); null where the browser does not tell. */
async function clipboardStatus() {
  try {
    return await navigator.permissions.query({ name: 'clipboard-read' });
  } catch {
    return null;
  }
}

/** Resolves once `status` changes (the artist answered the browser's question), or after a minute. */
const answered = (status) => new Promise((resolve) => {
  const timer = setTimeout(done, 60000);
  function done() {
    clearTimeout(timer);
    status.removeEventListener('change', done);
    resolve();
  }
  status.addEventListener('change', done);
});

/** The picture on the clipboard as a file, for "Paste" (the browser may ask to allow it); none: null. */
async function clipboardImage() {
  for (const item of await readClipboardItems()) {
    const type = item.types.find((t) => t.startsWith('image/'));
    // Named ".png": the hub names a new prop after its first file, and none after this one ("Prop 3", not "Pasted").
    if (type) return new File([await item.getType(type)], `.${type.split('/')[1]}`, { type });
  }
  return null;
}

/** A small picture of the prop's picture `src`, cropped to `rect` when it is a slice of it. */
function thumb(p, src, rect) {
  const img = h('img', { src: boxFile(p, src, 'images'), alt: '', draggable: 'false' });
  if (!rect) return h('span', { class: 'ip-thumb' }, img);
  const [, , w, hh] = rect;
  const frame = h('span', { class: 'ip-crop', style: `aspect-ratio: ${w} / ${hh}; ${w >= hh ? 'width' : 'height'}: 100%` }, img);
  img.addEventListener('load', () => {
    const c = cropBox(rect, { w: img.naturalWidth, h: img.naturalHeight });
    img.style.cssText = `width: ${c.width}%; height: ${c.height}%; left: ${c.left}%; top: ${c.top}%`;
    frame.classList.add('ready');
  });
  return h('span', { class: 'ip-thumb' }, frame);
}

/**
 * The Import tab's panel for the prop `propId`. Returns {el, update(), destroy()}: update() follows the
 * store (the workbench calls it on every change), destroy() lets go of its listeners.
 */
export function createImportPanel(propId) {
  let working = 0;       // pictures on their way to the hub, and what follows them (the views, the chosen one)
  let slow = false;      // ... for long enough to say so
  let showSoon = null;
  let shown = null;      // what the panel shows (update() repaints when it changes)
  let dead = false;

  const single = h('section', { class: 'ip-area ip-single', 'aria-label': 'Single view' });
  const multi = h('section', { class: 'ip-area ip-multi', 'aria-label': 'Multi-view' });
  const statusText = h('span');
  const statusIcon = h('span', { class: 'ip-status-icon' });
  const status = h('div', { class: 'ip-status', role: 'status', hidden: true }, statusIcon, statusText);
  const el = h('div', { class: 'ip', 'data-prop': propId }, single, multi, status);
  // A picture dropped between the zones: said, not guessed (and not opened by the browser in the page's place).
  dropTarget(el, () => toast(WHERE, 'warn'), () => toast(NOT_AN_IMAGE, 'warn'));

  const images = () => getProp(propId)?.images || {};
  const making = () => !!jobFor('images', propId);

  /** Nothing is on its way and the prop's pictures are not being made; else says why not. */
  function free() {
    const why = working ? UPLOADING : making() ? MAKING : '';
    if (why) toast(why, 'warn');
    return !why;
  }

  /** Runs one change of the prop's pictures: the panel waits for it, a failure shows as a toast. */
  async function work(fn) {
    if (dead || !free()) return;
    working += 1;
    statusText.textContent = 'Saving…';
    showSoon = setTimeout(() => { slow = true; update(); }, SHOW_SOON_MS);
    update();
    await attempt(fn);
    clearTimeout(showSoon);
    working -= 1;
    slow = false;
    update();
  }

  /** Adds `file` to the prop's pictures (the hub makes it the chosen one) -> its path. */
  async function upload(file) {
    const before = images().candidates || [];
    const progress = (f) => {
      statusText.textContent = f == null || f >= 1 ? 'Adding the picture…' : `Adding the picture… ${Math.round(f * 100)}%`;
    };
    progress(null);
    await uploadImages(propId, [file], progress);
    return newPicture(before, images());
  }

  /** Single view: `file` becomes the prop's chosen picture, made into 3D alone (the views, if any, wait). */
  const addSingle = (file) => work(() => upload(file));

  /** Multi-view: `file` becomes the picture of the side `view`. */
  const fill = (view, file) => work(async () => {
    const src = await upload(file);
    if (!src) return;
    const next = fillSlot(images(), view, src);
    queueViews(propId, next.views);
    if (next.chosen !== images().chosen) await selectImage(propId, next.chosen);
  });

  /** Multi-view: the side `view` has no picture any more (its picture stays among the prop's). */
  const empty = (view) => work(async () => {
    const next = emptySlot(images(), view);
    queueViews(propId, next.views);
    if (next.chosen && next.chosen !== images().chosen) await selectImage(propId, next.chosen);
  });

  async function choose(use) {
    const file = await pickFile(IMAGE_TYPES);
    if (!file) return;
    if (file.type.startsWith('image/')) use(file);
    else toast(NOT_AN_IMAGE, 'warn');
  }

  /** "Paste": the picture on the clipboard goes where a dropped one would (`use`). */
  async function paste(use) {
    let file = null;
    try {
      file = await clipboardImage();
    } catch (e) {
      // Not allowed, or a browser (or a page not on localhost) without clipboard reading
      toast(e?.name === 'NotAllowedError' || !navigator.clipboard?.read ? NO_CLIPBOARD : NOTHING_COPIED, 'warn');
      return;
    }
    if (file) use(file);
    else toast(NOTHING_COPIED, 'warn');
  }

  const pasteButton = (title, use, off) => h('button', {
    type: 'button', class: 'btn ghost small ip-paste', title, disabled: off, onclick: () => paste(use),
  }, 'Paste');

  const head = (name, on) => h('div', { class: 'ip-head' }, h('h3', null, name),
    on ? h('span', { class: 'ip-used', title: 'The 3D model is made from this' }, icon('check'), 'In use') : null);

  function singleArea(p, one, off) {
    const zone = h('button', {
      type: 'button', class: `ip-zone${one ? ' filled' : ''}`, disabled: off, onclick: () => choose(addSingle),
      title: 'One picture of the object: the 3D model is made from it alone',
    }, one ? thumb(p, one.src, one.rect) : icon('upload'),
    h('span', null, h('b', null, one ? 'Use another picture' : 'Choose a picture'),
      h('small', null, 'Drop it here or click · PNG, JPG or WebP')));
    const row = h('div', { class: 'ip-row' }, zone, pasteButton('Use the picture you copied', addSingle, off));
    dropTarget(row, addSingle, () => toast(NOT_AN_IMAGE, 'warn'));
    return row;
  }

  function slotCard(p, slot, off) {
    const filled = !!slot.src;
    const zone = h('button', {
      type: 'button', class: 'ip-slot-zone', disabled: off, onclick: () => choose((f) => fill(slot.view, f)),
      title: filled ? `Use another picture as the ${slot.label} view` : `Choose the ${slot.label} picture, or drop it here`,
      'aria-label': filled ? `${slot.label}: use another picture` : `${slot.label}: choose a picture`,
    }, filled ? thumb(p, slot.src, slot.rect)
      : [slot.view === 'left' || slot.view === 'right' ? facingIcon(slot.view) : icon('upload'), h('small', null, 'Drop or click')]);
    const card = h('div', { class: `ip-slot${filled ? ' filled' : ''}`, 'data-view': slot.view },
      zone,
      filled && slot.rect ? h('span', { class: 'ip-slice', title: 'A part of a bigger picture, marked with “Slice into views”' }, 'slice') : null,
      filled ? h('button', {
        type: 'button', class: 'ip-x', title: `Empty the ${slot.label} slot (the picture stays with the prop)`,
        'aria-label': `Empty the ${slot.label} slot`, disabled: off, onclick: () => empty(slot.view),
      }, icon('x')) : null,
      h('div', { class: 'ip-slot-foot' },
        h('span', { class: 'ip-slot-name' }, h('b', null, slot.label), h('small', null, slot.hint)),
        pasteButton(`Use the picture you copied as the ${slot.label} view`, (f) => fill(slot.view, f), off)));
    dropTarget(card, (f) => fill(slot.view, f), () => toast(NOT_AN_IMAGE, 'warn'));
    return card;
  }

  function update() {
    const p = getProp(propId);
    if (dead || !p) return;
    const img = p.images || {};
    const url = (src) => boxFile(p, src, 'images');
    const [one, slots, used, wait, job] = [singleOf(img), slotsOf(img), inUse(img), waiting(img), making()];
    const off = working > 0 || job;
    const key = JSON.stringify([off, slow, job, used, wait, needsFront(img), one && [url(one.src), one.rect],
      slots.map((s) => s.src && [url(s.src), s.rect])]);
    if (key === shown) return;
    shown = key;

    single.classList.toggle('on', used === 'single');
    single.replaceChildren(head('Single view', used === 'single'), singleArea(p, one, off),
      h('p', { class: 'hint' }, 'One picture of the object: the 3D model is made from it alone.'));

    multi.classList.toggle('on', used === 'multi');
    // (replaceChildren would print a missing note as the word "null": only the notes there are.)
    multi.replaceChildren(...[head('Multi-view', used === 'multi'),
      h('p', { class: 'hint' }, 'Front plus at least one more make a multi-view model. The object should look the same '
        + 'in all of them: same pose, same lighting.'),
      h('div', { class: 'ip-slots' }, slots.map((slot) => slotCard(p, slot, off))),
      needsFront(img) ? h('p', { class: 'ip-note warn' }, icon('alert'), 'Add the Front picture: the 3D model needs one to face.') : null,
      wait ? h('p', { class: 'ip-note' }, h('span', null, 'Not used now: the 3D model is made from the single picture.'),
        h('button', {
          type: 'button', class: 'btn ghost small', disabled: off, title: 'Make the 3D model from these views again',
          onclick: () => work(() => selectImage(propId, wait.chosen)),
        }, 'Use these views')) : null].filter(Boolean));

    status.hidden = !(slow || (job && !working));
    if (!working) statusText.textContent = job ? MAKING : '';
    statusIcon.replaceChildren(working ? spinner() : icon('info'));
  }

  update();
  return {
    el,
    update,
    destroy() {
      dead = true;
      clearTimeout(showSoon);
    },
  };
}
