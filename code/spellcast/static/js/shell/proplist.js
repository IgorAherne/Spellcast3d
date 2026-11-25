// The props list in the props column: one tile per prop (not the draft), in project order. A tile is
// the prop's name over its picture in a rounded rectangle, and the status it has in the stage on
// screen as a small, quiet overlay: a check (done here; never in Images, where every prop has its
// picture), a clock (queued), an orange progress bar and a pulse (being made), the same bar standing still
// (begun, waiting for its next step), a red "!" (failed), an amber dot (made from older input), a hollow
// dot (ready to review), dimmed (not reachable yet). A stage view may put a box to tick beside each tile
// (view.tileCheck: Omni-Texture's, ticked for the props chosen there): it stands right of the tile, over the
// workbench's edge, and a click on it goes to the view alone (handlers.check) - the prop is not selected. A view
// may also put a row of small number boxes on each tile's picture, along its bottom edge (view.tileSizes: Retopo's sizes): what is
// typed there goes to the view alone (handlers.sizes), and neither selects the prop nor reaches the page's keys. A prop with
// several views shows its picture (its Front view) as the top card of a small fanned deck; hovering it, or
// reaching it with the keyboard, shows its views next to it, each with its side. Hovering a tile says its
// status (hovering its check says what the check means); a click selects it and the workbench shows it,
// the pencil after its name (or a double click on the name) renames it, the bin on its right deletes it (asked first;
// Ctrl+Z in the stage it was deleted in brings it back).
// Tiles are kept per prop and only touched where something changed, so a long list stays smooth; their
// pictures are the smallest there are (cropped.propPicture: the hub's thumbnail while it shows the prop's
// picture, else that picture itself, cut to its Front slice), loaded as they scroll into view.

import { fileUrl } from '../api.js';
import { h, isTextField, plural } from '../dom.js';
import { icon } from '../icons.js';
import { bustFor, projectId } from '../store.js';
import { deleteProp, renameProp, restoreProp } from '../actions.js';
import { attempt, confirmDialog, dialogOpen } from '../ui.js';
import { isNewest, isUndoKey, undoList, undoStamp } from '../undo.js';
import { croppedPicture, propPicture } from '../cropped.js';
import { VIEW_ORDER, effectiveViews, tileable } from '../stages/rules.js';
import { VIEWS } from '../stages/slicer.js';
import { SWIPE, fullMotion, pop, ring, settle, slideIn } from './anim.js';
import { hideTip, menuOpen, retip, showTip, tipShown } from './popups.js';

// idle: nothing done here yet and nothing queued; waiting: queued (or waiting for its tool to install);
// resting: its job has begun it and takes its other props through the same step before this one goes on;
// review: results the artist still has to pick from (Retopo).
const STATES = ['idle', 'waiting', 'processing', 'resting', 'done', 'review', 'stale', 'error', 'unreachable'];
const STATE_WORDS = {
  idle: 'Not started', waiting: 'Queued', processing: 'Working…', done: 'Done', review: 'Ready to review',
  stale: 'Needs redoing', error: 'Failed', unreachable: 'Not ready yet',
};
const POP_STAGGER = 45;   // ms between the checks popping in after a stage change
const POP_MAX = 12;       // at most this many are animated; the rest simply appear
const DONE_TIP = 'Ready for the next stage';

/**
 * The views a hovered deck shows next to it: those of a multi-view prop (deckOf), front, left, back, right,
 * each {view, label, url, crop: its slice of that picture, or null for all of it}; [] for a prop with one picture.
 */
export function tileViews(prop, pid = projectId()) {
  if (!deckOf(prop)) return [];
  const views = effectiveViews(prop.images);
  return VIEW_ORDER.map((side) => views.find((v) => v.view === side)).filter(Boolean).map((v) => ({
    view: v.view, label: VIEWS[v.view]?.short || v.view,
    url: fileUrl(pid, prop.id, v.src, bustFor('images', prop.id)), crop: Array.isArray(v.rect) ? v.rect : null,
  }));
}

/** The overlay a tile shows for a status in `stage`: the check never shows in Images; a prop being made has its bar. */
export function badgeOf(stage, state) {
  if (!STATES.includes(state) || ['idle', 'unreachable', 'processing', 'resting'].includes(state)) return '';
  return state === 'done' && stage === 'images' ? '' : state;
}

/**
 * The thin progress bar of a tile for a status ({state, fraction}): how far its prop is (0..1), 'unknown' (the
 * bar slides) or '' (no bar). A prop being made has one; so has one that waits for its next step, at the place
 * it got to - never sliding: nothing happens to it meanwhile.
 */
export function barOf(st) {
  const far = typeof st?.fraction === 'number' ? Math.min(1, Math.max(0, st.fraction)) : null;
  if (st?.state === 'processing') return far ?? 'unknown';
  return st?.state === 'resting' ? far ?? 0 : '';
}

/**
 * How many views a multi-view prop has (its 3D model is made from all of them: rules.effectiveViews), 0 for
 * a prop with one picture: the tile of a multi-view prop is drawn as a deck.
 */
export function deckOf(prop) {
  const n = effectiveViews(prop?.images).length;
  return n >= 2 ? n : 0;
}

/**
 * What a number box on a tile keeps of what is typed in it (view.tileSizes: `max`, `decimals`): digits and, with
 * decimals, one separator - a dot or a comma, so ".3" and "0,25" can be typed -, as many digits before it as `max`
 * has and `decimals` after it. Whatever else is typed is dropped.
 */
export function boxText(text, { max, decimals = 0 }) {
  const whole = String(Math.trunc(max)).length;
  const kept = text.replace(/[^\d.,]/g, '');
  const at = decimals ? kept.search(/[.,]/) : -1;
  if (at < 0) return kept.replace(/\D/g, '').slice(0, whole);
  return kept.slice(0, at).slice(0, whole) + kept[at] + kept.slice(at + 1).replace(/\D/g, '').slice(0, decimals);
}

/** The number a box's text stands for (a comma is a decimal point: ",3" is 0.3), or null: none from `min` to `max`. */
export function boxNumber(text, { min, max }) {
  const n = /^\d*[.,]?\d*$/.test(text) && /\d/.test(text) ? Number(text.replace(',', '.')) : NaN;
  return n >= min && n <= max ? n : null;
}

/**
 * What the number boxes on a tile hold: their numbers (boxNumber), in the order of the boxes; an empty box, or one
 * that holds no such number yet ("0", "0.", "."), is left out.
 */
export function boxNumbers(texts, offer) {
  return texts.map((text) => boxNumber(text, offer)).filter((n) => n !== null);
}

/**
 * handlers: {select(id), pick(id)?, check(id)?, sizes(id, numbers)?}: pick is a click (or Enter / Space) on a tile,
 * select otherwise; check is a click on the box beside a tile; sizes, typing in the number boxes on it. Returns the
 * list: {el, list, scroller, tail, render(ctx, view, shownId), tile(id), reveal(id, smooth)}. `tail`: what follows
 * the last tile in the scrolling list (the column's "+").
 */
export function createPropList(handlers) {
  const list = h('div', { class: 'props-list', role: 'list', 'aria-label': 'Your props' });
  const empty = h('div', { class: 'props-empty' }, 'Your props appear here');
  const tail = h('div', { class: 'props-tail' });
  // The tiles and what follows them scroll together: the "+" is always right under the last prop.
  const scroller = h('div', { class: 'props-scroll' }, list, empty, tail);
  const el = h('div', { class: 'props' }, scroller);
  const tiles = new Map();   // prop id -> tile record
  const pick = (id) => (handlers.pick || handlers.select)(id);
  let shownProject = null;  // the project the tiles belong to
  let shownStage = null;    // the stage their statuses belong to

  function makeTile(prop) {
    // A picture that can't be loaded (moved away on disk) leaves the icon; a new one is tried when it changes.
    const picture = croppedPicture({
      class: 'tile-img', ratio: 4 / 3, lazy: true,
      onError: () => {
        if (picture.el.parentNode !== pic) return;
        picture.el.replaceWith(holder);
        if (!t.placeholder) {
          t.placeholder = 'image';
          holder.replaceChildren(icon('image'));
        }
      },
    });
    const holder = h('span', { class: 'tile-ph', 'aria-hidden': 'true' });
    const barFill = h('span', { class: 'tile-bar-fill' });
    const bar = h('span', { class: 'tile-bar', 'aria-hidden': 'true' }, barFill);
    const badge = h('span', {
      class: 'tile-badge', 'aria-hidden': 'true',
      // The check says what it means (the other marks' words are the tile's own tooltip).
      // (A deck's views wait meanwhile: they would lie over these words.)
      onmouseenter: () => {
        if (t.badgeKind !== 'done') return;
        hidePeek();
        showTip(badge, DONE_TIP);
      },
      onmouseleave: () => {
        if (!tipShown(badge)) return;
        hideTip(badge);
        peekNow(t);
      },
    });
    // What the prop is, when it is no object: a tileable texture (rules.tileable), in every stage.
    const tag = h('span', { class: 'tile-tag', hidden: true }, 'tileable');
    const pic = h('div', { class: 'tile-pic' }, holder, tag, badge, bar);
    // A double click on the name renames, as the pencil after it does (the click before it selected the prop).
    const name = h('div', { class: 'tile-name', ondblclick: (e) => { e.stopPropagation(); startRename(t); } });
    const edit = h('button', {
      type: 'button', class: 'tile-edit', title: 'Rename this prop', 'aria-label': 'Rename',
      onclick: (e) => { e.stopPropagation(); startRename(t); },
    }, icon('pencil'));
    const more = h('button', {
      type: 'button', class: 'tile-more', title: 'Delete this prop', 'aria-label': 'Delete',
      onclick: (e) => { e.stopPropagation(); deleteTile(t); },
    }, icon('trash'));
    // The stage's box to tick, right of the tile (hidden unless the view offers one: render).
    const check = h('button', {
      type: 'button', class: 'tile-check', role: 'checkbox', hidden: true,
      onclick: (e) => { e.stopPropagation(); hidePeek(); handlers.check?.(t.id); },
      ondblclick: (e) => e.stopPropagation(),
      onmouseenter: (e) => { e.stopPropagation(); hideTip(pic); hidePeek(); },
    }, icon('check'));
    // The view's number boxes on the picture (none until the view offers them: paintSizes). Typing keeps to what
    // a number can be written with (boxText); a click, a double click or a key in them stays theirs.
    const sizesUnit = h('span', { class: 'tile-sizes-unit' });
    const sizes = h('div', {
      class: 'tile-sizes', hidden: true,
      onclick: (e) => e.stopPropagation(), ondblclick: (e) => e.stopPropagation(),
      onmouseenter: () => { hideTip(pic); hidePeek(); },
    }, sizesUnit);
    const typed = () => {
      for (const input of t.numbers) input.value = boxText(input.value, t.sizesOffer);
      handlers.sizes?.(t.id, boxNumbers(t.numbers.map((input) => input.value), t.sizesOffer));
    };
    const numberBox = () => h('input', {
      class: 'tile-size', type: 'text', autocomplete: 'off', spellcheck: 'false',
      oninput: typed,
      onkeydown: (e) => {
        e.stopPropagation();
        if (e.key === 'Enter' || e.key === 'Escape') e.target.blur();
      },
      // What the boxes say once the artist is done with one: the numbers kept, in order.
      onblur: () => paintSizes(t, t.sizesOffer, true),
    });
    const tileEl = h('div', {
      class: 'tile', role: 'listitem', tabindex: -1, 'data-id': prop.id,
      onclick: () => { hidePeek(); pick(t.id); },
      onkeydown: (e) => {
        if ((e.key === 'Enter' || e.key === ' ') && e.target === tileEl) { e.preventDefault(); pick(t.id); }
      },
      onmouseenter: () => { showTip(pic, t.tip); peekNow(t); },
      onmouseleave: () => { hideTip(pic); hidePeek(); },
      // Reached with the keyboard (not by a click, which says where the pointer is anyway).
      onfocus: () => { if (tileEl.matches(':focus-visible')) peekNow(t); },
      onblur: () => hidePeek(),
      // The name (with its pencil and its bin) over the picture: read first, like a card's title.
    }, h('div', { class: 'tile-head' }, name, edit, more), pic, sizes, check);
    const t = {
      id: prop.id, el: tileEl, pic, picture, holder, tag, badge, bar, barFill, name, more, check, sizes, sizesUnit, numberBox,
      src: undefined, placeholder: '', views: 0, deck: [], deckKey: '', state: undefined, badgeKind: '', fraction: '', label: '', ticked: '',
      tip: '', propName: '', renaming: false, selected: null, shown: null, tabStop: false,
      numbers: [], sizesOffer: null, sized: '',
    };
    return t;
  }

  /**
   * The view's number boxes on the tile (view.tileSizes: {values, count, min, max, decimals, unit, title}), or none
   * (null): `count` boxes for numbers from `min` to `max` with up to `decimals` decimals, holding `values` in order
   * (written with a dot), `unit` after them. While one of them is typed in, none is rewritten: the values are kept
   * in order without gaps, so a box emptied on the way to another number ("2" -> "0" -> "0.3") would pull its
   * neighbours' numbers into other boxes, and the next key would lose them. A box that says its value already keeps
   * its text ("0,3" is not made "0.3"). Touched only when something changed, unless `again` (one was left: they
   * show the numbers kept).
   */
  function paintSizes(t, offer, again = false) {
    const key = offer ? [offer.values.join(), offer.count, offer.min, offer.max, offer.decimals, offer.unit, offer.title,
      t.propName].join('|') : '';
    if (key === t.sized && !again) return;
    t.sized = key;
    t.sizesOffer = offer;
    t.sizes.hidden = !offer;
    if (!offer) return;
    while (t.numbers.length < offer.count) {
      const input = t.numberBox();
      t.numbers.push(input);
      t.sizes.insertBefore(input, t.sizesUnit);
    }
    t.sizes.title = offer.title;
    t.sizesUnit.textContent = offer.unit || '';
    const decimals = offer.decimals || 0;
    const typing = !again && t.numbers.includes(document.activeElement);
    t.numbers.forEach((input, i) => {
      const value = offer.values[i] ?? null;
      input.hidden = i >= offer.count;
      input.maxLength = String(Math.trunc(offer.max)).length + (decimals ? 1 + decimals : 0);
      input.setAttribute('inputmode', decimals ? 'decimal' : 'numeric');
      input.setAttribute('aria-label', `${offer.title} (box ${i + 1}): ${t.propName}`);
      if (typing || input === document.activeElement) return;
      if (value === null || boxNumber(input.value, offer) !== value) input.value = value === null ? '' : String(value);
    });
  }

  /**
   * The prop's picture (cropped.propPicture: asked anew at every render, so it is the chosen picture's the
   * moment that changes), or an icon while it has none; touched only when it changed, and the picture before
   * stays until the new one has loaded (no flicker, no empty tile).
   */
  function paintPicture(t, prop) {
    const shot = propPicture(prop);
    const src = shot ? `${shot.url}|${shot.crop || ''}` : null;
    if (src === t.src) return;
    t.src = src;
    if (shot) {
      t.picture.el.classList.toggle('picture', !!prop.images?.chosen);   // not a 3D snapshot: its see-through parts show
      t.picture.update(shot.url, shot.crop);
      if (t.picture.el.parentNode !== t.pic) t.holder.replaceWith(t.picture.el);
      return;
    }
    const kind = prop.mesh?.glb ? 'cube' : 'image';
    if (t.placeholder !== kind) {
      t.placeholder = kind;
      t.holder.replaceChildren(icon(kind));
    }
    if (t.picture.el.parentNode === t.pic) t.picture.el.replaceWith(t.holder);
    t.picture.update(null);
  }

  // ---- the views of a deck, next to it while it is hovered -----------------------------------------------

  let peek = null;        // {el, note, t}: the views showing now, and the tile they belong to

  /** Takes the views away at once (the pointer left, a click, the list scrolled, a rename, a menu). */
  function hidePeek() {
    peek?.el.remove();
    peek = null;
  }

  /** Shows the views of the deck `t` the moment the pointer is on it; nothing for a prop with one picture. */
  function peekNow(t) {
    hidePeek();
    if (t.deck.length && !t.renaming && !menuOpen() && t.el.isConnected) showPeek(t);
  }

  /** What the tooltip of `t` says beyond "4 views" (its status), for under the views that take its place. */
  const peekNote = (t) => (t.tip !== plural(t.views, 'view') ? t.tip : '');

  /**
   * The views of `t` right of its tile, above everything like the tooltips (no scrolling list clips them),
   * and out of the pointer's way (it cannot be clicked or hovered: the workbench under it stays in reach).
   */
  function showPeek(t) {
    hideTip(t.pic);
    const note = h('div', { class: 'tile-views-note' }, peekNote(t));
    note.hidden = !note.textContent;
    const el = h('div', { class: 'tile-views', 'aria-hidden': 'true' },
      h('div', { class: 'tile-views-row' }, t.deck.map((v) => {
        const shot = croppedPicture({ class: 'tile-view-pic', fit: 'contain' });
        shot.update(v.url, v.crop);
        return h('div', { class: 'tile-view' }, shot.el, h('span', null, v.label));
      })), note);
    document.body.append(el);
    const r = t.el.getBoundingClientRect();
    const p = t.pic.getBoundingClientRect();
    const w = el.offsetWidth;
    const hh = el.offsetHeight;
    el.style.left = `${Math.max(8, Math.min(r.right + 10, innerWidth - w - 8))}px`;
    el.style.top = `${Math.max(8, Math.min(p.top + p.height / 2 - hh / 2, innerHeight - hh - 8))}px`;
    peek = { el, note, t };
  }

  function paintStatus(t, st, stage) {
    const state = STATES.includes(st?.state) ? st.state : null;
    const before = t.state;
    if (state !== before) {
      for (const s of STATES) t.el.classList.toggle(`st-${s}`, s === state);
      t.state = state;
    }
    const kind = badgeOf(stage, state);
    if (kind !== t.badgeKind) {
      t.badgeKind = kind;
      if (kind) t.el.dataset.badge = kind;
      else delete t.el.dataset.badge;
      t.badge.replaceChildren(kind === 'done' ? icon('check') : kind === 'waiting' ? icon('clock') : kind === 'error' ? '!' : '');
      if (kind !== 'done') hideTip(t.badge);
    }
    const f = barOf(st);
    if (f !== t.fraction) {
      t.fraction = f;
      t.bar.classList.toggle('indeterminate', f === 'unknown');
      t.barFill.style.transform = typeof f === 'number' ? `scaleX(${f})` : '';
    }
    const words = st?.message || STATE_WORDS[state] || '';
    const deck = t.views ? plural(t.views, 'view') : '';   // "4 views": why the tile is a deck
    const label = [t.propName, deck, words !== t.propName ? words : ''].filter(Boolean).join(' · ');
    if (label !== t.label) {
      t.label = label;
      t.el.setAttribute('aria-label', label);
    }
    // The tooltip only adds what the tile does not say: what is wrong or still to come, and a deck's views.
    const tip = state && state !== 'done' && !(stage === 'images' && state === 'idle') ? label : deck;
    if (tip !== t.tip) {
      t.tip = tip;
      retip(t.pic, tip);
      if (peek?.t === t) {
        peek.note.textContent = peekNote(t);
        peek.note.hidden = !peek.note.textContent;
      }
    }
  }

  /** The name becomes a field: Enter (or a click elsewhere) saves it, Escape keeps the old one, as an empty name does. */
  function startRename(t) {
    if (t.renaming) return;
    hidePeek();
    t.renaming = true;
    t.el.classList.add('renaming');
    const input = h('input', { class: 'tile-rename', type: 'text', value: t.propName, maxlength: 80, 'aria-label': 'Prop name' });
    let done = false;
    // Enter / Escape give the focus back to the tile; a click elsewhere keeps it where it went.
    const finish = async (save, refocus = false) => {
      if (done) return;
      done = true;
      t.renaming = false;
      t.el.classList.remove('renaming');
      const value = input.value.trim();
      input.replaceWith(t.name);
      if (refocus) t.el.focus({ preventScroll: true });
      if (save && value && value !== t.propName) {
        t.name.textContent = value;
        await attempt(() => renameProp(t.id, value));
      }
    };
    input.addEventListener('keydown', (e) => {
      e.stopPropagation();
      if (e.key === 'Enter') finish(true, true);
      if (e.key === 'Escape') finish(false, true);
    });
    input.addEventListener('blur', () => finish(true));
    input.addEventListener('click', (e) => e.stopPropagation());
    input.addEventListener('dblclick', (e) => e.stopPropagation());
    t.name.replaceWith(input);
    input.focus();
    input.select();
  }

  // Props deleted here, the last one last: Ctrl+Z brings back the newest one deleted in the stage on screen, when
  // nothing done in that stage since is still to undo (undo.js).
  const deletions = [];   // {project, stage, name: its folder in the project's "deleted" one, at: its undo stamp}
  const lastDeletion = () => deletions.findLast((d) => d.project === shownProject && d.stage === shownStage);
  undoList(() => lastDeletion()?.at);

  /** The bin on a tile: the prop goes after a question (its files are kept in the project's "deleted" folder). */
  async function deleteTile(t) {
    hideTip();
    hidePeek();
    const ok = await confirmDialog({
      title: `Delete “${t.propName}”?`,
      message: 'The prop and everything made for it are moved to the project’s “deleted” folder. Ctrl+Z brings it back.',
      ok: 'Delete prop', danger: true,
    });
    if (!ok) return;
    const project = shownProject;
    const stage = shownStage;
    const name = await attempt(() => deleteProp(t.id));
    if (name) deletions.push({ project, stage, name, at: undoStamp() });
  }

  globalThis.addEventListener?.('keydown', (e) => {
    const last = lastDeletion();
    if (!last || !isUndoKey(e) || !isNewest(last.at) || dialogOpen() || isTextField()) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    deletions.splice(deletions.indexOf(last), 1);
    attempt(() => restoreProp(last.name));
  }, true);

  /** Forgets every tile (another project is open): its list is drawn afresh, without animations. */
  function clear() {
    hidePeek();
    for (const t of tiles.values()) t.el.remove();
    tiles.clear();
    scroller.scrollTop = 0;
  }

  scroller.addEventListener('scroll', hidePeek, { passive: true });

  /**
   * Draws the props of `ctx` with their status in `view` (the stage on screen). shownId: the prop the
   * workbench shows although none is selected (a later stage follows the work): it gets a lighter ring.
   */
  function render(ctx, view, shownId = null) {
    const pid = ctx.project?.id ?? null;
    const opening = pid !== shownProject;
    if (opening) {
      clear();
      shownProject = pid;
    }
    const stage = ctx.stageId;
    const restage = !opening && shownStage !== null && stage !== shownStage;
    if (restage) hidePeek();   // the page swipes to another stage: nothing floats over it
    shownStage = stage;
    el.dataset.stage = stage;
    const props = ctx.boxes;
    empty.hidden = props.length > 0;
    const ids = new Set(props.map((p) => p.id));
    for (const [id, t] of tiles) {
      if (ids.has(id)) continue;
      tiles.delete(id);
      if (peek?.t === t) hidePeek();
      settle([t.el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: fullMotion() ? 180 : 80, fill: 'forwards' })], 600)
        .then(() => t.el.remove());
    }
    const earned = [];
    let checks = false;
    // One tab stop for the whole list (the prop in use, else the first): the arrow keys walk the props.
    const stop = ctx.selectedId || (shownId && ids.has(shownId) ? shownId : props[0]?.id);
    let prev = null;
    for (const prop of props) {
      let t = tiles.get(prop.id);
      const fresh = !t;
      if (fresh) {
        t = makeTile(prop);
        tiles.set(prop.id, t);
      }
      const expected = prev ? prev.el.nextElementSibling : list.firstElementChild;
      if (expected !== t.el) list.insertBefore(t.el, expected);
      prev = t;
      if (fresh && !opening) slideIn(t.el);
      const propName = prop.name || 'Prop';
      if (propName !== t.propName) {
        t.propName = propName;
        if (!t.renaming) t.name.textContent = propName;
        t.name.title = propName;
      }
      const selected = ctx.selectedId === prop.id;
      if (selected !== t.selected) {
        t.selected = selected;
        t.el.classList.toggle('selected', selected);
        t.el.setAttribute('aria-current', String(selected));
      }
      const shown = !ctx.selectedId && shownId === prop.id;
      if (shown !== t.shown) {
        t.shown = shown;
        t.el.classList.toggle('shown', shown);
      }
      if ((stop === prop.id) !== t.tabStop) {
        t.tabStop = stop === prop.id;
        t.el.tabIndex = t.tabStop ? 0 : -1;
      }
      paintPicture(t, prop);
      t.tag.hidden = !tileable(prop);
      const views = deckOf(prop);
      if (views !== t.views) {
        t.views = views;
        t.el.classList.toggle('multi', views > 0);
      }
      // What hovering the deck shows; views that changed while they show are taken away (the next hover has the new ones).
      const deck = views ? tileViews(prop) : [];
      const deckKey = deck.map((v) => `${v.view}|${v.url}|${v.crop || ''}`).join();
      if (deckKey !== t.deckKey) {
        t.deck = deck;
        t.deckKey = deckKey;
        if (peek?.t === t) hidePeek();
      }
      let st = null;
      let box = null;     // {on, title}: the view's box beside this tile, if it has one
      let offer = null;   // the view's number boxes on it, if it has them (paintSizes)
      try {
        st = view.status(prop, ctx);
        box = view.tileCheck?.(prop, ctx) || null;
        offer = view.tileSizes?.(prop, ctx) || null;
      } catch (e) {
        console.error(e);
      }
      paintSizes(t, offer);
      checks ||= !!box;
      const ticked = box ? `${box.on}|${box.title}` : '';
      if (ticked !== t.ticked) {
        t.ticked = ticked;
        t.check.hidden = !box;
        t.check.setAttribute('aria-checked', String(!!box?.on));
        t.check.title = box?.title || '';
        t.check.setAttribute('aria-label', box ? `${box.title}: ${propName}` : '');
      }
      const had = t.badgeKind;
      const known = t.state !== undefined;   // a new tile's check is not news
      paintStatus(t, st, stage);
      // A check earned while the artist looks (a job finished, a result accepted) pops in; after a
      // stage change, the checks of that stage pop in one after another.
      if (known && t.badgeKind === 'done' && had !== 'done') {
        if (restage) earned.push(t);
        else pop(t.badge);
      }
    }
    if (earned.length) popEarned(earned, ctx.moving);
    el.classList.toggle('checks', checks);   // the list makes room for the boxes beside it (shell.css)
  }

  /**
   * After a stage change, the checks the props have in the new stage pop in one after another (those in
   * view); `moving`: once the page has swiped there.
   */
  function popEarned(earned, moving = false) {
    if (!fullMotion()) return;
    const after = moving ? SWIPE.duration : 0;
    const top = scroller.scrollTop;
    const bottom = top + scroller.clientHeight;
    let i = 0;
    for (const t of earned) {
      const y = t.el.offsetTop;   // the scroller is the tiles' offset parent
      if (y + t.el.offsetHeight < top || y > bottom) continue;
      if (i >= POP_MAX) break;
      pop(t.badge, after + 120 + i++ * POP_STAGGER);
    }
  }

  const tileOf = (id) => tiles.get(id)?.el || null;

  return {
    el,
    list,
    scroller,
    tail,
    render,
    tile: tileOf,
    /** A ring around the tile, twice: what the prop is has just changed. */
    pulse(id) {
      ring(tileOf(id), 2);
    },
    /** Scrolls the list so the tile shows. */
    reveal(id, smooth = false) {
      const tile = tileOf(id);
      if (!tile) return;
      const r = scroller.getBoundingClientRect();
      const b = tile.getBoundingClientRect();
      if (b.top < r.top || b.bottom > r.bottom) {
        scroller.scrollTo({ top: scroller.scrollTop + b.top - r.top - (r.height - b.height) / 2, behavior: smooth && fullMotion() ? 'smooth' : 'auto' });
      }
    },
  };
}
