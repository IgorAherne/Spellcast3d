// The painting tools of a viewport that shows a model to paint (paint3d.js paints): Projection's baked low poly, and
// Omni-Texture's result, which brings a tool of its own (stages/maskbar.js). In the viewport's bottom-right corner, on
// a glass, the brush size, "P" (a pen's pressure sizes the brush) and "F" (it sets the paint's opacity); rising from
// that glass's right end, the tools on a glass of their own, from the top down: Orbit (no brush: the left button
// orbits again), the clone stamp, the marker (any colour: the rainbow circle opens the palette) and the stage's own
// tool if it has one - with, to its left while it is up, an x that wipes what it painted. To the glass's left, in
// two lines, how to use the tool that is up. While a tool is up the left button paints; Alt + left, the middle and the
// right button still move the camera as everywhere (viewer3d.js dragAction), unless the tool is a stage's own that
// holds the view still (`still`). Shift + moving sideways over the model sizes the brush, as on the Images picture.
// Ctrl+Z / Ctrl+Y take strokes back and bring them back; the painting is saved into the model's files a moment after
// each change.
//
// What differs from stage to stage comes in createPaintBar's options (each may be left out):
//   save(binding, image) -> the prop as the hub has it after the save (or null): how a painting is saved (binding:
//     {project, propId, glb}; image: paint3d.js readAll's). By default into the bake's files (saveToBake).
//   patches: true when `save` sends readAll's patch (the texels painted alone): the models' sessions keep the files'
//     bytes to tell them (paint3d.js).
//   overlay: the stage's own tool, which paints the model's overlay (paint3d.js: one colour over the base colour, on
//     the page only, never saved): {button (its node at the column's foot: a click takes it up or puts it down; it is
//     ringed ("on") while up), colour ('#rrggbb'), hint (the first of the two lines while it is up: the second names
//     the keys, as for every tool), slot (a node at the glass's left end), up() / down() (it was taken up / put down),
//     wipe (what the x beside the button does, in words: its tooltip), lost (words added to what the artist is told
//     when the graphics card is reset: the overlay went with it), still (true: while it is up nothing moves the
//     camera - viewer3d.js holdView - and the hint's second line says so)}.
//   busy() -> true while the stage works on the model: nothing paints, no tool is taken up, Ctrl+Z / Ctrl+Y wait.
// A model that is not the stage's to paint yet is bound with `begin` (bind): no tool is up on it, and the first one
// asked for has the stage make it paintable (begin(): awaited; the stage then binds the model anew, without it) -
// that tool is up from then. Omni-Texture's: a prop's 3D model, before it has a result of its own to paint.
//   undo('undo' | 'redo') -> true when the stage took the key for a change of its own: asked when no stroke on the
//     model is the newest change to take back (Ctrl+Z), or none is there to bring back (Ctrl+Y).
//   changed(layer) -> a stroke, undo or redo changed 'base' (it is being saved) or 'overlay'; 'overlay' too when a
//     reset of the graphics card took the overlay.

import { ApiError, propApi, putBytes } from '../api.js';
import { replaceProp } from '../actions.js';
import { h, isTextField } from '../dom.js';
import { icon } from '../icons.js';
import { projectId } from '../store.js';
import { dialogOpen, toast } from '../ui.js';
import { isNewest, isUndoKey, undoList, undoStamp } from '../undo.js';
import { ringCursor } from './marks.js';
import { DEFAULT_SIZE, createPressureReader, dragSize, posToSize, sizeToPos } from './paintstroke.js';
import { undoKey } from './strokes.js';

const SAVE_DELAY = 2500;         // ms of quiet after a change before it is saved (a 4K save rewrites ~100 MB of files)
const MAX_CURSOR = 120;          // browsers draw a cursor up to 128 px; a bigger brush gets a drawn ring
const ONLY_COLOUR = 'The brushes paint the base colour only, so the view went to “All” to show it. Roughness and '
  + 'Metallic can’t be painted yet.';
const BASE = 'base';             // the layers of paint3d.js: the base colour, saved ...
const OVERLAY = 'overlay';       // ... and the stage's own (its tool has its name)
// How to use the viewport, in two lines to the left of the glass: to look around, or with a tool up.
const NAV = ['Orbit: middle or left drag · Pan: right drag', 'Zoom: wheel · F: frame'];
const KEYS = 'Shift + move sideways: brush size · Alt + drag or middle drag: orbit · Ctrl+Z: undo · Esc: put it down';
const STILL_KEYS = 'Esc: put it down to move the view again · Shift + move sideways: brush size · Ctrl+Z: undo';
const HINTS = {
  marker: ['Marker: drag on the model to paint · Alt + click: pick the colour under the pointer', KEYS],
  clonePick: ['Clone stamp: Ctrl+click the model where to copy from, then paint', KEYS],
  clone: ['Clone stamp: paint to copy from the + mark · Ctrl+click: copy from elsewhere', KEYS],
};

// The brush as the artist set it, kept from prop to prop and from stage to stage while the page is open.
const brush = { tool: null, size: DEFAULT_SIZE, colour: '#e5484d', bySize: true, byFade: false };
const pressure = createPressureReader();

// Saves under way, and bindings with changes still to save: closing the page then asks first.
const saving = new Set();
const waiting = new Set();
const savers = new WeakMap();    // a painted model's session -> its saver: one save at a time, in order
const painters = new Set();      // the painters of the models the bars are bound to now
globalThis.addEventListener?.('beforeunload', (e) => {
  if (!saving.size && !waiting.size) return;
  for (const flush of waiting) flush();
  e.preventDefault();
});

// What the artist is told when the graphics card is reset under the painting (paint3d.js never saves the emptied
// texture, and fills it with the painting saved last once the card is back).
const LOST = 'The graphics card was reset, so painting is paused until the page has it back. The painting saved last '
  + 'then shows on the model again; strokes from the last few seconds may be missing. If the model stays black or '
  + 'frozen, reload the page (F5).';
const RESTORED = 'The graphics card is back: the model shows the painting saved last, and you can paint again.';

/**
 * A stroke is under way, or a painting waits to be saved or is on its way into its files: no moment to load a whole
 * model beside it (stages/thumbs.js draws the props' snapshots once this is over).
 */
export const paintingNow = () => saving.size > 0 || waiting.size > 0 || [...painters].some((p) => p.painting());

/** Saves a painting into the bake whose GLB is `glb` (Projection's): the hub answers with the prop. */
export function saveToBake({ project, propId, glb }, image) {
  const query = `width=${image.width}&height=${image.height}&glb=${encodeURIComponent(glb)}`;
  return putBytes(`${propApi(project, propId)}/projection/paint?${query}`, image.data);
}

/**
 * Saves one model's painting: a moment after each change (schedule), or at once (flush), by `save` (saveToBake: into
 * the bake whose GLB is `glb`; once another bake replaced that one the painting is not saved, the hub says so).
 * The session's texture is held (paint3d.js hold) while something of it is still to save, so a model the viewer lets
 * go of meanwhile is saved whole, never read once freed. Exported for the tests.
 */
export function createSaver({ project, propId, glb, session, save = saveToBake }) {
  let timer = 0;
  let saves = Promise.resolve();   // the saves asked for, one after another ...
  let due = 0;                     // ... and how many of them are not done
  let gaveUp = false;
  let release = null;
  const dirty = () => !gaveUp && session.version !== session.saved;

  async function saveNow() {
    if (dirty()) {
      const running = (async () => {
        try {
          const image = await session.readAll();
          const prop = await save({ project, propId, glb }, image);
          session.savedAs(image);
          if (prop && projectId() === project) replaceProp(prop);
        } catch (e) {
          console.warn(e);
          // The texture went with the graphics card (paint3d.js): nothing of it is sent or tried again - the model
          // shows the painting saved last once the card is back, and the painter told the artist.
          if (e?.lost) return;
          if (e instanceof ApiError && (e.status === 404 || e.status === 409 || e.status === 422)) gaveUp = true;
          else waiting.add(flush);   // tried again with the next change, or when the page closes
          toast(e?.message || String(e), 'error');
        }
      })();
      saving.add(running);
      try {
        await running;
      } finally {
        saving.delete(running);
      }
    }
    // Saved, given up or lost, and no change since: nothing of the texture is needed any more.
    if (due === 1 && !waiting.has(flush)) {
      release?.();
      release = null;
    }
  }

  /** Saves what changed until now, after the save on its way if there is one -> once it is in the files (or failed). */
  function flush() {
    clearTimeout(timer);
    timer = 0;
    waiting.delete(flush);
    due++;
    saves = saves.then(saveNow).finally(() => { due--; });
    return saves;
  }

  return {
    schedule() {
      clearTimeout(timer);
      waiting.add(flush);
      release ||= session.hold();
      timer = setTimeout(flush, SAVE_DELAY);
    },
    flush,
    get busy() { return due > 0 || !!timer; },
  };
}

/**
 * The tools of one view that shows a model to paint (options: see the top of this file): {el (the one node for the
 * viewport), bind({viewer, project, propId, glb, select(view), begin?}) (resolves once the model's painter is there, or it
 * cannot be painted), unbind(), session(make), painting(), pick(tool), pending(), flush(), painted(), destroy()}.
 * select: picks a map view (the map bar's own).
 */
export function createPaintBar(options = {}) {
  const {
    save = saveToBake, patches = false, overlay = null, busy = () => false, undo = null, changed: onChange = null,
  } = options;
  let viewer = null;        // the viewer while a model is shown and can be painted
  let painter = null;       // paint3d.js's painter on it
  let binding = null;       // {project, propId, glb, select}
  let token = 0;
  let sizing = null;        // {x, start} while Shift is held over the model
  let pointer = null;       // the pointer over the model (client px), for the drawn ring and the clone's mark
  let cursorSize = 0;
  let overlayUp = false;    // the stage's own tool is up (as up() / down() told the stage)
  const due = new Set();    // the savers of the models this view painted with something on its way (flush waits)

  // ---- the glass: the brush size, P and F ----------------------------------------------------------------
  const size = h('input', {
    type: 'range', class: 'range pt-size', min: 0, max: 1000, value: sizeToPos(brush.size),
    title: 'Brush size (or hold Shift and move sideways over the model)', 'aria-label': 'Brush size',
  });
  const showSize = () => {
    size.value = sizeToPos(brush.size);
    size.style.setProperty('--fill', `${size.value / 10}%`);
    size.setAttribute('aria-valuetext', `${brush.size} pixels`);
    setCursor();
  };
  size.addEventListener('input', () => {
    brush.size = posToSize(+size.value);
    showSize();
  });
  size.addEventListener('pointerup', () => size.blur());   // Shift and Ctrl+Z belong to the viewer again

  const flag = (key, label, title) => {
    const b = h('button', {
      type: 'button', class: 'pt-flag', title, 'aria-label': title.split(':')[0], 'aria-pressed': String(brush[key]),
    }, label);
    b.addEventListener('click', () => {
      brush[key] = !brush[key];
      b.setAttribute('aria-pressed', String(brush[key]));
      b.blur();
    });
    return b;
  };
  const bySize = flag('bySize', 'P', 'Pressure: a pen’s pressure sizes the brush: a light touch paints thin. A mouse always '
    + 'paints at the full size.');
  const byFade = flag('byFade', 'F', 'Fade: a pen’s pressure sets how strongly the paint covers: a light touch paints faintly. '
    + 'With Pressure too, both follow the pen.');
  const status = h('span', { class: 'pt-status', hidden: true }, 'Saving…');
  const bar = h('div', { class: 'pt-bar' }, overlay?.slot, status, size, bySize, byFade);

  // ---- the tools' column, from the top down: Orbit, the clone stamp, the marker, the stage's own ---------------
  const orbit = h('button', {
    type: 'button', class: 'pt-tool', title: 'Orbit: put the brush down, so the left button turns the view again (Esc)',
    'aria-label': 'Orbit', onclick: () => pick(null),
  }, icon('orbit'));
  // The marker is the colour field itself: a click takes it up and opens the palette (the browser's own).
  const marker = h('input', {
    type: 'color', class: 'pt-tool pt-marker', value: brush.colour,
    title: 'Marker: paint on the model in any colour (a click opens the palette; Alt + click on the model picks its '
      + 'colour)', 'aria-label': 'Marker of any colour',
  });
  marker.addEventListener('click', () => pick('marker'));
  marker.addEventListener('input', () => {
    brush.colour = marker.value;
    pick('marker');
  });
  marker.addEventListener('change', () => marker.blur());
  const clone = h('button', {
    type: 'button', class: 'pt-tool', title: 'Clone stamp: Ctrl+click where to copy from, then paint the copy elsewhere',
    'aria-label': 'Clone stamp', onclick: () => pick(toolUp() === 'clone' ? null : 'clone'),
  }, icon('stamp'));
  overlay?.button.addEventListener('click', () => pick(toolUp() === OVERLAY ? null : OVERLAY));
  // Left of the stage's own tool while it is up: an x that wipes what the tool painted (Ctrl+Z brings it back).
  const wipe = overlay && h('button', {
    type: 'button', class: 'pt-wipe', hidden: true, title: overlay.wipe, 'aria-label': overlay.wipe,
    onclick: () => {
      if (!busy() && !painter?.painting() && session()?.wipe(OVERLAY)) changed(OVERLAY);
    },
  }, icon('x'));
  const tools = h('div', { class: 'pt-tools' }, orbit, clone, marker,
    overlay && h('span', { class: 'pt-own' }, wipe, overlay.button));

  const lines = [h('span'), h('span')];
  const hint = h('div', { class: 'pt-hint' }, ...lines);
  const ring = h('div', { class: 'pt-ring', hidden: true });
  const mark = h('div', { class: 'pt-source', hidden: true });
  const el = h('div', { class: 'pt-layer', hidden: true }, ring, mark,
    h('div', { class: 'pt-dock' }, tools, h('div', { class: 'pt-row' }, hint, bar)));

  /** The tool up here: the brush's, unless it is a stage's own this bar does not have, or the model is not begun. */
  const toolUp = () => (painter && !binding?.begin && (brush.tool !== OVERLAY || overlay) ? brush.tool : null);

  /** The stage makes the model bound paintable (its `begin`); it binds it anew then, and the tool asked for is up. */
  async function begin() {
    const mine = binding;
    if (mine.beginning) return;
    mine.beginning = true;
    try {
      await mine.begin();
    } catch (e) {
      console.warn(e);
      toast(e?.message || String(e), 'error');
      if (binding === mine) brush.tool = null;
    } finally {
      mine.beginning = false;
      if (binding === mine) sync();
    }
  }

  function pick(tool) {
    if (tool && busy()) return;
    brush.tool = tool;
    if (tool && painter && !painter.paintable()) brush.tool = null;
    if (brush.tool && binding?.begin) begin();
    // The brushes paint the base colour alone, which the lit view and the base colour's show: from another map's
    // view the brush goes to the lit one, and says why (the artist looking at Roughness meant to paint that).
    if (tool && viewer && !['all', 'basecolor'].includes(viewer.view)) {
      binding?.select?.('all');
      toast(ONLY_COLOUR);
    }
    sync();
  }

  function sync() {
    el.hidden = !painter;
    const tool = toolUp();
    orbit.classList.toggle('on', !tool);
    marker.classList.toggle('on', tool === 'marker');
    clone.classList.toggle('on', tool === 'clone');
    overlay?.button.classList.toggle('on', tool === OVERLAY);
    orbit.setAttribute('aria-pressed', String(!tool));
    clone.setAttribute('aria-pressed', String(tool === 'clone'));
    overlay?.button.setAttribute('aria-pressed', String(tool === OVERLAY));
    const words = !tool ? NAV : tool === OVERLAY ? [overlay.hint, overlay.still ? STILL_KEYS : KEYS]
      : tool === 'marker' ? HINTS.marker : painter.hasSource() ? HINTS.clone : HINTS.clonePick;
    lines.forEach((line, i) => { line.textContent = words[i] || ''; });
    hint.classList.toggle('active', !!tool);
    if (viewer) {
      viewer.holdLight = !!tool;   // a light circling the model while it is painted only distracts
      if (overlay?.still) viewer.holdView(tool === OVERLAY);
    }
    if (overlay && overlayUp !== (tool === OVERLAY)) {
      overlayUp = tool === OVERLAY;
      (overlayUp ? overlay.up : overlay.down)?.();
    }
    syncWipe();
    cursorSize = 0;
    setCursor();
    place();
  }

  /** The x shows while the stage's own tool is up, and can be pressed once that tool painted something. */
  function syncWipe() {
    if (!wipe) return;
    wipe.hidden = toolUp() !== OVERLAY;
    wipe.disabled = !session()?.painted(OVERLAY);
  }

  // ---- the cursor: a ring of the brush's size, the clone stamp's source mark --------------------------------
  function setCursor() {
    const canvas = viewer?.renderer?.domElement;
    if (!canvas) return;
    if (!toolUp()) {
      canvas.style.cursor = '';
      ring.hidden = true;
      return;
    }
    const d = Math.max(6, Math.round(brush.size));
    if (d === cursorSize) return;
    cursorSize = d;
    canvas.style.cursor = d <= MAX_CURSOR ? ringCursor(d) : 'crosshair';
    ring.style.setProperty('--d', `${d}px`);
    place();
  }

  /** The drawn ring (a brush too big for a cursor) and the clone's source mark follow the pointer. */
  function place() {
    const box = ring.parentElement?.getBoundingClientRect();
    const tool = toolUp();
    const show = !!(tool && pointer && box);
    ring.hidden = !show || cursorSize <= MAX_CURSOR;
    const source = show && tool === 'clone' ? painter.sourceAt(pointer.x, pointer.y) : null;
    mark.hidden = !source;
    if (!show) return;
    ring.style.transform = `translate(${pointer.x - box.left}px, ${pointer.y - box.top}px)`;
    if (source) mark.style.transform = `translate(${source.x - box.left}px, ${source.y - box.top}px)`;
  }

  // ---- the pointer over the model: sizing with Shift, the ring, the mark ------------------------------------
  function onMove(e) {
    if (!painter) return;
    pointer = { x: e.clientX, y: e.clientY };
    if (e.shiftKey && !e.buttons && toolUp() && !painter.painting()) {
      sizing ||= { x: e.clientX, start: brush.size };
      brush.size = dragSize(sizing.start, e.clientX - sizing.x);
      showSize();
    } else {
      sizing = null;
    }
    place();
  }
  const onLeave = () => {
    pointer = null;
    sizing = null;
    place();
  };

  // ---- keys: Esc puts the brush down, Ctrl+Z / Ctrl+Y take strokes back and bring them back ------------------
  const session = (make = false) => painter?.session(make) || null;
  const dropUndo = undoList(() => session()?.history.newest?.at);
  function onKey(e) {
    if (e.key === 'Shift') {
      if (e.type === 'keyup') sizing = null;
      return;
    }
    if (e.type !== 'keydown' || !painter || dialogOpen() || isTextField()) return;
    if (e.key === 'Escape' && brush.tool && binding?.begin) {     // asked for, not up yet (begin): not wanted any more
      brush.tool = null;
      e.preventDefault();
      return;
    }
    if (e.key === 'Escape' && toolUp() && !painter.painting()) {
      pick(null);
      e.preventDefault();
      return;
    }
    const which = isUndoKey(e) ? 'undo' : undoKey(e) === 'redo' ? 'redo' : null;
    if (!which || painter.painting() || busy()) return;
    const s = session();
    const step = which === 'undo' ? s?.history.newest : s?.history.nextRedo;
    const stroke = !!step && (which === 'redo' || isNewest(step.at));
    if (!stroke && !undo?.(which)) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    if (!stroke || !s[which]()) return;
    if (which === 'redo') step.at = undoStamp();       // made again: the newest change
    changed(step.layer === OVERLAY ? OVERLAY : BASE);
  }
  document.addEventListener('keydown', onKey, true);
  document.addEventListener('keyup', onKey, true);

  const saverOf = (s) => s && savers.get(s);
  function changed(layer) {
    const s = session();
    if (layer === BASE && s && binding) {
      if (!savers.has(s)) {
        savers.set(s, createSaver({ project: binding.project, propId: binding.propId, glb: binding.glb, session: s, save }));
      }
      savers.get(s).schedule();
      due.add(savers.get(s));
      status.hidden = false;
      watchSave();
    }
    syncWipe();
    onChange?.(layer);
  }

  const pending = () => [...due].some((saver) => saver.busy);
  /** Saves what is not saved yet of every model this view painted, now -> once it is in the files (or failed). */
  async function flush() {
    await Promise.all([...due].map((saver) => saver.flush()));
    for (const saver of due) if (!saver.busy) due.delete(saver);
  }

  // The "Saving…" note stays while a save is waiting or under way.
  let watching = 0;
  function watchSave() {
    clearInterval(watching);
    watching = setInterval(() => {
      if (pending()) return;
      status.hidden = true;
      clearInterval(watching);
    }, 250);
  }

  // ---- binding to a model -----------------------------------------------------------------------------------
  async function bind(next) {
    if (binding && next && binding.glb === next.glb && binding.propId === next.propId && viewer === next.viewer) return;
    unbind();
    if (!next?.viewer) return;
    const mine = token;          // a later bind() or unbind() moves it on
    const { createPainter } = await import('../paint3d.js');
    if (mine !== token) return;
    viewer = next.viewer;
    binding = { project: next.project, propId: next.propId, glb: next.glb, select: next.select, begin: next.begin };
    painter = createPainter(viewer, {
      patches,
      tool: () => (busy() ? null : toolUp()),
      size: () => brush.size,
      colour: () => (toolUp() === OVERLAY ? overlay.colour : brush.colour),
      bySize: () => brush.bySize,
      byFade: () => brush.byFade,
      pressure: (e) => pressure.read(e),
      strokeStart: () => pressure.start(),
      strokeEnd: () => pressure.end(),
      changed,
      source: () => sync(),
      picked: (hex) => {
        brush.colour = hex;
        marker.value = hex;
      },
      lost: () => toast(overlay?.lost ? `${LOST} ${overlay.lost}` : LOST, 'error'),
      restored: () => {
        toast(RESTORED, 'info');
        if (overlay) onChange?.(OVERLAY);
      },
    });
    if (!painter.paintable()) {
      const why = painter.why;
      unbind();
      if (why) console.info('Painting is off for this model:', why);
      return;
    }
    painters.add(painter);
    const canvas = viewer.renderer.domElement;
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerleave', onLeave);
    sync();
    const saver = saverOf(session());
    if (saver?.busy) {                   // back on a model whose painting is still on its way
      due.add(saver);
      status.hidden = false;
      watchSave();
    }
  }

  /** Lets go of the model: its painting is saved now (the save goes on after the view is gone). */
  function unbind() {
    token++;
    saverOf(session())?.flush();
    if (painter) {
      painters.delete(painter);
      painter.dropSource();
      painter.detach();
      const canvas = viewer.renderer.domElement;
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerleave', onLeave);
      canvas.style.cursor = '';
      viewer.holdLight = false;
      if (overlay?.still) viewer.holdView(false);
    }
    painter = null;
    viewer = null;
    binding = null;
    pointer = null;
    sync();
  }

  showSize();
  return {
    el,
    bind,
    unbind,
    /** The painting of the model on screen (paint3d.js), made now with `make`; null while there is none. */
    session,
    /** A stroke is under way. */
    painting: () => !!painter?.painting(),
    /** Takes up a tool ('marker', 'clone' or 'overlay': the stage's own) or puts the brush down (null). */
    pick,
    /** A stroke, undo or redo of this prop's models has not reached their files yet, or is on its way. */
    pending,
    /** Saves what is not saved yet, now -> once it is in the files (or failed: the toast says so). */
    flush,
    /** The model on screen has strokes on its base colour from this visit (its undo history). */
    painted: () => !!session()?.painted(BASE),
    destroy() {
      unbind();
      dropUndo();
      clearInterval(watching);
      document.removeEventListener('keydown', onKey, true);
      document.removeEventListener('keyup', onKey, true);
    },
  };
}
