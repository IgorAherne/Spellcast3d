// Marks over the Images workbench's picture (Render Images): the artist paints on the picture in the viewport, or on
// the black canvas that stands in for one, with a white, a black or an any-colour marker, and "Re-Render Viewport"
// sends the picture with its marks (composite) to the image generator. The markers are circles in the viewport's
// bottom-right corner, the size slider on their left. Shift + moving sideways over the picture sizes the marker too.
// The marker's outline is the mouse cursor itself: drawn by the system, it never trails the pointer. Marks belong to
// the picture they were painted on for as long as the page is open. Ctrl+Z takes strokes back in the workbench's order
// of changes (images.js); Ctrl+Y brings one back. The strokes are strokes.js's.
// Two tools on a glass of their own, left of the markers (each where its callback is given); the one taken up stays
// lit until the artist puts it down. What they draw is a selection, never paint and no mark of the picture: a window
// onto moving light and dark stripes (standing still with Reduced animations) on a layer of its own.
// Select Object (the wand, onSelect): a stroke circles an object, and is closed with a straight line back to its start
// when it is let go; a small box beside it asks what it is, and onSelect(object, picture) gets the answer and the
// picture with that circle alone in magenta (picture() -> a PNG), as wide as the marker but never thinner than
// CIRCLE_MIN of the picture. The circle goes when the question is answered - the wand is put down with it - or
// dismissed: then the next stroke circles anew.
// Smart Mask (the brush, onMask): strokes cover a part of the picture, one after the other; after each, a larger box
// asks what should be there, and onMask(words, picture, mask) gets the words, the picture with the mask over it -
// flat magenta, nothing of that part shows - and the mask alone (both () -> a PNG). The words stay in the box from
// mask to mask and from picture to picture;
// rendered, they show selected, to type over. Ctrl+Z takes the mask's last stroke back (in its box: while the words
// are as the box opened with them).
// Where the change takes a part of the picture only (maskRequest says what is asked for: mask_window.js - the patch
// of a large picture's redraw, a tileable picture's window), a dashed frame shows that part round the mask and the
// stroke under way, and fades out when the stroke is let go; a mask too large for a window is not sent, and its box
// says why.

import { h, isTextField } from '../dom.js';
import { icon } from '../icons.js';
import { fullMotion } from '../shell/anim.js';
import { dialogOpen } from '../ui.js';
import { maskWindow, windowParts } from './mask_window.js';
import {
  addPoint, brushWidth, createHistory, paintPart, paintStroke, strokePart, strokeParts, strokeTail, undoKey,
} from './strokes.js';

export const MARKERS = [
  { value: 'white', colour: '#ffffff', title: 'White marker' },
  { value: 'black', colour: '#000000', title: 'Black marker' },
];
export const MIN_SIZE = 2;
export const MAX_SIZE = 64;                  // in the picture's own pixels
const SWEEP_PX = 250;                        // Shift + this far sideways goes through the whole size range
const TIP_MS = 2000;
const MAX_CURSOR = 120;                      // browsers draw a cursor up to 128 px
export const OBJECT_HINT = 'wooden crate';      // the Select Object box's grey example, until something is typed
export const MASK_HINT = 'a polished brass padlock with fine engravings';   // Smart Mask's
export const CIRCLE_COLOUR = '#ff00ff';      // the circle as the image generator gets it: "the magenta marker"
export const CIRCLE_MIN = 0.008;             // ... at least this much of the picture's longer side wide: it outlives
                                             // the image generator shrinking the picture
const MASK_WIDER = 2;                        // the mask's brush, in circle widths: it covers areas, not lines
const STRIPE_PX = 12;                        // the circle on the page: a light and a dark stripe in this many screen
const STRIPE_MS = 600;                       // pixels along a row, moving on by a pair in this time
const STRIPES = ['#ffffff', '#141414'];      // light, dark
export const MASK_TOO_LARGE = 'This mask is too large for a picture this size. '
  + 'Brush a smaller part, or clear the mask.';

/** A cursor that is a ring `d` screen pixels across (white on a dark edge, a dot in the middle), as CSS. */
export function ringCursor(d) {
  const side = Math.round(d) + 4;
  const c = Math.round(side / 2);
  const r = d / 2;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${side}" height="${side}">`
    + `<circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="#000" stroke-opacity=".55" stroke-width="3"/>`
    + `<circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="#fff" stroke-width="1.5"/>`
    + `<circle cx="${c}" cy="${c}" r="1" fill="#fff"/></svg>`;
  return `url("data:image/svg+xml,${encodeURIComponent(svg)}") ${c} ${c}, crosshair`;
}

// The marker as the artist set it, kept from prop to prop while the page is open: white until another is picked.
const pen = { marker: 'white', size: 12, colour: '#e5484d' };

/** The size Shift + `dx` screen pixels sideways give from `start`. */
export function dragSize(start, dx) {
  return Math.round(Math.min(MAX_SIZE, Math.max(MIN_SIZE, start + (dx * (MAX_SIZE - MIN_SIZE)) / SWEEP_PX)));
}

/** A light and a dark stripe at 45 degrees, `pair` pixels along a row and down a column: a tile that repeats seamlessly. */
function stripeTile(pair) {
  const tile = document.createElement('canvas');
  tile.width = tile.height = pair;
  const g = tile.getContext('2d');
  g.fillStyle = STRIPES[1];
  g.fillRect(0, 0, pair, pair);
  g.strokeStyle = STRIPES[0];
  g.lineWidth = pair / (2 * Math.SQRT2);           // half the pair, measured across the stripes
  g.beginPath();
  for (const mid of [pair / 4, (5 * pair) / 4]) {   // the light stripes' middles, where x + y = mid
    g.moveTo(mid + pair, -pair);
    g.lineTo(mid - 2 * pair, 2 * pair);
  }
  g.stroke();
  return tile;
}

/** The whole pixels a part of a stroke (strokes.js strokePart) can touch: {x0, y0, x1, y1}. */
function reach(part) {
  const points = part.dot ? [part.to] : [part.from, part.ctrl, part.to];
  const r = part.width / 2 + 1;                      // + 1: the smoothed edge
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  return {
    x0: Math.floor(Math.min(...xs) - r), y0: Math.floor(Math.min(...ys) - r),
    x1: Math.ceil(Math.max(...xs) + r), y1: Math.ceil(Math.max(...ys) + r),
  };
}

/**
 * A tool's selection on the page (Select Object's circle, Smart Mask's mask), a canvas of its own over the marks: its
 * strokes are painted once, part by part, and the stripes are filled through them ('source-atop' keeps their shape),
 * so a frame of the animation is one fill of the stripes over what they cover.
 * {el, start(size, scale, veil), paint(part), clear()}.
 */
function createCircleLayer() {
  const el = h('canvas', { class: 'mk-circle', width: 1, height: 1, hidden: true });
  const g = el.getContext?.('2d') || null;
  let stripes = null;          // the pattern: a stripe pair is `pair` picture pixels along a row
  let pair = 1;
  let shift = 0;               // how far the stripes have moved on
  let box = null;              // what the circle covers so far
  let frame = 0;

  function tint(r) {
    g.save();
    g.globalCompositeOperation = 'source-atop';
    g.translate(shift, 0);
    g.fillStyle = stripes;
    g.fillRect(r.x0 - shift, r.y0, r.x1 - r.x0, r.y1 - r.y0);
    g.restore();
  }
  const tick = (now) => {
    shift = ((now % STRIPE_MS) / STRIPE_MS) * pair;
    if (box) tint(box);
    frame = requestAnimationFrame(tick);
  };

  return {
    el,
    /**
     * A selection begins on a picture of `size` ({w, h}) shown at `scale` screen pixels a picture pixel; one that
     * shows goes on as it is (the mask's next stroke). `veil`: the picture shows through it (a mask covers an area).
     */
    start(size, scale, veil = false) {
      if (!g || !el.hidden) return;
      el.classList.toggle('veil', veil);
      el.width = size.w;
      el.height = size.h;
      pair = Math.max(2, Math.round(STRIPE_PX / (scale || 1)));
      stripes = g.createPattern(stripeTile(pair), 'repeat');
      shift = 0;
      el.hidden = false;
      if (fullMotion() && globalThis.requestAnimationFrame) frame = requestAnimationFrame(tick);
    },
    paint(part) {
      if (el.hidden) return;
      paintPart(g, part, STRIPES[0]);
      const r = reach(part);
      box = box ? {
        x0: Math.min(box.x0, r.x0), y0: Math.min(box.y0, r.y0), x1: Math.max(box.x1, r.x1), y1: Math.max(box.y1, r.y1),
      } : r;
      tint(r);
    },
    /** The selection goes: nothing shows, nothing runs. */
    clear() {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
      box = stripes = null;
      el.width = el.height = 1;                      // empty, its memory let go
      el.hidden = true;
    },
  };
}

/**
 * The marks layer: {el (over the picture), bar (the markers, and the tools, in the viewport's corner), hint,
 * picture(img, path), show(on), selectable(on), refit(), path, undo(), composite(), destroy()}. onStroke(path) after
 * each change of a picture's marks (a stroke, a wipe, a redo); onSelect(object, picture) after Select Object and
 * onMask(words, picture, mask) after Smart Mask (null: without that tool). maskRequest(): the picture Smart Mask's
 * change of the shown one is asked for as, {width, height, tileable}, or null where all of it is drawn anew.
 */
export function createMarks({ onStroke = null, onSelect = null, onMask = null, maskRequest = null } = {}) {
  const canvas = h('canvas', { class: 'mk-canvas', width: 1, height: 1 });
  canvas.setAttribute('aria-label', 'The picture: press and drag to mark it');
  const ctx = canvas.getContext?.('2d') || null;
  const circle = createCircleLayer();
  // Smart Mask's window, over its mask: a page element, never in the pictures sent.
  const windowEl = h('div', { class: 'mk-window', hidden: true, 'aria-hidden': 'true' });
  const fit = h('div', { class: 'mk-fit' }, canvas, circle.el, windowEl);   // the picture's own fitted box
  const el = h('div', { class: 'mk', hidden: true }, h('div', { class: 'mk-area' }, fit));
  const books = new Map();      // a picture's path -> its strokes (strokes.js createHistory)
  let img = null;               // the <img> the picture shows in (or the black <canvas>)
  let path = null;              // its path in the prop (the black canvas: its own key)
  let always = false;           // a picture the page drew itself (the black canvas, one shifted): sent even without marks
  let natural = null;           // {w, h}: its size, once loaded
  let shown = false;            // the workbench shows the markers (show)
  let drawing = null;           // {id, rect, step, painted, tool, paint(part)} while a stroke is drawn (tool: the one
                                // taken up as it began)
  let sizing = null;            // {x, start} while Shift is held over the picture
  let tipTimer = 0;
  let cursorSize = 0;           // the cursor's ring, in screen pixels
  let tool = null;              // the tool taken up: 'select' (the wand: a stroke circles an object), 'mask' (Smart
                                // Mask: strokes cover what should change); null: the markers paint
  let asking = null;            // {box, close()} while a tool's box asks about what it drew
  let canSelect = true;         // the image generator takes "I circled ..." (selectable)
  let mask = [];                // Smart Mask's strokes on the shown picture
  let maskWords = '';           // what its box holds
  let maskWordsSent = false;    // ... rendered as they stand: the box shows them selected
  let tooLarge = false;         // the mask does not fit in the window its change would be drawn in: it is not sent

  const book = (p = path) => {
    if (!books.has(p)) books.set(p, createHistory());
    return books.get(p);
  };
  const hasMarks = (p = path) => !!p && books.has(p) && books.get(p).shown().strokes.length > 0;
  const colourOf = (marker) => MARKERS.find((m) => m.value === marker)?.colour || pen.colour;
  const circleWidth = () => Math.max(pen.size, Math.round(Math.max(natural.w, natural.h) * CIRCLE_MIN));
  /** How wide a stroke goes with `kind` taken up, in the picture's pixels (the marker's follows a pen's pressure). */
  const strokeWidth = (kind, e = {}) => (kind === 'select' ? circleWidth() : kind === 'mask' ? MASK_WIDER * circleWidth()
    : brushWidth(pen.size, e.pointerType, e.pressure));

  function redraw() {
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (path && books.has(path)) for (const stroke of books.get(path).shown().strokes) paintStroke(ctx, stroke);
  }

  // ---- the markers, the size, the wipe ----------------------------------------------------------------------
  const wipe = h('button', {
    type: 'button', class: 'mk-wipe', title: 'Wipe the marks off this picture (Ctrl+Z brings them back)',
    'aria-label': 'Wipe the marks off this picture', onclick: () => clear(),
  }, icon('trash'));
  const size = h('input', {
    type: 'range', class: 'range mk-size', min: MIN_SIZE, max: MAX_SIZE, value: pen.size,
    title: 'Marker size (or hold Shift and move sideways over the picture)', 'aria-label': 'Marker size',
  });
  const showSize = () => {
    size.value = pen.size;
    size.style.setProperty('--fill', `${((pen.size - MIN_SIZE) / (MAX_SIZE - MIN_SIZE)) * 100}%`);
    setCursor();
  };
  const hint = h('div', { class: 'mk-hint', hidden: true }, 'Tip: hold Shift and move sideways over the picture to size the marker');
  size.addEventListener('input', () => {
    pen.size = +size.value;
    showSize();
    hint.hidden = false;           // the quicker way, said for a moment
    clearTimeout(tipTimer);
    tipTimer = setTimeout(() => { hint.hidden = true; }, TIP_MS);
  });
  // Let go after a drag: a focused slider would take Shift (its outline) and keep Ctrl+Z from the workbench.
  size.addEventListener('pointerup', () => size.blur());
  showSize();
  const pens = MARKERS.map((m) => h('button', {
    type: 'button', class: `mk-pen mk-${m.value}`, title: `${m.title}: paint on the picture, then Re-Render Viewport`,
    'aria-label': m.title, 'aria-pressed': 'false', onclick: () => mark(m.value),
  }));
  // The rainbow marker is the colour field itself: a click opens the palette (the browser's own) and picks it.
  const rainbow = h('input', {
    type: 'color', class: 'mk-pen mk-rainbow', value: pen.colour,
    title: 'A marker of any colour: pick it from the palette', 'aria-label': 'Marker of any colour',
  });
  rainbow.addEventListener('click', () => mark('colour'));
  rainbow.addEventListener('input', () => {
    pen.colour = rainbow.value;
    mark('colour');
  });
  rainbow.addEventListener('change', () => rainbow.blur());   // the palette closed
  const toolButton = (kind, cls, label, title, glyph = null) => h('button', {
    type: 'button', class: `mk-tool ${cls}`, 'aria-label': label, 'aria-pressed': 'false', title,
    onclick: () => take(tool === kind ? null : kind),
  }, glyph);
  const wand = onSelect && toolButton('select', 'mk-select', 'Select object', 'Select object: circle it and say what it '
    + 'is. It comes out alone and large, in its style and facing the same way, on a gray background');
  const brush = onMask && toolButton('mask', 'mk-smart', 'Smart mask', 'Smart mask: brush over a part of the picture, '
    + 'then say what should be there. The picture is drawn again with it', icon('maskbrush'));
  const markers = h('div', { class: 'mk-bar', hidden: true }, wipe, size, pens, rainbow);
  // The tools on a glass of their own, left of the markers': tools of their own, not settings of the marker.
  const wandGlass = (wand || brush) && h('div', { class: 'mk-wand', hidden: true }, wand, brush);
  const bar = h('div', { class: 'mk-tools' }, wandGlass, markers);

  function pick(marker) {
    pen.marker = marker;
    pens.forEach((b, i) => {
      b.classList.toggle('on', MARKERS[i].value === marker);
      b.setAttribute('aria-pressed', String(MARKERS[i].value === marker));
    });
    rainbow.classList.toggle('on', marker === 'colour');
  }
  pick(pen.marker);

  /** The artist picks a marker: it paints from now on, so a tool that was taken up is put down. */
  function mark(marker) {
    take(null);
    pick(marker);
  }

  /** A tool taken up ('select', 'mask': its button lit) or put down (null: the markers paint), with what it drew. */
  function take(kind) {
    const next = canSelect && ((kind === 'select' && wand) || (kind === 'mask' && brush)) ? kind : null;
    if (next === tool) return;
    dropSelection();
    tool = next;
    for (const [button, its] of [[wand, 'select'], [brush, 'mask']]) {
      button?.classList.toggle('on', tool === its);
      button?.setAttribute('aria-pressed', String(tool === its));
    }
    markers.classList.toggle('idle', !!tool);    // one thing lit: the marker's ring waits while a tool is up
    setCursor();
  }

  function sync() {
    el.hidden = !shown || !natural;
    markers.hidden = !shown || !natural;
    if (wandGlass) wandGlass.hidden = markers.hidden || !canSelect;
    wipe.hidden = !hasMarks();
    if (markers.hidden) hint.hidden = true;
  }

  // ---- painting, and sizing with Shift ----------------------------------------------------------------------
  const scale = () => (natural ? canvas.getBoundingClientRect().width / natural.w : 1);

  /** The cursor shows on screen how wide the next stroke goes (made again only when that changes). */
  function setCursor() {
    if (!natural) return;
    const d = Math.min(MAX_CURSOR, Math.max(6, Math.round(strokeWidth(tool) * scale())));
    if (d === cursorSize) return;
    cursorSize = d;
    canvas.style.cursor = ringCursor(d);
  }

  function extend(e) {
    const { step, rect } = drawing;
    const added = addPoint(step.points, {
      x: Math.round(((e.clientX - rect.left) / (rect.width || 1)) * natural.w * 10) / 10,
      y: Math.round(((e.clientY - rect.top) / (rect.height || 1)) * natural.h * 10) / 10,
      w: strokeWidth(drawing.tool, e),
    });
    if (added) paintOn();
  }

  /** The parts of the stroke under way that its points make and that are not painted yet. */
  function paintOn() {
    const { step, paint } = drawing;
    for (; drawing.painted <= step.points.length - 2; drawing.painted++) paint(strokePart(step.points, drawing.painted));
    if (step.points.length === 1) paint(strokePart(step.points, 0));   // the dot shows at once
  }

  canvas.addEventListener('pointerdown', (e) => {
    if (!shown || !natural || !ctx || drawing || (e.pointerType === 'mouse' && e.button !== 0)) return;
    e.preventDefault();
    if (e.shiftKey) return;                              // Shift sizes the marker, it does not paint
    if (isTextField()) document.activeElement.blur();   // Ctrl+Z is the workbench's now, not the words field's
    try { canvas.setPointerCapture(e.pointerId); } catch { /* the pointer is gone already */ }
    // A tool's stroke goes on the selection's layer, never on the marks: a circle in place of the one asked about,
    // the mask one stroke larger (its box is out of the way meanwhile).
    if (tool === 'select') dropSelection();
    else asking?.close();
    const step = { kind: 'stroke', colour: tool ? CIRCLE_COLOUR : colourOf(pen.marker), points: [] };
    if (tool) circle.start(natural, scale(), tool === 'mask');
    const paint = tool ? circle.paint : (part) => paintPart(ctx, part, step.colour);
    drawing = { id: e.pointerId, rect: canvas.getBoundingClientRect(), step, painted: 1, tool, paint, measured: 0 };
    extend(e);
    if (tool === 'mask') windowEl.classList.remove('fade');   // back at full strength
    measureDrawing();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (e.shiftKey && !drawing) {                        // Shift + sideways: the size, from where Shift was first seen
      sizing ||= { x: e.clientX, start: pen.size };
      pen.size = dragSize(sizing.start, e.clientX - sizing.x);
      showSize();
      return;
    }
    sizing = null;
    setCursor();                                         // the picture may show at another size since
    if (drawing?.id !== e.pointerId) return;
    const events = e.getCoalescedEvents?.() || [];     // every place the pointer was since the last frame
    for (const one of events.length ? events : [e]) extend(one);
    measureDrawing();
  });
  const end = (e) => {
    if (drawing?.id !== e.pointerId) return;
    const { step, paint } = drawing;
    // The circle is closed: a straight line from where it is let go back to where it began.
    if (drawing.tool === 'select' && step.points.length > 2 && addPoint(step.points, { ...step.points[0] })) paintOn();
    const made = drawing.tool;
    drawing = null;
    if (!step.points.length) return;
    const tail = strokeTail(step.points);
    if (tail) paint(tail);
    if (made === 'select') {
      ask(step);
    } else if (made === 'mask') {
      mask.push(step);
      measure(mask);
      fadeWindow();
      askChange();
    } else {
      book().add(step);
      changed();
    }
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('pointerleave', () => { sizing = null; });

  function changed() {
    sync();
    onStroke?.(path);
  }

  // ---- Smart Mask's window: where the image server draws the change, when it draws a window only ------------
  /**
   * The window round the mask `strokes` shows (in parts round a tileable picture's edge) - round the least it would
   * take when the mask is too large for one (tooLarge) -; none where all of the picture is drawn anew.
   */
  function measure(strokes) {
    const got = strokes.length > 0 && maskRequest && natural
      && maskWindow(strokes, [natural.w, natural.h], maskRequest());
    tooLarge = !!got && !got.fits;
    windowEl.hidden = !got;
    if (!got) {
      windowEl.replaceChildren();
      return;
    }
    const [w, hh] = got.size;
    const pct = (v, of) => `${(v / of) * 100}%`;
    windowEl.replaceChildren(...windowParts(got.box, got.size).map(({ box: [x0, y0, x1, y1], open }) => h('i', {
      class: ['mk-window-part', ...open.map((side) => `open-${side}`)].join(' '),
      style: `left: ${pct(x0, w)}; top: ${pct(y0, hh)}; width: ${pct(x1 - x0, w)}; height: ${pct(y1 - y0, hh)}`,
    })));
  }

  /** The window for the mask and the stroke under way, once that has more points. */
  function measureDrawing() {
    if (drawing.tool !== 'mask' || drawing.measured === drawing.step.points.length) return;
    drawing.measured = drawing.step.points.length;
    measure([...mask, drawing.step]);
  }

  /** The window fades out (stages-images.css .mk-window.fade), from as it shows now. */
  function fadeWindow() {
    if (windowEl.hidden) return;
    void windowEl.offsetWidth;                 // its style now, for the fade to start from
    windowEl.classList.add('fade');
  }

  // ---- the tools' boxes: what was drawn, asked beside it ----------------------------------------------------
  /**
   * `box` is put on the picture where `strokes` leave room - under them, over them near the picture's bottom, along
   * that bottom when they reach both - centred on them and inside the picture, and is the one that asks from now on
   * (its field takes the keys: Enter, Escape and Ctrl+Z are its own). `gone()`: what else ends with it.
   */
  function putBox(box, field, strokes, gone = () => {}) {
    const points = strokes.flatMap((stroke) => stroke.points);
    const [x0, x1] = [Math.min(...points.map((p) => p.x)), Math.max(...points.map((p) => p.x))];
    const [y0, y1] = [Math.min(...points.map((p) => p.y)), Math.max(...points.map((p) => p.y))];
    const below = y1 / natural.h < 0.75;
    box.style.top = `${((below ? y1 : y0 / natural.h > 0.3 ? y0 : natural.h) / natural.h) * 100}%`;
    box.classList.toggle('above', !below);
    fit.append(box);
    // (a share of the picture's width: the box stays at the strokes as the picture is zoomed)
    const room = fit.clientWidth || 1;
    const wide = box.offsetWidth || 0;
    const left = Math.max(0, Math.min(room - wide, ((x0 + x1) / 2 / natural.w) * room - wide / 2));
    box.style.left = `${(left / room) * 100}%`;
    asking = {
      box,
      close() {
        if (asking?.box !== box) return;
        asking = null;
        box.remove();
        gone();
      },
    };
    field.focus({ preventScroll: true });
  }

  /** The box at the circle `lasso`: the object's name, Enter renders it. A press elsewhere, or Escape, lets it go. */
  function ask(lasso) {
    const input = h('input', {
      class: 'input mk-ask-input', placeholder: OBJECT_HINT, spellcheck: 'true', 'aria-label': 'What you circled',
    });
    const go = h('button', { type: 'button', class: 'btn primary small', disabled: true, onclick: () => done(input.value) },
      'Render');
    input.addEventListener('input', () => { go.disabled = !input.value.trim(); });
    const box = h('div', { class: 'mk-ask' }, h('span', { class: 'mk-ask-label' }, 'What did you circle?'),
      h('div', { class: 'mk-ask-row' }, input, go));
    input.addEventListener('keydown', (e) => {
      e.stopPropagation();
      if (e.key === 'Enter' && !e.isComposing) {
        e.preventDefault();
        if (input.value.trim()) done(input.value);   // nothing typed yet: the question stays
      } else if (e.key === 'Escape') {
        done(null);
      }
    });
    // A press elsewhere lets the circle go - on the picture it begins the next one, the wand still taken up.
    const away = (e) => { if (!box.contains(e.target)) done(null); };
    putBox(box, input, [lasso], () => document.removeEventListener('pointerdown', away, true));
    setTimeout(() => {                             // after the press that ended the circle
      if (asking?.box === box) document.addEventListener('pointerdown', away, true);
    });

    function done(answer) {
      if (asking?.box !== box) return;
      const object = answer?.trim();
      if (object) take(null);                      // rendered: the wand is put down
      else dropSelection();
      if (object) onSelect(object, () => pictureWith((g) => paintStroke(g, lasso)));
    }
  }

  /**
   * The box at the mask, with the words it held before (selected once they were rendered): Enter, or Render, sends
   * them with the mask - not while it is too large (the box says so) -; Escape, or "Clear mask", takes the mask off.
   * Shift+Enter is a new line.
   */
  function askChange() {
    const input = h('textarea', {
      class: 'input mk-ask-input', rows: 3, placeholder: MASK_HINT, spellcheck: 'true', 'aria-label': 'What should be under the mask',
    });
    input.value = maskWords;
    const opened = input.value;
    const go = h('button', { type: 'button', class: 'btn primary small', onclick: () => render() }, 'Render');
    const clear = h('button', { type: 'button', class: 'btn ghost small', title: 'Take the mask off (Esc)', onclick: () => dropSelection() },
      'Clear mask');
    const why = h('span', { class: 'mk-ask-why' }, MASK_TOO_LARGE);
    /** Render goes once there are words, and while the mask fits its window. */
    const allow = () => {
      go.disabled = !maskWords.trim() || tooLarge;
      why.hidden = !tooLarge;
    };
    allow();
    input.addEventListener('input', () => {
      maskWords = input.value;
      maskWordsSent = false;
      allow();
    });
    input.addEventListener('keydown', (e) => {
      e.stopPropagation();
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        render();
      } else if (e.key === 'Escape') {
        dropSelection();
      } else if (undoKey(e) === 'undo' && input.value === opened) {   // nothing of the field's own to take back
        e.preventDefault();
        unbrush();
      }
    });
    putBox(h('div', { class: 'mk-ask mk-ask-mask' }, h('span', { class: 'mk-ask-label' }, 'What should be here?'), input,
      why, h('div', { class: 'mk-ask-row' }, clear, go)), input, mask);
    asking.allow = allow;
    if (maskWordsSent) input.select();

    function render() {
      const words = maskWords.trim();
      if (!words || tooLarge) return;              // nothing typed yet, or too large: the question stays
      const strokes = mask;
      dropSelection();
      maskWordsSent = true;
      onMask(words, () => pictureWith((g) => paintMask(g, strokes)), () => png((g) => paintMaskAlone(g, strokes)));
    }
  }

  /**
   * The mask as the image generator gets it: its strokes in flat magenta over the picture. Nothing shows through:
   * what lies under it is not the model's to copy.
   */
  function paintMask(g, strokes) {
    for (const stroke of strokes) paintStroke(g, stroke);
  }

  /** Where the mask lies, alone: white on black. */
  function paintMaskAlone(g, strokes) {
    g.fillStyle = '#000000';
    g.fillRect(0, 0, natural.w, natural.h);
    for (const stroke of strokes) paintStroke(g, { ...stroke, colour: '#ffffff' });
  }

  /** What the tool taken up drew goes - a circle, a mask, still drawn or asked about - and its box with it. */
  function dropSelection() {
    if (drawing?.tool) drawing = null;
    asking?.close();
    mask = [];
    circle.clear();
    measure(mask);
  }

  /** Smart Mask's last stroke taken back: the rest is painted again, its window shows a moment, it is asked about. */
  function unbrush() {
    mask.pop();
    if (!mask.length) {
      dropSelection();
      return;
    }
    asking?.close();
    circle.clear();
    circle.start(natural, scale(), true);
    for (const stroke of mask) for (const part of strokeParts(stroke.points)) circle.paint(part);
    windowEl.classList.remove('fade');
    measure(mask);
    fadeWindow();
    askChange();
  }

  function clear() {
    if (drawing || !hasMarks()) return;
    book().add({ kind: 'clear' });
    redraw();
    changed();
  }

  /** Takes the shown picture's last mark back (true), or there is none (false). */
  function undo() {
    if (drawing || !path || !books.get(path)?.undo()) return false;
    redraw();
    sync();
    return true;
  }

  function redo() {
    if (drawing || !path || !books.get(path)?.redo()) return false;
    redraw();
    changed();
    return true;
  }

  // ---- keys: Ctrl+Y brings a stroke back (window, capturing); letting go of Shift ends the sizing -------------
  function onKey(e) {
    if (e.key === 'Shift') {
      if (e.type === 'keyup') sizing = null;
      return;
    }
    // Ctrl+Z takes the mask's last stroke back before any mark of the picture's (images.js listens after this).
    if (e.type === 'keydown' && mask.length && !drawing && !dialogOpen() && !isTextField() && undoKey(e) === 'undo') {
      e.preventDefault();
      e.stopImmediatePropagation();
      unbrush();
      return;
    }
    // Escape puts the tool down, and is spent on that (the page would let go of the prop next); with a box asking,
    // the box takes it first, and lets go of what was drawn.
    if (e.type === 'keydown' && e.key === 'Escape' && tool && !drawing && !asking) {
      take(null);
      e.preventDefault();
      return;
    }
    if (e.type !== 'keydown' || el.hidden || dialogOpen() || isTextField() || undoKey(e) !== 'redo') return;
    if (!path || !books.get(path)?.canRedo) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    redo();
  }
  globalThis.addEventListener?.('keydown', onKey, true);
  globalThis.addEventListener?.('keyup', onKey, true);

  /** What `draw` paints on a canvas of the shown picture's size (its 2D context), as a PNG Blob. */
  function png(draw) {
    const out = document.createElement('canvas');
    out.width = natural.w;
    out.height = natural.h;
    draw(out.getContext('2d'));
    return new Promise((resolve, reject) => {
      out.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('Could not read the marked picture.'))), 'image/png');
    });
  }

  /** The shown picture with what `over` paints on it, as a PNG Blob. */
  const pictureWith = (over) => png((g) => {
    g.drawImage(img, 0, 0, natural.w, natural.h);
    over(g);
  });

  function loaded(which, p) {
    const [w, hh] = [which.naturalWidth || which.width, which.naturalHeight || which.height];
    if (which !== img || p !== path || !w) return;
    natural = { w, h: hh };
    canvas.width = natural.w;
    canvas.height = natural.h;
    el.style.setProperty('--ar', String(natural.w / natural.h));
    redraw();
    sync();
    cursorSize = 0;
    setCursor();
  }

  return {
    el, bar, hint,

    /**
     * The workbench shows the <img> `picture` (null: none) of the prop's picture `p`, or the black <canvas> that
     * stands in for one, or a <canvas> the page drew it onto (`always`: `p` is its own key, and composite() always
     * gives it).
     */
    picture(picture, p, { always: drawn = false } = {}) {
      if (picture === img && p === path) return;
      drawing = sizing = null;
      dropSelection();
      img = picture;
      path = picture ? p : null;
      always = !!picture && drawn;
      natural = null;
      redraw();
      sync();
      if (!picture) return;
      if (picture.tagName === 'CANVAS' || (picture.complete && picture.naturalWidth)) loaded(picture, p);
      else picture.addEventListener('load', () => loaded(picture, p), { once: true });
    },

    /**
     * The markers show and paint (Render Images, a picture and no job). Hidden, the wand is put down - its picture is
     * being rendered, or the workbench shows something else; Smart Mask stays taken up for the next picture.
     */
    show(on) {
      shown = !!on;
      if (!shown) {
        drawing = sizing = null;
        if (tool === 'select') take(null);
        dropSelection();
      }
      sync();
    },

    /** The tools can be taken up (the image generator understands "I circled ..."): their glass shows. */
    selectable(on) {
      canSelect = !!on;
      if (!canSelect) take(null);
      sync();
    },

    /**
     * What maskRequest() gives may have changed (the Shape, the Resolution, Tileable): the mask's box says again
     * whether it fits. Its window stays out of sight until the next stroke.
     */
    refit() {
      if (!mask.length || drawing) return;
      measure(mask);
      windowEl.classList.add('fade');
      asking?.allow?.();
    },

    get path() { return path; },
    undo,

    /** The shown picture with its marks, as a PNG (a Blob), or null when it has none (a canvas of the page's: always). */
    composite() {
      if (!img || !natural || (!always && !hasMarks())) return Promise.resolve(null);
      return pictureWith((g) => g.drawImage(canvas, 0, 0));
    },

    destroy() {
      clearTimeout(tipTimer);
      dropSelection();
      globalThis.removeEventListener?.('keydown', onKey, true);
      globalThis.removeEventListener?.('keyup', onKey, true);
    },
  };
}
