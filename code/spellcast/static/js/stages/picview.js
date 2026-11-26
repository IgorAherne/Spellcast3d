// The Images workbench's picture looked at closely (images.js): the wheel zooms in about the pointer and back out to
// the whole picture, and the middle button drags it about. Zoomed all the way out, the wheel turned on outwards shows
// a tileable picture from afar, among copies of itself (Tileable); turned back from there, the picture shows alone
// again. The wheel pauses at the whole picture: a turn that came to it from either side goes no further that way for
// HOLD_MS, so one meant to stop there neither tiles the picture nor zooms into it as well.
// The view is three CSS properties of the viewport (--zoom, --pan-x, --pan-y) that the picture and every layer laid
// on it read (stages-images.css): they move as one, and what is drawn on top of them keeps its size on screen.

export const MAX_ZOOM = 16;         // times the whole picture
export const HOLD_MS = 400;
const ZOOM_PER_PX = 0.002;          // a wheel's notch (100 px) zooms by about a fifth
const LINE_PX = 33;                 // a wheel that counts in lines: about this many pixels each

const WHOLE = Object.freeze({ zoom: 1, x: 0, y: 0 });

/**
 * `view` ({zoom, x, y}: how many times the whole picture's size it shows, and where the picture's middle is from the
 * area's, in screen pixels) kept within reach: no smaller than the whole picture, and moved no further than to bring
 * the picture's edge to the area's. area, fitted: {w, h} of the area and of the whole picture fitted in it.
 */
export function held(view, area, fitted) {
  const zoom = Math.min(MAX_ZOOM, Math.max(1, view.zoom));
  const reach = (side, room) => Math.max(0, (side * zoom - room) / 2);
  const clamp = (v, r) => (r > 0 ? Math.min(r, Math.max(-r, v)) : 0);
  return { zoom, x: clamp(view.x, reach(fitted.w, area.w)), y: clamp(view.y, reach(fitted.h, area.h)) };
}

/** `view` zoomed by `factor` about the point `at` (from the area's middle, screen pixels), which stays where it is. */
export function zoomedAbout(view, factor, at, area, fitted) {
  const zoom = Math.min(MAX_ZOOM, Math.max(1, view.zoom * factor));
  const k = zoom / view.zoom;
  return held({ zoom, x: at.x - k * (at.x - view.x), y: at.y - k * (at.y - view.y) }, area, fitted);
}

/** The wheel's turn as the factor it zooms by (below 1: out), whatever unit the browser counts it in. */
export function wheelFactor(e, pagePx) {
  const px = e.deltaY * (e.deltaMode === 1 ? LINE_PX : e.deltaMode === 2 ? pagePx : 1);
  return Math.exp(-px * ZOOM_PER_PX);
}

/**
 * The view of the viewport `el` -> {afar (get / set), refit(), destroy()}. measure() -> {w, h, aspect} while a
 * picture shows: the layer it lies in as laid out now (the area at the view's zoom) and its width over its height;
 * null while none does (the view is then the whole of it). tiles() -> true while the picture may be seen from afar;
 * changed() after the wheel changed `afar`.
 */
export function createPictureView(el, { measure, tiles, changed }) {
  let view = WHOLE;
  let afar = false;
  let arrived = { at: -Infinity, out: false };   // when the wheel last came to the whole picture, turning which way
  let drag = null;           // {id, x, y}: the middle button pans (the pointer less the view's place)

  function show(next) {
    view = next;
    el.style.setProperty('--zoom', String(view.zoom));
    el.style.setProperty('--pan-x', `${view.x}px`);
    el.style.setProperty('--pan-y', `${view.y}px`);
  }

  /** The area with the whole picture in it ({area, fitted}), or null while no picture shows. */
  function sizes() {
    const now = measure();
    if (!now?.w || !now.h || !now.aspect) return null;
    const area = { w: now.w / view.zoom, h: now.h / view.zoom };
    const w = Math.min(area.w, area.h * now.aspect);
    return { area, fitted: { w, h: w / now.aspect } };
  }

  /** The view kept within reach of the picture shown now (its size, the viewport's): the whole of it with none. */
  function refit() {
    const now = sizes();
    show(now ? held(view, now.area, now.fitted) : WHOLE);
  }

  function onWheel(e) {
    const now = e.deltaY && !drag ? sizes() : null;
    if (!now) return;
    e.preventDefault();
    const out = e.deltaY > 0;
    if (afar) {
      if (out) return;
      afar = false;
      arrived = { at: performance.now(), out };
      changed();
      return;
    }
    if (arrived.out === out && performance.now() - arrived.at < HOLD_MS) return;
    if (out && view.zoom <= 1) {
      if (!tiles()) return;
      afar = true;
      changed();
      return;
    }
    const rect = el.getBoundingClientRect();
    const at = {
      x: e.clientX - (rect.left + el.clientLeft + el.clientWidth / 2),
      y: e.clientY - (rect.top + el.clientTop + el.clientHeight / 2),
    };
    const next = zoomedAbout(view, wheelFactor(e, el.clientHeight), at, now.area, now.fitted);
    if (out && next.zoom <= 1) arrived = { at: performance.now(), out };
    show(next);
  }

  // The middle button is the view's alone: the marks and the slices never see it, nor the browser's autoscroll.
  function onDown(e) {
    if (e.button !== 1 || e.pointerType !== 'mouse' || !sizes()) return;
    e.preventDefault();
    e.stopPropagation();
    if (afar || drag) return;
    drag = { id: e.pointerId, x: e.clientX - view.x, y: e.clientY - view.y };
    try { el.setPointerCapture(e.pointerId); } catch { /* the pointer is gone already */ }
    el.classList.add('panning');
  }
  function onMove(e) {
    if (drag?.id !== e.pointerId) return;
    const now = sizes();
    if (now) show(held({ zoom: view.zoom, x: e.clientX - drag.x, y: e.clientY - drag.y }, now.area, now.fitted));
  }
  function onUp(e) {
    if (drag?.id !== e.pointerId) return;
    drag = null;
    el.classList.remove('panning');
  }

  el.addEventListener('wheel', onWheel, { passive: false });
  el.addEventListener('pointerdown', onDown, true);
  el.addEventListener('pointermove', onMove);
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) el.addEventListener(type, onUp);
  const resized = globalThis.ResizeObserver ? new ResizeObserver(refit) : null;
  resized?.observe(el);

  return {
    get afar() { return afar; },
    /** Seen from afar or not, as the workbench sets it (Tileable off: never from afar). */
    set afar(on) { afar = !!on; },
    refit,
    /** (its listeners go with the viewport) */
    destroy() {
      resized?.disconnect();
    },
  };
}
