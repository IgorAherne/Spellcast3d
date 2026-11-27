// The shell's motion: the whole page swiping from one stage to the next, a new prop sliding into the list,
// the pop of a check a prop has just earned, the refusal shake and the call to action's attention pulse.
//
// Motion comes in two levels. 'full' is all of it. 'gentle' keeps what tells where things go, in short,
// plain moves: the stages cross-fade instead of swiping, and nothing pops, bounces or pulses. Each viewer
// chooses in Settings > Advanced > Animations: "Full" (the default) or "Reduced" (gentle). The system's
// wish for less motion is not followed: Windows asks for it whenever its "Animation effects" are off, a
// common setting for speed, and the swipe is what shows where the work went. The choice is kept in this
// browser, and set on <html data-motion> for the style sheets.

const PREF_KEY = 'spellcast3d.motion';
const MOTION_PREFS = ['full', 'reduced'];

let pref = null;

// Only "Reduced" is stored. Earlier versions stored 'always' (full motion; the default now) and kept
// nothing for 'auto': both read as full.
function readPref() {
  try {
    return localStorage.getItem(PREF_KEY) === 'reduced' ? 'reduced' : 'full';
  } catch {
    return 'full';
  }
}

/** The viewer's choice: 'full' or 'reduced'. */
export const motionPref = () => (pref ??= readPref());

/** 'full' or 'gentle'. */
export const motionLevel = () => (motionPref() === 'reduced' ? 'gentle' : 'full');
export const fullMotion = () => motionLevel() === 'full';

const applyMotion = () => { document.documentElement.dataset.motion = motionLevel(); };

/** Keeps the viewer's choice (for this browser; for this page only when the browser keeps nothing). */
export function setMotionPref(value) {
  pref = MOTION_PREFS.includes(value) ? value : 'full';
  try {
    if (pref === 'full') localStorage.removeItem(PREF_KEY);
    else localStorage.setItem(PREF_KEY, pref);
  } catch { /* this page keeps it */ }
  applyMotion();
}

/** Sets <html data-motion> for the style sheets. Call once, at start. */
export function initMotion() {
  applyMotion();
}

/** How the page swipes to another stage (full motion), or cross-fades (gentle). */
export const SWIPE = { duration: 600, easing: 'cubic-bezier(.55,0,.15,1)' };
export const FADE = { duration: 150, easing: 'ease-out' };
const SWIPE_GAP = '16px';   // between the page leaving and the page arriving, like two sections of a page

/**
 * Resolves once `animations` have finished, or `ms` after the start at the latest, when any still running
 * are made to finish: a page that paints no frames (a hidden tab) must never be left half-moved.
 */
export async function settle(animations, ms) {
  let timer;
  await Promise.race([
    Promise.allSettled(animations.map((a) => a.finished)),
    new Promise((resolve) => { timer = setTimeout(resolve, ms); }),
  ]);
  clearTimeout(timer);
  for (const a of animations) {
    if (a.playState === 'running' || a.playState === 'paused') {
      try { a.finish(); } catch { a.cancel(); }
    }
  }
}

export function fadeIn(el, ms = 180) {
  const a = el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: fullMotion() ? ms : 80, easing: 'ease-out' });
  return settle([a], ms + 400);
}

/** A new prop in the list: it fades in and slides up into its place (gentle motion: it fades in). */
export function slideIn(el) {
  const a = el.animate(fullMotion()
    ? [{ opacity: 0, transform: 'translateY(10px)' }, { opacity: 1, transform: 'none' }]
    : [{ opacity: 0 }, { opacity: 1 }], { duration: 200, easing: 'ease-out' });
  return settle([a], 600);
}

/**
 * The page goes to another stage: `from` (a still copy of the page being left, laid over it) and `to`
 * (the page, already drawn for the new stage) swipe up together going down (`down`), like scrolling to
 * the next section, or down together going up. Gentle motion: a short cross-fade. Resolves when it is over.
 */
export function swipe(from, to, down) {
  if (!fullMotion()) {
    return settle([
      ...(from ? [from.animate([{ opacity: 1 }, { opacity: 0 }], { ...FADE, fill: 'forwards' })] : []),
      to.animate([{ opacity: 0 }, { opacity: 1 }], FADE),
    ], FADE.duration + 500);
  }
  const out = down ? `calc(-100% - ${SWIPE_GAP})` : `calc(100% + ${SWIPE_GAP})`;
  const into = down ? `calc(100% + ${SWIPE_GAP})` : `calc(-100% - ${SWIPE_GAP})`;
  return settle([
    ...(from ? [from.animate([{ transform: 'none' }, { transform: `translateY(${out})` }], { ...SWIPE, fill: 'forwards' })] : []),
    to.animate([{ transform: `translateY(${into})` }, { transform: 'none' }], SWIPE),
  ], SWIPE.duration + 600);
}

/** "Look here": a ring swells out of `el` and fades, `times` over. No movement, so gentle motion keeps it. */
export function ring(el, times = 1) {
  el?.animate?.([
    { boxShadow: '0 0 0 0 rgba(139, 125, 255, 0.8)' },
    { boxShadow: '0 0 0 10px rgba(139, 125, 255, 0)' },
  ], { duration: 700, easing: 'ease-out', iterations: times });
}

/**
 * "Look here", for what is far from where the artist looks: a ring swells `spread` px out of `el`, `times` over, on a
 * layer of the page's own, so that no scrolling list round `el` cuts it off. No movement, as ring.
 */
export function ripple(el, { spread = 40, times = 2 } = {}) {
  const r = el?.getBoundingClientRect?.();
  if (!r?.width || !document.body) return;
  const wave = document.createElement('div');
  wave.className = 'ripple';
  wave.style.cssText = `left: ${r.left}px; top: ${r.top}px; width: ${r.width}px; height: ${r.height}px;`
    + ` border-radius: ${globalThis.getComputedStyle?.(el).borderRadius || '0'}`;
  document.body.append(wave);
  const run = wave.animate?.([
    { boxShadow: '0 0 0 0 rgba(139, 125, 255, 0.9)' },
    { boxShadow: `0 0 0 ${spread}px rgba(139, 125, 255, 0)` },
  ], { duration: 900, easing: 'ease-out', iterations: times });
  if (run) run.onfinish = run.oncancel = () => wave.remove();
  else wave.remove();
}

/** "Look here", for what is far from the pointer: the ring, and `el` swells and settles in a third of a second. */
export function beckon(el) {
  ring(el);
  if (fullMotion()) {
    el?.animate?.([{ transform: 'none' }, { transform: 'scale(1.22)', offset: 0.4 }, { transform: 'none' }],
      { duration: 340, easing: 'ease-out' });
  }
}

/** "No": 3 x 6 px sideways in 300 ms (a red flash with gentle motion). */
export function shake(el) {
  if (!el) return Promise.resolve();
  const a = fullMotion()
    ? el.animate([
      { transform: 'translateX(0)' }, { transform: 'translateX(-6px)' }, { transform: 'translateX(6px)' },
      { transform: 'translateX(-6px)' }, { transform: 'translateX(6px)' }, { transform: 'translateX(-6px)' },
      { transform: 'translateX(0)' },
    ], { duration: 300, easing: 'ease-in-out' })
    : el.animate([{ boxShadow: '0 0 0 3px rgba(240,107,107,.8)' }, { boxShadow: '0 0 0 0 rgba(240,107,107,0)' }], { duration: 400 });
  return settle([a], 800);
}

/** Something a prop has just earned (a check): it pops in, 0 -> 1.15 -> 1 (full motion only). */
export function pop(el, delay = 0) {
  if (!el || !fullMotion()) return Promise.resolve();
  return settle([el.animate([
    { transform: 'scale(0)', opacity: 0 }, { transform: 'scale(1.15)', opacity: 1, offset: 0.6 }, { transform: 'none', opacity: 1 },
  ], { duration: 320, delay, easing: 'ease-out', fill: 'backwards' })], 900 + delay);
}

/**
 * "Here, next": a short attention pulse on `el` (the call to action): a ring swells out of it twice
 * (`ring`, an element laid over it), and the button breathes with it. Gentle motion: the ring only glows.
 */
export function nudge(el, ring) {
  if (!el) return Promise.resolve();
  const full = fullMotion();
  const anims = [];
  if (ring) {
    anims.push(ring.animate(full
      ? [{ opacity: 0.9, transform: 'scale(1)' }, { opacity: 0, transform: 'scale(1.18, 1.5)' }]
      : [{ opacity: 0 }, { opacity: 0.9 }, { opacity: 0 }], { duration: 900, iterations: full ? 2 : 1, easing: 'ease-out' }));
  }
  if (full) {
    anims.push(el.animate([{ transform: 'none' }, { transform: 'scale(1.04)', offset: 0.3 }, { transform: 'none' }],
      { duration: 900, iterations: 2, easing: 'ease-in-out' }));
  }
  return settle(anims, 2400);
}
