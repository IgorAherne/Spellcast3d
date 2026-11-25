// The motion level of the web UI (static/js/shell/anim.js): full by default, whatever the system says
// (Windows asks for less motion whenever its Animation effects are off), or gentle when the viewer
// chose "Reduced".   node --test code/tests/web/motion.test.mjs
// (code/tests/test_web.py runs every code/tests/web/*.test.mjs when Node.js is installed)
import assert from 'node:assert/strict';
import { test } from 'node:test';

// What a browser page has, as far as anim.js could use it. The system asks for less motion throughout:
// that is the artist's Windows, and it must not matter.
globalThis.matchMedia = () => ({ matches: true, addEventListener: () => {} });
const saved = new Map();
let storageWorks = true;
const storage = {
  getItem: (k) => { if (!storageWorks) throw new Error('blocked'); return saved.get(k) ?? null; },
  setItem: (k, v) => { if (!storageWorks) throw new Error('blocked'); saved.set(k, String(v)); },
  removeItem: (k) => { if (!storageWorks) throw new Error('blocked'); saved.delete(k); },
};
Object.defineProperty(globalThis, 'localStorage', { value: storage, configurable: true });
globalThis.document = { documentElement: { dataset: {} } };
const html = document.documentElement.dataset;

const ANIM = new URL('../../spellcast/static/js/shell/anim.js', import.meta.url);
let loads = 0;
/** anim.js as a page loading it afresh finds it (it reads the stored choice once). */
const freshAnim = () => import(`${ANIM.href}?page=${++loads}`);

test('full motion is the default, even while the system asks for less', async () => {
  saved.clear();
  const { initMotion, motionLevel, motionPref, fullMotion } = await freshAnim();
  initMotion();
  assert.equal(motionPref(), 'full');
  assert.equal(motionLevel(), 'full');
  assert.equal(fullMotion(), true);
  assert.equal(html.motion, 'full');
});

test('"Reduced" is gentle, and this browser remembers it; "Full" stores nothing', async () => {
  saved.clear();
  const { motionLevel, setMotionPref } = await freshAnim();
  setMotionPref('reduced');
  assert.equal(motionLevel(), 'gentle');
  assert.equal(html.motion, 'gentle', 'the style sheets follow at once');
  assert.equal(saved.get('spellcast3d.motion'), 'reduced');
  const next = await freshAnim();
  next.initMotion();
  assert.equal(next.motionPref(), 'reduced', 'the next page load keeps it');
  next.setMotionPref('full');
  assert.equal(html.motion, 'full');
  assert.equal(saved.has('spellcast3d.motion'), false, 'full motion is the default: nothing is stored');
});

test('the choices of earlier versions read as full motion', async () => {
  for (const old of ['always', 'auto', 'nonsense']) {
    saved.clear();
    saved.set('spellcast3d.motion', old);
    const { initMotion, motionPref } = await freshAnim();
    initMotion();
    assert.equal(motionPref(), 'full', old);
    assert.equal(html.motion, 'full', old);
  }
});

test('a browser that keeps nothing still gets the choice for this page', async () => {
  saved.clear();
  storageWorks = false;
  try {
    const { initMotion, motionPref, setMotionPref } = await freshAnim();
    initMotion();
    assert.equal(motionPref(), 'full');
    setMotionPref('reduced');
    assert.equal(motionPref(), 'reduced');
    assert.equal(html.motion, 'gentle');
    setMotionPref('nonsense');
    assert.equal(motionPref(), 'full');
    assert.equal(html.motion, 'full');
  } finally {
    storageWorks = true;
  }
});
