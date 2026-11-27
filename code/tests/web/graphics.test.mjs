// How the browser draws the 3D view, as far as the artist chooses (static/js/shell/graphics.js): with
// anti-aliasing unless Settings' "Faster 3D view" is on.   node --test code/tests/web/graphics.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

const saved = new Map();
let storageWorks = true;
const storage = {
  getItem: (k) => { if (!storageWorks) throw new Error('blocked'); return saved.get(k) ?? null; },
  setItem: (k, v) => { if (!storageWorks) throw new Error('blocked'); saved.set(k, String(v)); },
  removeItem: (k) => { if (!storageWorks) throw new Error('blocked'); saved.delete(k); },
};
Object.defineProperty(globalThis, 'localStorage', { value: storage, configurable: true });

const GRAPHICS = new URL('../../spellcast/static/js/shell/graphics.js', import.meta.url);
let loads = 0;
/** graphics.js as a page loading it afresh finds it (it reads the stored choice once). */
const freshPage = () => import(`${GRAPHICS.href}?page=${++loads}`);

test('the 3D view is anti-aliased unless the artist chose the faster one, which this browser remembers', async () => {
  saved.clear();
  const page = await freshPage();
  assert.equal(page.plainEdges(), false);
  page.setPlainEdges(true);
  assert.equal(page.plainEdges(), true);
  assert.equal((await freshPage()).plainEdges(), true, 'the next page load keeps it');
  page.setPlainEdges(false);
  assert.equal(saved.size, 0, 'anti-aliasing is the default: nothing is stored');
  assert.equal((await freshPage()).plainEdges(), false);
});

test('a browser that keeps nothing still gets the choice for this page', async () => {
  saved.clear();
  storageWorks = false;
  try {
    const page = await freshPage();
    assert.equal(page.plainEdges(), false);
    page.setPlainEdges(true);
    assert.equal(page.plainEdges(), true);
  } finally {
    storageWorks = true;
  }
});
