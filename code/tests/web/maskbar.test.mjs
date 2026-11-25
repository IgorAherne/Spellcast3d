// The Omni-Texture tools' own parts (static/js/stages/maskbar.js): what counts as masked, the square "Render Masked"
// sends, the steps around the one shown, and how hand paint is saved.  node --test code/tests/web/maskbar.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

globalThis.addEventListener ??= () => {};
globalThis.document ??= { addEventListener() {}, removeEventListener() {} };
const { maskOf, saveHandPaint, squareTan, stepsAround } = await import(new URL('../../spellcast/static/js/stages/maskbar.js', import.meta.url));

test('the mask is where the overlay covers at least half way', () => {
  // painted whole, an edge texel past half way, one just at half way, one under it, untouched
  const { mask, count } = maskOf(new Uint8Array([255, 200, 128, 127, 0]));
  assert.deepEqual([...mask], [255, 255, 255, 0, 0]);
  assert.equal(count, 3);
});

test('the square crop of the viewport spans the angle of its shorter side', () => {
  const half = Math.tan((32 * Math.PI) / 360);
  assert.ok(Math.abs(squareTan(32, 1600, 900) - half) < 1e-12);            // wide: the full height
  assert.ok(Math.abs(squareTan(32, 1050, 1350) - half * (1050 / 1350)) < 1e-12);   // tall: the width's share of it
});

test('the steps before and after the model shown', () => {
  const steps = ['a', 'b', 'c'];
  assert.deepEqual(stepsAround(steps, 'b'), { index: 1, before: 'a', after: 'c' });
  assert.deepEqual(stepsAround(steps, 'a'), { index: 0, before: null, after: 'b' });
  assert.deepEqual(stepsAround(steps, 'c'), { index: 2, before: 'b', after: null });
  assert.deepEqual(stepsAround(undefined, 'a'), { index: -1, before: null, after: null });
});

test('hand paint is saved as its patch into the step painted, and nothing is sent without one', async () => {
  const sent = [];
  globalThis.fetch = async (path, init) => {
    sent.push([path, init.method, init.body]);
    return { status: 200, text: async () => '{"id": "p_1"}' };
  };
  const step = { project: 'Demo', propId: 'p_1', glb: 'omnitexture/ot_001/retouch_002/omnitexture.glb' };
  assert.equal(await saveHandPaint(step, { width: 2, height: 2, data: new Uint8Array(16), patch: null }), null);
  assert.equal(sent.length, 0);
  const patch = new Uint8Array(16).fill(255, 0, 4);
  assert.deepEqual(await saveHandPaint(step, { width: 2, height: 2, data: new Uint8Array(16), patch }), { id: 'p_1' });
  assert.deepEqual(sent, [['/api/projects/Demo/props/p_1/omnitexture/paint?glb=omnitexture%2Fot_001%2Fretouch_002%2Fomnitexture.glb&width=2&height=2', 'PUT', patch]]);
});
