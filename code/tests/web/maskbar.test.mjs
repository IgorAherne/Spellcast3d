// The Omni-Texture tools' own parts (static/js/stages/maskbar.js): what counts as masked, the square "Render Masked"
// sends and the camera it sees through (the viewer's lens, bent or not), the steps around the one shown, and how hand
// paint is saved.  node --test code/tests/web/maskbar.test.mjs
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { test } from 'node:test';

// The page's import map (index.html), for Node.
const STATIC = new URL('../../spellcast/static/', import.meta.url);
registerHooks({
  resolve(specifier, context, next) {
    if (specifier === 'three') return { url: new URL('vendor/three/build/three.module.js', STATIC).href, shortCircuit: true };
    if (specifier.startsWith('three/addons/')) {
      return { url: new URL(`vendor/three/examples/jsm/${specifier.slice('three/addons/'.length)}`, STATIC).href, shortCircuit: true };
    }
    return next(specifier, context);
  },
});
globalThis.addEventListener ??= () => {};
globalThis.document ??= { addEventListener() {}, removeEventListener() {} };
const THREE = await import('three');
const { LensCamera } = await import(new URL('js/viewer3d.js', STATIC));
const {
  askPlace, cameraOf, maskOf, perspectiveOf, saveHandPaint, squareShare, stepsAround,
} = await import(new URL('js/stages/maskbar.js', STATIC));

const near = (a, b) => Math.abs(a - b) < 1e-12;

test('the mask is where the overlay covers at least half way', () => {
  // painted whole, an edge texel past half way, one just at half way, one under it, untouched
  const { mask, count } = maskOf(new Uint8Array([255, 200, 128, 127, 0]));
  assert.deepEqual([...mask], [255, 255, 255, 0, 0]);
  assert.equal(count, 3);
});

test('the square crop of the viewport is as high as its shorter side', () => {
  assert.equal(squareShare(1600, 900), 1);             // wide: the full height
  assert.equal(squareShare(1050, 1350), 1050 / 1350);  // tall: the width's share of it
});

test('"Inverse perspective": off is a camera\'s own view; on, its slider goes through no perspective at a quarter to inverse', () => {
  assert.deepEqual([0, 0.25, 0.5, 1].map((amount) => perspectiveOf({ on: true, amount })), [1, 0, -1, -3]);
  assert.equal(perspectiveOf({ on: false, amount: 1 }), 1);
});

test('the camera "Render Masked" sends is the viewport\'s square through the viewer\'s lens, in the model\'s own space', () => {
  const camera = new LensCamera(32, 16 / 9, 0.01, 100);
  camera.position.set(0, 0, 4);                        // looking at the point it turns about, the origin
  const model = new THREE.Group();
  model.scale.setScalar(2);                            // (the model's own units are half the viewer's)
  const viewer = { camera, current: new THREE.Group().add(model), renderer: { domElement: { clientWidth: 1600, clientHeight: 900 } } };
  const half = Math.tan((32 * Math.PI) / 360);
  const plain = cameraOf(viewer);
  assert.deepEqual(Object.keys(plain), ['eye', 'forward', 'up', 'tan'], 'a camera sends no lens');
  assert.deepEqual([plain.eye, plain.forward, plain.up].map((v) => v.map((n) => n + 0)), [[0, 0, 2], [0, 0, -1], [0, 1, 0]]);
  assert.ok(near(plain.tan, half));
  // Bent: half the square's side at depth z is wide + tan * z, and what lies at the point orbited (depth 2 in the
  // model's units) keeps the size the camera gave it.
  camera.perspective = perspectiveOf({ on: true, amount: 0.5 });
  const bent = cameraOf(viewer);
  assert.ok(near(bent.tan, -half) && near(bent.wide, half * 4));
  assert.ok(near(bent.wide + bent.tan * 2, plain.tan * 2));
  // A tall viewport: the square is its width's share of the height, lens and all.
  viewer.renderer.domElement = { clientWidth: 900, clientHeight: 1800 };
  const tall = cameraOf(viewer);
  assert.ok(near(tall.tan, -half / 2) && near(tall.wide, half * 2));
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

/**
 * A viewport 800 x 600 seen as 40 x 30 pixels of 20 (paint3d.js Session.sight's shape): the model where `body` says
 * ([x0, y0, x1, y1] in its pixels, the end left out), masked where `masked` says.
 */
function sightOf(body, masked) {
  const sight = { width: 40, height: 30, model: new Uint8Array(1200), cover: new Uint8Array(1200) };
  const fill = (array, [x0, y0, x1, y1], value) => {
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) array[y * 40 + x] = value;
  };
  if (body) fill(sight.model, body, 1);
  if (masked) fill(sight.cover, masked, 255);
  return sight;
}
const VIEW = { w: 800, h: 600 };
const BOX = { w: 200, h: 80 };
/** The rect a placed box takes. */
const boxAt = ({ x, y }) => ({ x0: x, y0: y, x1: x + BOX.w, y1: y + BOX.h });

test('the box for the mask\'s words goes beside the mask, in the room the model leaves, as near the mask as there is room', () => {
  // A figure 300 .. 500 px across, masked at 320 .. 400 x 280 .. 360: the box goes to its left, 16 px off it.
  const figure = [15, 2, 25, 28];
  const at = askPlace(sightOf(figure, [16, 14, 20, 18]), VIEW, BOX);
  const r = boxAt(at);
  assert.ok(r.x1 <= 300 - 16, `left of the figure: ${JSON.stringify(r)}`);
  assert.ok(r.x1 > 300 - 16 - 20, 'as near it as it can be');
  assert.ok(r.y0 < 360 && r.y1 > 280, 'beside the mask, at its height');
  // Masked on the figure's right: the box goes to its right.
  assert.ok(askPlace(sightOf(figure, [21, 14, 25, 18]), VIEW, BOX).x >= 500 + 16);
  // A button where the box would go (the viewport's own): it keeps clear of it, to the figure's other side.
  const button = { x0: 0, y0: 200, x1: 300, y1: 440 };
  const clear = boxAt(askPlace(sightOf(figure, [16, 14, 20, 18]), VIEW, BOX, { clear: [button] }));
  assert.ok(clear.x0 >= 500 + 16 || clear.y1 <= 200 - 16 || clear.y0 >= 440 + 16, JSON.stringify(clear));
});

test('what the black cover hides is room; where the model leaves none the box covers as little of it as can be, never the mask', () => {
  // The model fills the viewport but its sides are under the black cover (only 250 .. 550 is seen).
  const everywhere = [0, 0, 40, 30];
  const mask = [18, 13, 22, 17];
  const r = boxAt(askPlace(sightOf(everywhere, mask), VIEW, BOX, { square: { x0: 250, y0: 0, x1: 550, y1: 600 } }));
  assert.ok(r.x1 <= 250 - 16 || r.x0 >= 550 + 16, `on the black: ${JSON.stringify(r)}`);
  // Seen whole, it covers the model somewhere - not the mask.
  const masked = { x0: 360, y0: 260, x1: 440, y1: 340 };
  const on = boxAt(askPlace(sightOf(everywhere, mask), VIEW, BOX));
  assert.ok(on.x1 + 16 <= masked.x0 || on.x0 - 16 >= masked.x1 || on.y1 + 16 <= masked.y0 || on.y0 - 16 >= masked.y1);
  // ... and a button rather not: with one over the whole left half, the box goes right, still off the mask.
  const right = boxAt(askPlace(sightOf(everywhere, mask), VIEW, BOX, { clear: [{ x0: 0, y0: 0, x1: 400, y1: 600 }] }));
  assert.ok(right.x0 >= 400 + 16 && (right.y1 + 16 <= masked.y0 || right.y0 - 16 >= masked.y1 || right.x0 - 16 >= masked.x1),
    JSON.stringify(right));
  // Nothing masked in sight: no box.
  assert.equal(askPlace(sightOf(everywhere, null), VIEW, BOX), null);
});
