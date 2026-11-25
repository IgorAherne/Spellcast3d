// The Projection painter's strokes without a GPU (stages/paintstroke.js, used by paint3d.js): the brush size on its
// slider and under Shift + sideways, the pens' pressure (Windows Ink's off first reading, pens that report none), the
// dabs a stroke is made of (round and close, tapering with the pressure instead of stepping), the undo tiles a
// stroke may have changed, and the undo history's memory. Then the painting through a lost WebGL context (paint3d.js
// on a stand-in card, standin_card.mjs; the saver of stages/paintbar.js): never read back or saved once the context
// went, filled again with what the bake's files hold once it is back. Then the overlay (Omni-Texture's mask): a layer
// of its own that the base colour, its saves and a model never painted with it know nothing of, one undo history for
// both, kept and put back alone. Then the patch an Omni-Texture save sends: the texels the hand changed alone, told
// from the files' bytes as the card holds them. Last the tool bar (stages/paintbar.js) on a fake page: the dock's
// shape, a stage's own tool, and a flush that waits for every save.
//   node --test code/tests/web/paint.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { letGo, moveTo, press, pressAt, standInModel, standInViewer, strokeAt } from './standin_card.mjs';

const STATIC = new URL('../../spellcast/static/js/', import.meta.url);
const {
  DEFAULT_SIZE, MAX_SIZE, MIN_SIZE, createPressureReader, createStroke, createUndo, dirtyTiles, dragSize, patchOf,
  posToSize, sizeToPos, texelHex, tileRects,
} = await import(new URL('stages/paintstroke.js', STATIC));

const pen = (pressure) => ({ pointerType: 'pen', pressure });

test('the size slider is logarithmic and Shift + sideways sweeps it within its range', () => {
  assert.equal(posToSize(0), MIN_SIZE);
  assert.equal(posToSize(1000), MAX_SIZE);
  for (const size of [MIN_SIZE, 5, 12, DEFAULT_SIZE, 100, MAX_SIZE]) assert.ok(Math.abs(posToSize(sizeToPos(size)) - size) <= 1, size);
  assert.ok(sizeToPos(10) > 250, 'small sizes get a good part of the slider');
  assert.equal(dragSize(DEFAULT_SIZE, 0), DEFAULT_SIZE);
  assert.ok(dragSize(DEFAULT_SIZE, 40) > DEFAULT_SIZE && dragSize(DEFAULT_SIZE, -40) < DEFAULT_SIZE);
  assert.equal(dragSize(DEFAULT_SIZE, 10000), MAX_SIZE);
  assert.equal(dragSize(DEFAULT_SIZE, -10000), MIN_SIZE);
  assert.equal(dragSize(MIN_SIZE, 250), MAX_SIZE, 'the whole range in 250 px');
});

test('a mouse and a finger have no pressure; a pen has, unless it only ever says 0.5', () => {
  const r = createPressureReader();
  assert.equal(r.read({ pointerType: 'mouse', pressure: 0.5 }), null);
  assert.equal(r.read({ pointerType: 'touch', pressure: 0.3 }), null);
  r.start();
  assert.equal(r.read(pen(0.3)), 0.3);
  assert.equal(r.read(pen(1.4)), 1, 'clamped');
  r.end();
  r.start();
  for (let i = 0; i < 5; i++) assert.equal(r.read(pen(0.5)), 0.5, 'a few 0.5 readings are a real pressure');
  assert.equal(r.read(pen(0.5)), null, 'six in a row with nothing else: a pen without pressure');
  r.end();
  assert.equal(r.penHasPressure, false);
  r.start();
  assert.equal(r.read(pen(0.5)), null, 'remembered for the next stroke');
  assert.equal(r.read(pen(0.42)), 0.42, 'until it reports a real value');
  assert.equal(r.penHasPressure, true);
});

/** The dabs of a stroke through `points` ([x, y, pressure]), one sample per ms. */
function dabsOf(points, options) {
  const s = createStroke({ radius: 10, ...options });
  const out = [];
  points.forEach(([x, y, p], i) => out.push(...s.add(x, y, p, i * 5)));
  out.push(...s.end());
  return out;
}

const line = (n, step, pressure = () => 1) => Array.from({ length: n }, (_, i) => [i * step, 0, pressure(i, n)]);

test('a tap paints one dot at its pressure; a pen that said 0 on its tap still paints one', () => {
  const [dot] = dabsOf([[5, 5, 0.6]]);
  assert.deepEqual(dot, { x: 5, y: 5, r: 6, a: 1 });
  assert.equal(dabsOf([[5, 5, 0]])[0].r, 5);
  assert.equal(dabsOf([[5, 5, null]])[0].r, 10, 'a mouse: full size');
});

test('Windows Ink\'s off first reading takes the second one\'s value', () => {
  const spike = dabsOf([[0, 0, 1], [3, 0, 0.3], [6, 0, 0.3], [9, 0, 0.3]]);
  assert.ok(Math.abs(spike[0].r - 3) < 1e-9, `a spike at the touch: ${spike[0].r}`);
  const zero = dabsOf([[0, 0, 0], [3, 0, 0.4], [6, 0, 0.4]]);
  assert.ok(Math.abs(zero[0].r - 4) < 1e-9, 'a zero at the touch');
  const real = dabsOf([[0, 0, 0.1], [3, 0, 0.3], [6, 0, 0.5]]);
  assert.ok(Math.abs(real[0].r - 1) < 1e-9, 'a light touch is kept: the stroke tapers from it');
});

test('dabs follow the path closely enough to keep the edge round, and land the same however the samples arrive', () => {
  const dabs = dabsOf(line(40, 3));
  assert.ok(dabs.length > 20);
  for (let i = 1; i < dabs.length; i++) {
    const gap = Math.hypot(dabs[i].x - dabs[i - 1].x, dabs[i].y - dabs[i - 1].y);
    assert.ok(gap <= 0.1 * 10 + 1e-6, `gap ${gap} at ${i}`);
    assert.ok(Math.abs(dabs[i].y) < 1e-9);
  }
  assert.ok(Math.abs(dabs.at(-1).x - 39 * 3) < 1e-6, 'it ends at the last sample');
  // A tiny brush still gets a dab at least every half pixel.
  const tiny = dabsOf(line(10, 3), { radius: 0.5 });
  for (let i = 1; i < tiny.length; i++) assert.ok(tiny[i].x - tiny[i - 1].x <= 0.5 + 1e-9);
});

test('pressure tapers the line instead of stepping it: a light start grows smoothly, a lift thins it to a point', () => {
  // A pen pressed from a light touch to full over 10 samples, then lifted over the last 10, sampled every 4 px.
  const pts = line(40, 4, (i, n) => Math.min(1, Math.max(0.02, Math.min(i, n - 1 - i) / 10)));
  const dabs = dabsOf(pts);
  assert.ok(dabs[0].r < 0.5, `the tip is a point: ${dabs[0].r}`);
  assert.ok(dabs.at(-1).r < 1.5, `the end thins out: ${dabs.at(-1).r}`);
  let steepest = 0;
  for (let i = 1; i < dabs.length; i++) {
    const along = Math.hypot(dabs[i].x - dabs[i - 1].x, dabs[i].y - dabs[i - 1].y);
    steepest = Math.max(steepest, Math.abs(dabs[i].r - dabs[i - 1].r) / Math.max(along, 1e-9));
  }
  // The pen's own slope is 10 px of radius over 40 px of path (0.25); smoothing and the curve keep it close to that.
  assert.ok(steepest < 0.4, `no step in the width: steepest ${steepest.toFixed(3)} px of radius per px`);
  assert.ok(Math.max(...dabs.map((d) => d.r)) > 9, 'full pressure reaches the full size');
});

test('Pressure sizes the brush, Fade sets its opacity, together both', () => {
  const pts = line(10, 3, () => 0.4);
  const sized = dabsOf(pts, { bySize: true, byFade: false });
  assert.ok(sized.every((d) => Math.abs(d.r - 4) < 1e-9 && d.a === 1));
  const faded = dabsOf(pts, { bySize: false, byFade: true });
  assert.ok(faded.every((d) => d.r === 10 && Math.abs(d.a - 0.4) < 1e-9));
  const both = dabsOf(pts, { bySize: true, byFade: true });
  assert.ok(both.every((d) => Math.abs(d.r - 4) < 1e-9 && Math.abs(d.a - 0.4) < 1e-9));
  const mouse = dabsOf(line(10, 3, () => null), { bySize: true, byFade: true });
  assert.ok(mouse.every((d) => Math.abs(d.r - 10) < 1e-9 && Math.abs(d.a - 1) < 1e-9), 'a mouse paints full and opaque');
});

test('a hand at rest adds no dabs', () => {
  const still = dabsOf([[0, 0, 1], [0.1, 0, 1], [0.2, 0.1, 1], [0, 0, 1]]);
  assert.equal(still.length, 1);
});

// Two triangles side by side on screen and in UV: (u, v) per corner and (x, y, w) per corner.
const quad = {
  uv: new Float32Array([0, 0, 0.5, 0, 0, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 1]),
  // The first winds as a front face does on screen (y down); the second the other way: it faces away.
  screen: new Float32Array([0, 0, 1, 0, 100, 1, 100, 0, 1, 200, 200, 1, 100, 200, 1, 200, 100, 1]),
};

test('the undo tiles: what a stroke can reach, grown by the gutter, back faces only on two-sided models', () => {
  const near = dirtyTiles({ ...quad, box: { x0: 10, y0: 10, x1: 20, y1: 20 }, size: 64, tiles: 4 });
  assert.deepEqual([...near], [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const away = dirtyTiles({ ...quad, box: { x0: 150, y0: 150, x1: 160, y1: 160 }, size: 64, tiles: 4 });
  assert.equal(away.reduce((a, b) => a + b), 0, 'the second triangle faces away');
  const both = dirtyTiles({ ...quad, box: { x0: 150, y0: 150, x1: 160, y1: 160 }, size: 64, tiles: 4, twoSided: true });
  assert.ok(both.reduce((a, b) => a + b) > 0);
  const off = dirtyTiles({ ...quad, box: { x0: 500, y0: 500, x1: 510, y1: 510 }, size: 64, tiles: 4 });
  assert.equal(off.reduce((a, b) => a + b), 0, 'off the triangles: nothing');
  const grown = dirtyTiles({ ...quad, box: { x0: 10, y0: 10, x1: 20, y1: 20 }, size: 64, tiles: 4, margin: 2 });
  assert.equal(grown[2], 1, 'the margin reaches the next tile');
  const behind = { uv: quad.uv.slice(0, 6), screen: new Float32Array([0, 0, -1, 0, 0, -1, 0, 0, -1]) };
  assert.equal(dirtyTiles({ ...behind, box: { x0: -1e9, y0: -1e9, x1: 1e9, y1: 1e9 }, size: 64, tiles: 4 }).reduce((a, b) => a + b), 0);
});

test('the tiles become few rectangles that cover them exactly', () => {
  const tiles = 6;
  const flags = new Uint8Array(tiles * tiles);
  for (const [x, y] of [[1, 1], [2, 1], [1, 2], [2, 2], [4, 2], [0, 5], [5, 5]]) flags[y * tiles + x] = 1;
  const rects = tileRects(flags, tiles, 60);
  const covered = new Uint8Array(tiles * tiles);
  for (const { x, y, w, h } of rects) {
    for (let ty = y / 10; ty < (y + h) / 10; ty++) {
      for (let tx = x / 10; tx < (x + w) / 10; tx++) {
        assert.equal(covered[ty * tiles + tx], 0, 'no overlap');
        covered[ty * tiles + tx] = 1;
      }
    }
  }
  assert.deepEqual([...covered], [...flags]);
  assert.equal(rects.length, 4, 'the 2x2 block is one rectangle');
});

test('undo keeps at least five strokes, more while they fit in its memory', () => {
  const u = createUndo({ keep: 5, budget: 100, max: 8 });
  const dropped = [];
  for (let i = 0; i < 7; i++) dropped.push(...u.add({ i, bytes: 40 }));
  assert.deepEqual(dropped.map((s) => s.i), [0, 1], 'over the budget: the oldest go, down to five');
  const small = createUndo({ keep: 5, budget: 1000, max: 8 });
  for (let i = 0; i < 10; i++) small.add({ i, bytes: 1 });
  assert.equal(small.newest.i, 9);
  let n = 0;
  while (small.undo()) n++;
  assert.equal(n, 8, 'small strokes: up to the most it keeps');
  assert.equal(small.redo().i, 2);
  const gone = small.add({ i: 'new', bytes: 1 });
  assert.equal(gone.length, 7, 'a new stroke after undos: what was undone cannot come back');
  assert.equal(small.canRedo, false);
});

test('a history takes the steps of another: the same strokes to undo and to redo', () => {
  const first = createUndo();
  for (const i of [1, 2, 3]) first.add({ i, bytes: 1 });
  first.undo();
  const second = createUndo();
  second.take(first.all);
  first.clear();
  assert.equal(second.newest.i, 2);
  assert.equal(second.redo().i, 3);
  assert.deepEqual([second.undo().i, second.undo().i, second.undo().i, second.undo()], [3, 2, 1, null]);
});

test('the eyedropper averages the texels it reads, alpha left out', () => {
  assert.equal(texelHex(new Uint8Array([255, 0, 16, 7])), '#ff0010');
  assert.equal(texelHex(new Uint8Array([10, 20, 30, 0, 12, 22, 33, 255])), '#0b1520', 'rounded per channel');
  assert.equal(texelHex(new Uint8Array(0)), null);
});

// ---- the painting through a lost WebGL context -------------------------------------------------------------------

// A page as far as the tool bar uses one: elements with attributes, children, classes, text and listeners.
class FakeNode {}
class FakeEl extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attrs = {};
    this.className = '';
    this.listeners = {};
    this.hidden = false;
    this.parentElement = null;
    this.style = { setProperty(name, value) { this[name] = value; } };
  }

  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter((f) => f !== fn); }
  click() { (this.listeners.click || []).forEach((fn) => fn({ type: 'click' })); }
  blur() {}
  getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; }
  append(...nodes) {
    for (const node of nodes) if (node instanceof FakeEl) node.parentElement = this;
    this.children.push(...nodes);
  }

  get textContent() { return this.children.map((c) => (typeof c === 'string' ? c : c.textContent)).join(''); }
  set textContent(v) { this.children = [String(v)]; }
  get classList() {
    const el = this;
    const list = () => el.className.split(/\s+/).filter(Boolean);
    return {
      contains: (c) => list().includes(c),
      toggle: (c, on = !list().includes(c)) => {
        el.className = on ? [...new Set([...list(), c])].join(' ') : list().filter((x) => x !== c).join(' ');
        return on;
      },
    };
  }
}
class FakeTemplate extends FakeEl {
  set innerHTML(html) { this.html = html; }
  get content() { return { firstChild: new FakeEl('svg') }; }
}
const pageKeys = [];   // the page's keydown listeners (the tool bar's Esc, Ctrl+Z, Ctrl+Y)
globalThis.Node = FakeNode;
globalThis.addEventListener ??= () => {};
globalThis.document = {
  addEventListener: (type, fn) => { if (type === 'keydown') pageKeys.push(fn); },
  removeEventListener: (type, fn) => { if (pageKeys.includes(fn)) pageKeys.splice(pageKeys.indexOf(fn), 1); },
  activeElement: null,
  querySelector: () => null,
  createElement: (tag) => (tag === 'template' ? new FakeTemplate(tag) : new FakeEl(tag)),
};

/** A key pressed on the page (`mods`: ctrlKey...), as far as the listeners let it go. */
function keyDown(key, mods = {}) {
  let stopped = false;
  const e = { type: 'keydown', key, ...mods, preventDefault() {}, stopImmediatePropagation() { stopped = true; } };
  for (const fn of [...pageKeys]) {
    if (stopped) break;
    fn(e);
  }
}

const { createPainter } = await import(new URL('paint3d.js', STATIC));
const { createPaintBar, createSaver } = await import(new URL('stages/paintbar.js', STATIC));
const { saveHandPaint } = await import(new URL('stages/maskbar.js', STATIC));

const SIZE = 8;

/** The painter's ui, telling what it was told (lost / restored) and what changed; `use` is the tool up. */
function standInUi() {
  const told = [];
  const changes = [];
  const ui = {
    told, changes, use: 'marker',
    tool: () => ui.use, size: () => 20, colour: () => (ui.use === 'overlay' ? '#ff00ff' : '#201314'),
    bySize: () => false, byFade: () => false, pressure: () => null, strokeStart() {}, strokeEnd() {},
    changed: (layer) => changes.push(layer), source() {}, picked() {},
    lost: () => told.push('lost'), restored: () => told.push('restored'),
  };
  return ui;
}

/** Paints 2 x 2 texels at (2, 2) over the texture, as a stroke would, with its undo step. */
function paintTexels(session) {
  const rect = { x: 2, y: 2, w: 2, h: 2 };
  const before = session.read(session.paint, rect);
  const strokes = Uint8Array.from({ length: 16 }, (_, i) => (i % 4 === 3 ? 255 : 9));
  assert.ok(session.put({ size: SIZE, tiles: [{ ...rect, data: strokes }], steps: [{ rects: [{ ...rect, data: before }], bytes: 16, at: 1 }], undone: [] }));
}

test('once the WebGL context went, the painting is never read back, and it comes back as the bake\'s files hold it', async () => {
  const { viewer, renderer, picture } = standInViewer(SIZE);
  const ui = standInUi();
  const painter = createPainter(viewer, ui);
  const s = painter.session();
  assert.deepEqual((await s.readAll()).data, picture, 'filled with the model\'s own picture');
  paintTexels(s);
  assert.notDeepEqual((await s.readAll()).data, picture);
  assert.notEqual(s.version, s.saved, 'something to save');

  renderer.loseContext();
  assert.deepEqual(ui.told, ['lost'], 'the artist is told');
  await assert.rejects(s.readAll(), (e) => e.lost === true && /graphics card was reset/.test(e.message));
  assert.equal(s.undo(), false, 'no Ctrl+Z onto a texture the card no longer holds');
  assert.equal(s.keep(), null, 'nor is it kept for the model\'s return');
  assert.equal(viewer.paintHook.down(press), true, 'a press paints nothing (and orbits nothing) ...');
  assert.deepEqual(ui.told, ['lost', 'lost'], '... and says why');

  renderer.restoreContext();
  assert.deepEqual(ui.told, ['lost', 'lost', 'restored']);
  assert.deepEqual((await s.readAll()).data, picture, 'nothing saved yet: the picture the model came with');
  assert.ok(!s.history.canUndo && s.version === s.saved, 'the history starts afresh, with nothing to save');
  painter.detach();
});

test('after a save, a lost context brings back the painting saved, without what came after it', async () => {
  const { viewer, renderer, picture } = standInViewer(SIZE);
  const s = createPainter(viewer, standInUi()).session();
  paintTexels(s);
  const saved = await s.readAll();
  s.savedAs(saved);
  assert.equal(s.version, s.saved);
  assert.ok(s.undo(), 'the stroke taken back, not saved yet');
  assert.deepEqual((await s.readAll()).data, picture);
  renderer.loseContext();
  renderer.restoreContext();
  assert.deepEqual((await s.readAll()).data, saved.data);
  assert.equal(s.version, s.saved);
});

test('a read the context was lost under is refused, though three.js hands back zeros', async () => {
  const { viewer, renderer } = standInViewer(SIZE);
  const s = createPainter(viewer, standInUi()).session();
  let finish;
  renderer.slowRead = new Promise((resolve) => { finish = resolve; });
  const reading = s.readAll();
  renderer.loseContext();
  renderer.restoreContext();
  finish();
  await assert.rejects(reading, (e) => e.lost === true);
});

test('a save on its way when the context goes: the model then shows what it saved', async () => {
  const { viewer, renderer, picture } = standInViewer(SIZE);
  const s = createPainter(viewer, standInUi()).session();
  paintTexels(s);
  const sent = await s.readAll();               // read whole, then the card is reset while it travels
  renderer.loseContext();
  renderer.restoreContext();
  assert.deepEqual((await s.readAll()).data, picture, 'filled again from what the files held before');
  s.savedAs(sent);
  assert.deepEqual((await s.readAll()).data, sent.data, 'what the files hold now');
  assert.equal(s.version, s.saved);
});

test('no painting is begun while the context is lost; a press says why, and the card\'s return is told', () => {
  const { viewer, renderer } = standInViewer(SIZE);
  const ui = standInUi();
  const painter = createPainter(viewer, ui);
  renderer.loseContext();
  assert.deepEqual(ui.told, [], 'nothing painted on screen: nothing lost');
  assert.equal(painter.session(), null);
  assert.match(painter.why, /graphics card was reset/);
  assert.equal(viewer.paintHook.down(press), true);
  assert.deepEqual(ui.told, ['lost']);
  renderer.restoreContext();
  assert.deepEqual(ui.told, ['lost', 'restored']);
  assert.ok(painter.session(), 'painted on once the card is back');
});

test('a model the viewer lets go of is read whole while a save holds it, and never once it is freed', async () => {
  const { viewer, picture } = standInViewer(SIZE);
  const s = createPainter(viewer, standInUi()).session();
  const release = s.hold();
  s.dispose();
  assert.deepEqual((await s.readAll()).data, picture);
  release();
  await assert.rejects(s.readAll(), (e) => e.lost === true, 'three.js would read a freed target as zeros');
});

test('the saver holds the texture until it is saved, hands the pixels back, and never sends a lost read', async () => {
  const sent = [];
  globalThis.fetch = async (path, init) => {
    sent.push([path, init.body]);
    return { status: 200, text: async () => '' };
  };
  const pixels = new Uint8Array(16).fill(7);
  const session = {
    version: 2, saved: 1, holds: 0, kept: null, read: null,
    hold() {
      this.holds++;
      return () => { this.holds--; };
    },
    readAll() { return this.read(); },
    savedAs(image) {
      this.kept = image;
      this.saved = image.version;
    },
  };
  session.read = async () => ({ width: 2, height: 2, data: pixels, version: session.version, fills: 1 });
  const saver = createSaver({ project: 'Demo', propId: 'p_1', glb: 'projection/bake_001/projection.glb', session });
  saver.schedule();
  assert.equal(session.holds, 1, 'held from the change on');
  await saver.flush();
  assert.equal(sent.length, 1);
  assert.match(sent[0][0], /\/projection\/paint\?width=2&height=2&glb=projection%2Fbake_001%2Fprojection\.glb$/);
  assert.equal(sent[0][1], pixels);
  assert.equal(session.kept.data, pixels, 'the pixels saved: what a lost context brings back');
  assert.equal(session.holds, 0, 'let go once saved');

  session.version = 3;
  session.read = async () => { throw Object.assign(new Error('reset'), { lost: true }); };
  const warn = console.warn;
  console.warn = () => {};
  try {
    saver.schedule();
    await saver.flush();
    await saver.flush();
  } finally {
    console.warn = warn;
  }
  assert.equal(sent.length, 1, 'nothing read of a lost texture is sent, nor tried again');
  assert.equal(session.holds, 0);
  assert.equal(saver.busy, false);
});

// ---- the overlay: a layer of its own over the base colour (Omni-Texture's mask) ------------------------------------

const TEX = 64;            // texels per side: an undo tile (32 per side) is 2 x 2 texels
const MAGENTA = [255, 0, 255];

/** A target's bytes on the stand-in card as they are now. */
const bytesOf = (renderer, rt) => renderer.memory.get(rt.texture).slice();
/** The RGB of texel (x, y) in RGBA bytes `TEX` texels wide. */
const rgbAt = (data, x, y) => [...data.subarray((y * TEX + x) * 4, (y * TEX + x) * 4 + 3)];

/** The coverage strokes over `rects` leave: a byte per texel, 255 under them. */
function coverage(...rects) {
  const out = new Uint8Array(TEX * TEX);
  for (const { x, y, w, h } of rects) for (let row = y; row < y + h; row++) out.fill(255, row * TEX + x, row * TEX + x + w);
  return out;
}

test('a model never painted with the overlay has none: no targets, no passes, nothing of it read', async () => {
  const { viewer, renderer, own } = standInViewer(TEX);
  const ui = standInUi();
  const s = createPainter(viewer, ui).session();
  const passes = renderer.renders;
  strokeAt(viewer, { x: 8, y: 8, w: 4, h: 4 });
  assert.equal(renderer.renders - passes, 4, 'a stroke: its snapshot, the mask cleared, the dabs, the paint');
  assert.deepEqual(ui.changes, ['base']);
  assert.ok(s.undo() && s.redo());
  assert.ok(s.keep().tiles.every((tile) => tile.layer === 'base'));
  assert.deepEqual((await s.readOverlay()).data, new Uint8Array(TEX * TEX), 'nothing covered');
  assert.equal(s.overlay, null);
  assert.equal(s.shown, null);
  assert.equal(own.map, s.paint.texture, 'the model shows the painted texture itself');
  const targets = [...renderer.memory.keys()].filter((texture) => texture.image.width === TEX);
  assert.deepEqual(new Set(targets), new Set([s.paint.texture, s.before.texture, s.islands.texture]));
});

test('an overlay stroke leaves the base and its version alone; the marker and the clone stamp leave the overlay alone', async () => {
  const { viewer, renderer, picture, own } = standInViewer(TEX);
  const ui = standInUi();
  const s = createPainter(viewer, ui).session();
  const saved = s.version;
  const a = { x: 8, y: 8, w: 6, h: 6 };
  ui.use = 'overlay';
  strokeAt(viewer, a);
  assert.deepEqual((await s.readAll()).data, picture, 'the base colour as it was: what a save sends');
  assert.equal(s.version, saved, 'nothing to save');
  assert.deepEqual(ui.changes, ['overlay']);
  assert.deepEqual((await s.readOverlay()).data, coverage(a));
  assert.equal(own.map, s.shown.texture, 'the model shows the composite');
  let shown = bytesOf(renderer, s.shown);
  assert.deepEqual(rgbAt(shown, 8, 8), MAGENTA);
  assert.deepEqual(rgbAt(shown, 13, 13), MAGENTA);
  assert.deepEqual(rgbAt(shown, 0, 0), rgbAt(picture, 0, 0), 'the base where nothing is covered');
  const passes = renderer.renders;
  strokeAt(viewer, { x: 20, y: 8, w: 2, h: 2 });
  assert.equal(renderer.renders - passes, 5, 'one pass more than a marker\'s stroke: the composite');

  const overlay = bytesOf(renderer, s.overlay);
  ui.use = 'marker';
  strokeAt(viewer, { x: 10, y: 10, w: 8, h: 8 });
  ui.use = 'clone';
  assert.equal(viewer.paintHook.down({ ...press, clientX: 40, clientY: 50, ctrlKey: true }), true, 'where it copies from');
  strokeAt(viewer, { x: 30, y: 30, w: 4, h: 4 });
  assert.deepEqual(bytesOf(renderer, s.overlay), overlay, 'the overlay as it was');
  assert.deepEqual(ui.changes, ['overlay', 'overlay', 'base', 'base']);
  assert.equal(s.version, saved + 2);
  const base = (await s.readAll()).data;
  assert.deepEqual(rgbAt(base, 12, 12), [0x20, 0x13, 0x14], 'the marker painted the base under the overlay');
  assert.notDeepEqual(rgbAt(base, 31, 31), rgbAt(picture, 31, 31), 'and the clone stamp');
  shown = bytesOf(renderer, s.shown);
  assert.deepEqual(rgbAt(shown, 12, 12), MAGENTA, 'covered still');
  assert.deepEqual(rgbAt(shown, 16, 16), [0x20, 0x13, 0x14]);
});

test('undo and redo across both layers give each layer its exact bytes back, in the order painted', () => {
  const { viewer, renderer } = standInViewer(TEX);
  const ui = standInUi();
  const s = createPainter(viewer, ui).session();
  const state = () => ({ base: bytesOf(renderer, s.paint), overlay: s.overlay && bytesOf(renderer, s.overlay) });
  const start = state();
  const states = [];
  for (const [tool, rect] of [['overlay', { x: 8, y: 8, w: 6, h: 6 }], ['marker', { x: 10, y: 10, w: 8, h: 8 }],
    ['overlay', { x: 12, y: 4, w: 10, h: 4 }]]) {
    ui.use = tool;
    strokeAt(viewer, rect);
    states.push(state());
  }
  const version = s.version;
  assert.ok(s.undo());
  assert.deepEqual(state(), states[1]);
  assert.equal(s.version, version, 'the overlay taken back: nothing to save');
  assert.ok(s.undo());
  assert.deepEqual(state(), states[0]);
  assert.equal(s.version, version + 1);
  assert.ok(s.undo());
  assert.deepEqual(state(), { base: start.base, overlay: new Uint8Array(TEX * TEX * 4) });
  assert.deepEqual(bytesOf(renderer, s.shown), start.base, 'the composite follows: the base alone');
  assert.equal(s.undo(), false);
  for (const expected of states) {
    assert.ok(s.redo());
    assert.deepEqual(state(), expected);
  }
  assert.ok(s.painted('overlay') && s.painted('base'));
});

test('the overlay wiped: nothing of it is left, the base is as it was, and Ctrl+Z brings every stroke of it back', async () => {
  const { viewer, renderer } = standInViewer(TEX);
  const ui = standInUi();
  const s = createPainter(viewer, ui).session();
  assert.equal(s.wipe('overlay'), false, 'nothing painted: nothing to wipe');
  ui.use = 'marker';
  strokeAt(viewer, { x: 30, y: 30, w: 6, h: 6 });
  ui.use = 'overlay';
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  strokeAt(viewer, { x: 40, y: 12, w: 10, h: 4 });
  const base = bytesOf(renderer, s.paint);
  const masked = bytesOf(renderer, s.overlay);
  const version = s.version;
  assert.ok(s.wipe('overlay'));
  assert.deepEqual(bytesOf(renderer, s.overlay), new Uint8Array(TEX * TEX * 4), 'no texel of the mask is left');
  assert.deepEqual(bytesOf(renderer, s.paint), base, 'the hand paint stays');
  assert.deepEqual(bytesOf(renderer, s.shown), base, 'the model shows the base alone');
  assert.equal(s.version, version, 'nothing to save');
  assert.equal(s.painted('overlay'), false);
  assert.equal(s.painted('base'), true);
  assert.equal(s.wipe('overlay'), false, 'wiped already');
  assert.ok(s.undo());
  assert.deepEqual(bytesOf(renderer, s.overlay), masked, 'both strokes are back at once');
  assert.equal(s.painted('overlay'), true);
  assert.ok(s.redo());
  assert.equal(s.painted('overlay'), false);
  strokeAt(viewer, { x: 20, y: 20, w: 4, h: 4 });
  assert.equal(s.painted('overlay'), true, 'masked anew after the wipe');
  assert.equal((await s.readOverlay()).data[8 * TEX + 8], 0, 'and only what was painted since');
});

test('the history forgets its oldest strokes, the model does not: a mask painted long ago is still there to render and to wipe', async () => {
  const { viewer, renderer } = standInViewer(TEX);
  const ui = standInUi();
  const s = createPainter(viewer, ui).session();
  ui.use = 'overlay';
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  ui.use = 'marker';
  for (let i = 0; i < 70; i++) strokeAt(viewer, { x: 30 + (i % 8), y: 30, w: 2, h: 2 });
  assert.equal(s.history.all.steps.some((step) => step.layer === 'overlay'), false, 'its stroke left the history');
  assert.equal((await s.readOverlay()).data[8 * TEX + 8], 255, 'the mask itself is on the model');
  assert.equal(s.painted('overlay'), true);
  assert.ok(s.wipe('overlay'));
  assert.deepEqual([(await s.readOverlay()).data[8 * TEX + 8], s.painted('overlay'), s.painted('base')], [0, false, true]);
  assert.ok(s.undo());
  assert.equal(s.painted('overlay'), true, 'the wipe taken back: masked again');
  assert.ok(s.redo());
  // A kept mask (the model leaves the viewer and comes back) keeps the count with it.
  ui.use = 'overlay';
  strokeAt(viewer, { x: 12, y: 50, w: 4, h: 4 });
  const kept = s.keep('overlay');
  assert.equal(kept.strokes.overlay, 1);
  const again = standInViewer(TEX);
  const other = createPainter(again.viewer, standInUi()).session();
  assert.ok(other.put(kept));
  assert.equal(other.painted('overlay'), true);
  assert.equal(renderer.lost, false);
});

test('the overlay is kept alone and put back on the model loaded again, stroke by stroke; the base comes from its files', async () => {
  const one = standInViewer(TEX);
  const ui = standInUi();
  const s = createPainter(one.viewer, ui).session();
  ui.use = 'overlay';
  strokeAt(one.viewer, { x: 8, y: 8, w: 6, h: 6 });
  const first = (await s.readOverlay()).data;
  ui.use = 'marker';
  strokeAt(one.viewer, { x: 30, y: 30, w: 6, h: 6 });
  ui.use = 'overlay';
  strokeAt(one.viewer, { x: 40, y: 8, w: 4, h: 20 });
  const mask = (await s.readOverlay()).data;
  assert.ok(s.keep().tiles.some((tile) => tile.layer === 'base'), 'kept whole, both layers go');
  const kept = s.keep('overlay');
  assert.ok(kept.tiles.length && kept.tiles.every((tile) => tile.layer === 'overlay'));
  assert.deepEqual(kept.steps.map((step) => step.layer), ['overlay', 'overlay'], 'the marker\'s stroke stays behind');

  const two = standInViewer(TEX);
  const again = createPainter(two.viewer, standInUi()).session();
  assert.ok(again.put(kept));
  assert.deepEqual((await again.readOverlay()).data, mask);
  assert.deepEqual((await again.readAll()).data, two.picture, 'the base as its files hold it');
  assert.equal(again.version, again.saved, 'nothing to save');
  assert.equal(two.own.map, again.shown.texture);
  assert.ok(again.undo());
  assert.deepEqual((await again.readOverlay()).data, first);
  assert.ok(again.undo());
  assert.deepEqual((await again.readOverlay()).data, new Uint8Array(TEX * TEX));
  assert.equal(again.undo(), false);
  assert.equal(again.put(kept), false, 'only onto a model without strokes');
});

test('a reset of the graphics card takes the overlay: the model shows the base again, and nothing of it is read', async () => {
  const { viewer, renderer, picture, own } = standInViewer(TEX);
  const ui = standInUi();
  const s = createPainter(viewer, ui).session();
  ui.use = 'overlay';
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  renderer.loseContext();
  await assert.rejects(s.readOverlay(), (e) => e.lost === true);
  assert.equal(s.keep('overlay'), null);
  const fills = s.fills;
  renderer.restoreContext();
  assert.equal(s.fills, fills + 1, 'filled once, though the painter and the session both heard of it');
  assert.deepEqual(ui.told, ['lost', 'restored']);
  assert.equal(s.overlay, null);
  assert.equal(own.map, s.paint.texture, 'the base colour texture, filled again: never the emptied composite');
  assert.deepEqual((await s.readOverlay()).data, new Uint8Array(TEX * TEX));
  assert.deepEqual((await s.readAll()).data, picture);
  assert.ok(!s.history.canUndo);
  strokeAt(viewer, { x: 20, y: 20, w: 2, h: 2 });
  assert.deepEqual((await s.readOverlay()).data, coverage({ x: 20, y: 20, w: 2, h: 2 }), 'a new overlay');
});

// ---- the patch an Omni-Texture save sends: the texels the hand changed alone ------------------------------------------

const MARKER = [0x20, 0x13, 0x14];

/** A patch's texels: [x, y] of those it holds (alpha 255), each with the colour it carries -> {"x,y": [r, g, b]}. */
function texelsOf(patch) {
  const out = {};
  for (let t = 0; t < patch.length / 4; t++) if (patch[t * 4 + 3]) out[`${t % TEX},${Math.floor(t / TEX)}`] = [...patch.subarray(t * 4, t * 4 + 3)];
  return out;
}

/** What `rects` cover, each texel with the colour `colourOf(x, y)` -> as texelsOf gives it. */
function texelsIn(rects, colourOf) {
  const out = {};
  for (const { x, y, w, h } of rects) for (let row = y; row < y + h; row++) for (let col = x; col < x + w; col++) out[`${col},${row}`] = colourOf(col, row);
  return out;
}

/** A painter whose sessions are saved as patches, on a stand-in model of `TEX` texels. */
function patchPainter() {
  const stand = standInViewer(TEX);
  const ui = { ...standInUi(), patches: true };
  return { ...stand, ui, painter: createPainter(stand.viewer, ui) };
}

test('a patch holds the texels whose colour differs, with their colour now and alpha 255; none differing: none', () => {
  const was = Uint8Array.from([1, 2, 3, 200, 4, 5, 6, 255, 7, 8, 9, 0]);
  const now = Uint8Array.from([1, 2, 3, 200, 4, 9, 6, 255, 7, 8, 9, 0]);
  assert.deepEqual([...patchOf(now, was)], [0, 0, 0, 0, 4, 9, 6, 255, 0, 0, 0, 0]);
  assert.equal(patchOf(was, was.slice()), null);
  assert.equal(patchOf(Uint8Array.from([1, 2, 3, 9]), Uint8Array.from([1, 2, 3, 200])), null, 'alpha is never painted');
});

test('a session for patches reads the files\' bytes back once it is filled, and its patch is what the hand painted; a Projection session reads nothing more', async () => {
  const projection = standInViewer(TEX);
  const plain = createPainter(projection.viewer, standInUi()).session();
  assert.equal(projection.renderer.reads, 0, 'no read of what the files hold');
  strokeAt(projection.viewer, { x: 8, y: 8, w: 4, h: 4 });
  assert.equal((await plain.readAll()).patch, undefined);
  assert.equal(plain.files, null);

  const { viewer, renderer, picture, painter } = patchPainter();
  const s = painter.session();
  assert.equal(renderer.reads, 1, 'the texture read back once, right after its fill');
  const a = { x: 8, y: 8, w: 4, h: 4 };
  strokeAt(viewer, a);
  const image = await s.readAll();
  assert.deepEqual(texelsOf(image.patch), texelsIn([a], () => MARKER));
  assert.deepEqual(s.stored.data, picture, 'the bytes read back stand in for the picture');
});

test('a stroke taken back before its save leaves an empty patch, never sent; taken back after a save, its texels go back', async () => {
  const sent = [];
  globalThis.fetch = async (path, init) => {
    sent.push([path, init.body]);
    return { status: 200, text: async () => '' };
  };
  const { viewer, picture, painter } = patchPainter();
  const s = painter.session();
  const save = createSaver({ project: 'Demo', propId: 'p_1', glb: 'omnitexture/ot_001/omnitexture.glb', session: s, save: saveHandPaint });
  const a = { x: 8, y: 8, w: 4, h: 4 };
  strokeAt(viewer, a);
  assert.ok(s.undo());
  save.schedule();
  await save.flush();
  assert.equal(sent.length, 0, 'nothing to send');
  assert.equal(s.version, s.saved, 'and nothing left to save');

  assert.ok(s.redo());
  save.schedule();
  await save.flush();
  assert.equal(sent.length, 1);
  assert.match(sent[0][0], /\/props\/p_1\/omnitexture\/paint\?glb=omnitexture%2Fot_001%2Fomnitexture\.glb&width=64&height=64$/);
  assert.deepEqual(texelsOf(sent[0][1]), texelsIn([a], () => MARKER));
  assert.ok(s.undo());
  save.schedule();
  await save.flush();
  assert.deepEqual(texelsOf(sent[1][1]), texelsIn([a], (x, y) => rgbAt(picture, x, y)), 'the colours the files had before it');
});

test('a save while a stroke is under way sends what was painted until then, and the rest goes with the next', async () => {
  const { viewer, renderer, painter } = patchPainter();
  const s = painter.session();
  await s.files;
  const first = { x: 8, y: 8, w: 4, h: 4 };
  const then = { x: 30, y: 30, w: 4, h: 4 };
  pressAt(viewer, first);
  moveTo(viewer, 60, 50, first);
  let read;
  renderer.slowRead = new Promise((resolve) => { read = resolve; });
  const reading = s.readAll();                     // the save, asked for now
  moveTo(viewer, 70, 50, then);
  letGo(viewer);
  read();
  const image = await reading;
  assert.deepEqual(texelsOf(image.patch), texelsIn([first], () => MARKER), 'as the texture was when the save asked');
  s.savedAs(image);
  assert.notEqual(s.version, s.saved, 'the stroke ended after it: still to save');
  renderer.slowRead = null;
  assert.deepEqual(texelsOf((await s.readAll()).patch), texelsIn([then], () => MARKER));
});

test('no patch is read across a lost context, and the files\' bytes are read again from their picture once it is back', async () => {
  const { viewer, renderer, painter } = patchPainter();
  let read;
  renderer.slowRead = new Promise((resolve) => { read = resolve; });
  const s = painter.session();                     // the files' bytes on their way back ...
  renderer.loseContext();                          // ... when the card is reset
  await assert.rejects(s.readAll(), (e) => e.lost === true);
  renderer.restoreContext();
  assert.equal(renderer.reads, 2, 'filled from the picture again, and read again');
  read();
  renderer.slowRead = null;
  const a = { x: 8, y: 8, w: 4, h: 4 };
  strokeAt(viewer, a);
  assert.deepEqual(texelsOf((await s.readAll()).patch), texelsIn([a], () => MARKER),
    'against the files as they are: never the zeros of the read the loss spoilt');
});

test('a flush resolves once every change made before it is in the files, also with a save on its way', async () => {
  let reads = 0;
  let letRead;
  const session = {
    version: 2, saved: 1,
    hold: () => () => {},
    async readAll() {
      reads++;
      const version = this.version;
      if (reads === 1) await new Promise((resolve) => { letRead = resolve; });
      return { width: 1, height: 1, data: new Uint8Array(4), version, fills: 1 };
    },
    savedAs(image) { this.saved = image.version; },
  };
  const saved = [];
  const saver = createSaver({ project: 'Demo', propId: 'p_1', glb: 'g', session, save: async (b, image) => { saved.push(image.version); return null; } });
  saver.schedule();
  const first = saver.flush();                      // reading version 2 ...
  await new Promise((resolve) => setTimeout(resolve, 0));
  session.version = 3;                              // ... when another stroke ends
  let done = false;
  const second = saver.flush().then(() => { done = true; });
  letRead();
  await first;
  await new Promise((resolve) => setTimeout(resolve, 0));
  await second;
  assert.ok(done);
  assert.deepEqual(saved, [2, 3], 'the second flush waited for the change it was asked after');
  assert.equal(session.saved, 3);
  assert.equal(saver.busy, false);
});

test('the bar\'s flush waits for the save of every model it painted, not only the one on screen', async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const saved = [];
  const bar = createPaintBar({ save: async ({ glb }) => { await gate; saved.push(glb); return null; } });
  const { viewer } = standInViewer(TEX);
  await bar.bind({ viewer, project: 'Demo', propId: 'p_1', glb: 'step_1.glb', select() {} });
  bar.pick('marker');
  strokeAt(viewer, { x: 8, y: 8, w: 4, h: 4 });
  bar.unbind();                                     // its save goes on after it (step_1.glb)
  viewer.current = standInModel(TEX).pivot;
  await bar.bind({ viewer, project: 'Demo', propId: 'p_1', glb: 'step_2.glb', select() {} });
  let done = false;
  const flushing = bar.flush().then(() => { done = true; });
  assert.equal(bar.pending(), true);
  await new Promise((resolve) => setTimeout(resolve, 5));
  assert.equal(done, false, 'the first model\'s save is still on its way');
  release();
  await flushing;
  assert.deepEqual(saved, ['step_1.glb']);
  assert.equal(bar.pending(), false);
  bar.pick(null);
  bar.destroy();
});

test('a tool asked for on a model that is not the stage\'s to paint yet comes up once the stage made it so, unless Escape was pressed meanwhile', async () => {
  const { viewer } = standInViewer(TEX);
  const bar = createPaintBar({ save: async () => null });
  const tools = bar.el.children[2].children[0];
  const marker = tools.children[2];
  let made;
  const begin = () => new Promise((resolve) => { made = resolve; });
  const model = { viewer, project: 'Demo', propId: 'p_1', select() {} };
  for (const escape of [false, true]) {
    await bar.bind({ ...model, glb: 'mesh/mesh.glb', begin });
    marker.click();
    assert.equal(marker.classList.contains('on'), false, 'asked for, not up: nothing paints the 3D model itself');
    if (escape) keyDown('Escape');
    await bar.bind({ ...model, glb: 'omnitexture/ot_001/omnitexture.glb' });   // (what the stage does once begun)
    made();
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(marker.classList.contains('on'), !escape, escape ? 'Escape: not wanted any more' : 'up now');
    bar.pick(null);
    bar.unbind();
  }
  bar.destroy();
});

test('a brush taken up while another map shows goes to the lit view and says why: the brushes paint the base colour alone', async () => {
  const { viewer } = standInViewer(TEX);
  const bar = createPaintBar({ save: async () => null });
  const marker = bar.el.children[2].children[0].children[2];
  const toasts = document.createElement('div');
  document.getElementById = () => toasts;
  const { setTimeout: realTimeout } = globalThis;
  globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 1000 ? 0 : realTimeout(fn, ms, ...rest));   // a toast goes by itself
  const said = () => toasts.children.map((toast) => toast.textContent);
  const went = [];
  try {
    await bar.bind({
      viewer, project: 'Demo', propId: 'p_1', glb: 'omnitexture/ot_001/omnitexture.glb',
      select: (view) => { went.push(view); viewer.view = view; },
    });
    for (const view of ['all', 'basecolor']) {
      viewer.view = view;
      marker.click();
      bar.pick(null);
    }
    assert.deepEqual([went, said()], [[], []], 'the paint shows there: the view stays');
    viewer.view = 'roughness';
    marker.click();
    assert.deepEqual(went, ['all']);
    assert.equal(said().length, 1);
    assert.match(said()[0], /^The brushes paint the base colour only, so the view went to “All” to show it\. Roughness and Metallic can’t be painted yet\./);
    assert.equal(marker.classList.contains('on'), true);
    bar.pick(null);
    viewer.view = 'metallic';
    bar.pick(null);
    assert.equal(said().length, 1, 'putting the brush down says nothing');
  } finally {
    delete document.getElementById;
    globalThis.setTimeout = realTimeout;
    viewer.view = 'all';
    bar.unbind();
    bar.destroy();
  }
});

// ---- the tool bar (stages/paintbar.js) on a fake page ---------------------------------------------------------------

test('the tool bar: the hint left of its glass, the tools in a column over its right end - Orbit on top, a stage\'s own tool at its foot with the x that wipes it', async () => {
  const label = (node) => node.getAttribute('aria-label');
  const plain = createPaintBar();
  assert.deepEqual(plain.el.children[2].children[0].children.map(label), ['Orbit', 'Clone stamp', 'Marker of any colour'],
    'top to bottom: Orbit first');
  assert.equal(plain.el.children[2].children[1].children[1].children[0].className, 'pt-status');
  plain.destroy();

  const mine = document.createElement('button');
  mine.setAttribute('aria-label', 'Mask marker');
  const slot = document.createElement('button');
  const calls = [];
  const bar = createPaintBar({
    overlay: {
      button: mine, colour: '#ff00ff', hint: 'Mask: paint over what to make again', slot, wipe: 'Discard the mask',
      up: () => calls.push('up'), down: () => calls.push('down'),
    },
    changed: (layer) => calls.push(layer),
    undo: (which) => {
      calls.push(which);
      return true;
    },
    save: async () => { calls.push('saved'); return null; },
  });
  assert.equal(bar.el.className, 'pt-layer');
  assert.equal(bar.el.hidden, true, 'nothing until a model is bound');
  const [ring, mark, dock] = bar.el.children;
  assert.deepEqual([ring.className, mark.className, dock.className], ['pt-ring', 'pt-source', 'pt-dock']);
  const [tools, row] = dock.children;
  assert.equal(tools.className, 'pt-tools');
  assert.deepEqual(tools.children.slice(0, 3).map(label), ['Orbit', 'Clone stamp', 'Marker of any colour']);
  const own = tools.children[3];
  assert.deepEqual([own.className, own.children.map(label)], ['pt-own', ['Discard the mask', 'Mask marker']],
    'the stage\'s own at the column\'s foot, by the glass, the x that wipes it to its left');
  const wipe = own.children[0];
  assert.equal(wipe.hidden, true, 'no x until the tool is up');
  const [hint, glass] = row.children;
  assert.deepEqual([row.className, hint.className, glass.className], ['pt-row', 'pt-hint', 'pt-bar'], 'the hint left of the glass');
  assert.equal(hint.children.length, 2, 'in two lines');
  assert.equal(glass.children[0], slot, 'the stage\'s slot at the glass\'s left end');
  assert.deepEqual(glass.children.slice(1).map((node) => node.className), ['pt-status', 'range pt-size', 'pt-flag', 'pt-flag']);
  assert.deepEqual(glass.children.slice(3).map((node) => node.textContent), ['P', 'F']);

  const { viewer } = standInViewer(TEX);
  await bar.bind({ viewer, project: 'Demo', propId: 'p_1', glb: 'omnitexture/x.glb', select() {} });
  assert.equal(bar.el.hidden, false);
  const lines = () => hint.children.map((line) => line.textContent);
  assert.deepEqual(lines(), ['Orbit: middle or left drag · Pan: right drag', 'Zoom: wheel · F: frame']);
  for (const [tool, first] of [[tools.children[1], 'Clone stamp: Ctrl+click'], [tools.children[2], 'Marker: drag']]) {
    tool.click();
    assert.ok(lines()[0].startsWith(first) && lines()[1].startsWith('Shift + move sideways: brush size'), lines()[0]);
    assert.equal(wipe.hidden, true);
  }
  mine.click();
  assert.deepEqual(calls, ['up']);
  assert.ok(mine.classList.contains('on') && !tools.children[2].classList.contains('on'), 'its ring, the marker\'s gone');
  assert.deepEqual([wipe.hidden, wipe.disabled], [false, true], 'the x is there, with nothing to wipe yet');
  assert.equal(lines()[0], 'Mask: paint over what to make again');
  assert.ok(lines()[1].startsWith('Shift + move sideways: brush size'), 'the keys, as for every tool');
  assert.equal(viewer.holdLight, true);

  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  assert.deepEqual(calls, ['up', 'overlay']);
  assert.equal(bar.pending(), false, 'nothing of an overlay stroke is saved');
  assert.equal((await bar.session().readOverlay()).data[8 * TEX + 8], 255);
  // The x wipes the mask as one change of the overlay; Ctrl+Z brings it back.
  assert.equal(wipe.disabled, false);
  wipe.click();
  assert.deepEqual(calls.splice(2), ['overlay']);
  assert.equal((await bar.session().readOverlay()).data[8 * TEX + 8], 0);
  assert.deepEqual([bar.session().painted('overlay'), wipe.disabled], [false, true]);
  wipe.click();
  assert.deepEqual(calls, ['up', 'overlay'], 'nothing left to wipe: no change');
  keyDown('z', { ctrlKey: true });
  assert.deepEqual(calls.splice(2), ['overlay']);
  assert.equal((await bar.session().readOverlay()).data[8 * TEX + 8], 255);
  assert.equal(wipe.disabled, false);
  keyDown('Escape');
  assert.deepEqual(calls, ['up', 'overlay', 'down']);
  assert.equal(wipe.hidden, true, 'the tool put down: the x goes');

  tools.children[2].click();                        // the marker
  strokeAt(viewer, { x: 30, y: 30, w: 4, h: 4 });
  assert.deepEqual(calls.slice(3), ['base']);
  assert.equal(bar.pending(), true, 'the marker\'s stroke is on its way to the files');
  await bar.flush();
  assert.deepEqual(calls.slice(4), ['saved']);
  assert.equal(bar.painted(), true);

  keyDown('z', { ctrlKey: true });                 // the marker's stroke back, then the overlay's
  keyDown('z', { ctrlKey: true });
  assert.deepEqual(calls.slice(5), ['base', 'overlay']);
  assert.equal(bar.painted(), false);
  keyDown('z', { ctrlKey: true });                 // no stroke left: the stage's own change
  keyDown('y', { ctrlKey: true });                 // the strokes come back first
  assert.deepEqual(calls.slice(7), ['undo', 'overlay']);
  await bar.flush();
  bar.destroy();
  assert.equal(bar.el.hidden, true);
});
