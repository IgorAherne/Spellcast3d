// The strokes painted on the Images workbench's picture (strokes.js, used by marks.js), without a page: how a pen's
// pressure thins the line, which points a stroke keeps, the parts a stroke is painted in (the same while it is drawn
// and when it is painted again), undo / redo / wipe, the keys; and the markers' own rules (marks.js): the size a
// Shift + sideways move gives and the ring cursor.
//   node --test code/tests/web/strokes.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

const STATIC = new URL('../../spellcast/static/js/', import.meta.url);
const load = (path) => import(new URL(path, STATIC));
const {
  addPoint, brushWidth, createHistory, paintStroke, strokePart, strokeParts, strokeTail, undoKey,
} = await load('stages/strokes.js');

/** A canvas's 2D context that writes down what it is asked to paint. */
function fakeCtx() {
  const calls = [];
  const note = (name) => (...args) => calls.push([name, ...args]);
  return {
    calls, beginPath: note('beginPath'), moveTo: note('moveTo'), quadraticCurveTo: note('quadraticCurveTo'),
    stroke() { calls.push(['stroke', this.strokeStyle, this.lineWidth, this.lineCap]); },
    arc: note('arc'), fill() { calls.push(['fill', this.fillStyle]); },
  };
}

const pt = (x, y, w = 10) => ({ x, y, w });
const stroke = (...points) => ({ kind: 'stroke', colour: '#16181d', points });

test('a pen thins the line with its pressure; a mouse and a finger paint it full', () => {
  assert.equal(brushWidth(20, 'mouse', 0.5), 20);
  assert.equal(brushWidth(20, 'touch', 0.2), 20);
  assert.equal(brushWidth(20, 'pen', 1), 20);
  assert.equal(brushWidth(20, 'pen', 0.5), 12.5);
  assert.equal(brushWidth(20, 'pen', 0), 20, 'a pen that reports no pressure');
  assert.equal(brushWidth(2, 'pen', 0.01), 1, 'never thinner than a pixel');
  assert.equal(brushWidth(20), 20);
});

test('a hand at rest adds no points to a stroke', () => {
  const points = [];
  assert.equal(addPoint(points, pt(10, 10)), true);
  assert.equal(addPoint(points, pt(10.5, 10.5)), false);
  assert.equal(addPoint(points, pt(12, 10)), true);
  assert.equal(addPoint(points, pt(12, 10)), false);
  assert.deepEqual(points, [pt(10, 10), pt(12, 10)]);
});

test('a stroke is a dot, then curves through the middles of its points, then a straight end', () => {
  assert.deepEqual(strokeParts([]), []);
  assert.deepEqual(strokeParts([pt(5, 5, 8)]), [{ dot: true, to: pt(5, 5, 8), width: 8 }], 'a click: a dot');
  assert.equal(strokeTail([pt(5, 5)]), null);
  const two = [pt(0, 0), pt(10, 0, 6)];
  assert.deepEqual(strokeParts(two).slice(1), [{ from: two[0], ctrl: two[1], to: two[1], width: 6 }], 'two points: a straight line');
  const four = [pt(0, 0), pt(10, 0, 4), pt(10, 10, 6), pt(20, 10, 8)];
  const parts = strokeParts(four);
  assert.equal(parts.length, 4);
  assert.deepEqual(parts[1], { from: four[0], ctrl: four[1], to: { x: 10, y: 5 }, width: 4 });
  assert.deepEqual(parts[2], { from: { x: 10, y: 5 }, ctrl: four[2], to: { x: 15, y: 10 }, width: 6 }, 'each part starts where the one before ended');
  assert.deepEqual(parts[3], { from: { x: 15, y: 10 }, ctrl: four[3], to: four[3], width: 8 });
  // While it is drawn, a part is painted as soon as the point after it is there, and never changes after.
  for (let n = 2; n <= four.length; n++) {
    for (let k = 0; k <= n - 2; k++) assert.deepEqual(strokePart(four.slice(0, n), k), parts[k], `part ${k} with ${n} points`);
  }
});

test('a stroke is painted in its colour, with round ends, each part as wide as the pen pressed', () => {
  const ctx = fakeCtx();
  paintStroke(ctx, stroke(pt(0, 0, 4), pt(10, 0, 6), pt(10, 10, 8)));
  const strokes = ctx.calls.filter((c) => c[0] === 'stroke');
  assert.deepEqual(strokes, [['stroke', '#16181d', 6, 'round'], ['stroke', '#16181d', 8, 'round']]);
  assert.deepEqual(ctx.calls.find((c) => c[0] === 'arc').slice(0, 4), ['arc', 0, 0, 2], 'the dot it starts with');
  assert.deepEqual(ctx.calls.filter((c) => c[0] === 'fill'), [['fill', '#16181d']]);
  assert.deepEqual(ctx.calls.filter((c) => c[0] === 'quadraticCurveTo'), [['quadraticCurveTo', 10, 0, 10, 5], ['quadraticCurveTo', 10, 10, 10, 10]]);
});

test('undo and redo go stroke by stroke; a new stroke after an undo drops what was taken back', () => {
  const history = createHistory();
  const [a, b, c] = [stroke(pt(1, 1)), stroke(pt(2, 2)), stroke(pt(3, 3))];
  assert.equal(history.canUndo || history.canRedo, false);
  assert.equal(history.undo(), false);
  assert.equal(history.redo(), false);
  history.add(a);
  history.add(b);
  assert.deepEqual(history.shown(), { cleared: false, strokes: [a, b] });
  assert.equal(history.undo(), true);
  assert.deepEqual(history.shown().strokes, [a]);
  assert.equal(history.canRedo, true);
  assert.equal(history.redo(), true);
  assert.deepEqual(history.shown().strokes, [a, b]);
  history.undo();
  history.add(c);
  assert.equal(history.canRedo, false);
  assert.deepEqual(history.shown().strokes, [a, c]);
  history.reset();
  assert.equal(history.canUndo || history.canRedo, false);
});

test('a wipe is a step like a stroke: Undo brings the marks back', () => {
  const history = createHistory();
  const [a, b] = [stroke(pt(1, 1)), stroke(pt(2, 2))];
  history.add(a);
  history.add({ kind: 'clear' });
  assert.deepEqual(history.shown(), { cleared: true, strokes: [] });
  history.add(b);
  assert.deepEqual(history.shown(), { cleared: true, strokes: [b] });
  history.undo();
  history.undo();
  assert.deepEqual(history.shown(), { cleared: false, strokes: [a] });
});

test('a history with a limit lets its oldest step go', () => {
  const history = createHistory(2);
  const [a, b, c] = [stroke(pt(1, 1)), stroke(pt(2, 2)), stroke(pt(3, 3))];
  assert.equal(history.add(a), null);
  assert.equal(history.add(b), null);
  assert.equal(history.add(c), a);
  assert.deepEqual(history.shown().strokes, [b, c]);
  history.undo();
  history.undo();
  assert.equal(history.undo(), false, 'no further back than the steps kept');
});

test('Ctrl+Z takes back, Ctrl+Y and Ctrl+Shift+Z bring back (Cmd on a Mac)', () => {
  assert.equal(undoKey({ ctrlKey: true, key: 'z' }), 'undo');
  assert.equal(undoKey({ ctrlKey: true, key: 'Z', shiftKey: true }), 'redo');
  assert.equal(undoKey({ ctrlKey: true, key: 'y' }), 'redo');
  assert.equal(undoKey({ metaKey: true, key: 'z' }), 'undo');
  assert.equal(undoKey({ key: 'z' }), null);
  assert.equal(undoKey({ ctrlKey: true, altKey: true, key: 'z' }), null);
  assert.equal(undoKey({ ctrlKey: true, key: 'c' }), null);
});

test('the markers: Shift + 250 px sideways goes through the whole size range; the cursor is a ring that size', async () => {
  globalThis.document ??= { createElement: () => ({}) };
  const { MAX_SIZE, MIN_SIZE, dragSize, ringCursor } = await load('stages/marks.js');
  assert.deepEqual([MIN_SIZE, MAX_SIZE], [2, 64]);
  assert.equal(dragSize(12, 0), 12);
  assert.equal(dragSize(12, 125), 43, 'half the way: half the range');
  assert.equal(dragSize(12, -1000), MIN_SIZE);
  assert.equal(dragSize(12, 1000), MAX_SIZE);
  const css = ringCursor(30);
  assert.match(css, /^url\("data:image\/svg\+xml,.+"\) 17 17, crosshair$/, 'its middle is the hotspot');
  const svg = decodeURIComponent(css.slice(css.indexOf(',') + 1, css.lastIndexOf('")')));
  assert.match(svg, /width="34" height="34"/);
  assert.equal((svg.match(/r="15"/g) || []).length, 2, 'a white ring on a dark one, 30 px across');
});
