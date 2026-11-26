// The Images picture's zoom and pan (static/js/stages/picview.js): the wheel zooms about the pointer, no further out
// than the whole picture and never so that the picture leaves the viewport's edge; zoomed all the way out, a tileable
// picture is shown from afar; and at the whole picture the wheel pauses before it goes on the way it came.
//   node --test code/tests/web/picview.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

const STATIC = new URL('../../spellcast/static/', import.meta.url);
const { HOLD_MS, MAX_ZOOM, createPictureView, held, wheelFactor, zoomedAbout } = await import(new URL('js/stages/picview.js', STATIC));

const area = { w: 800, h: 600 };
const square = { w: 600, h: 600 };      // a square picture fitted in that area

test('the view keeps the whole picture at the least, and its edge no further in than the area\'s', () => {
  assert.deepEqual(held({ zoom: 0.5, x: 30, y: -40 }, area, square), { zoom: 1, x: 0, y: 0 });
  // Twice as large: 1200 x 1200 in 800 x 600 - 200 px of room to each side, 300 up and down.
  assert.deepEqual(held({ zoom: 2, x: 500, y: -500 }, area, square), { zoom: 2, x: 200, y: -300 });
  // 1.2 times: 720 wide, still inside the 800: held in the middle that way.
  assert.deepEqual(held({ zoom: 1.2, x: 50, y: 20 }, area, square), { zoom: 1.2, x: 0, y: 20 });
  assert.equal(held({ zoom: 99, x: 0, y: 0 }, area, square).zoom, MAX_ZOOM);
});

test('zoomed about a point, that point of the picture stays under it', () => {
  const at = { x: 120, y: -90 };
  const before = { zoom: 2, x: 50, y: 30 };
  const after = zoomedAbout(before, 1.5, at, area, square);
  assert.equal(after.zoom, 3);
  // A point of the picture: where it is on screen against the picture's middle, in the picture's own units.
  const own = (v) => ({ x: (at.x - v.x) / v.zoom, y: (at.y - v.y) / v.zoom });
  assert.ok(Math.abs(own(before).x - own(after).x) < 1e-9 && Math.abs(own(before).y - own(after).y) < 1e-9);
  // Out to the whole picture: back in the middle.
  assert.deepEqual(zoomedAbout(before, 0.1, at, area, square), { zoom: 1, x: 0, y: 0 });
});

test('a wheel\'s notch zooms by about a fifth, the same in pixels, lines or pages', () => {
  assert.ok(Math.abs(wheelFactor({ deltaY: -100, deltaMode: 0 }, 600) - 1.2214) < 1e-3);
  assert.ok(wheelFactor({ deltaY: 100, deltaMode: 0 }, 600) < 1);
  assert.ok(Math.abs(wheelFactor({ deltaY: -3, deltaMode: 1 }, 600) - wheelFactor({ deltaY: -99, deltaMode: 0 }, 600)) < 1e-12);
  assert.equal(wheelFactor({ deltaY: -1, deltaMode: 2 }, 600), wheelFactor({ deltaY: -600, deltaMode: 0 }, 600));
});

/** A viewport 820 x 620 (the area 800 x 600 in it) with a square picture, and the wheel turned over its middle. */
function viewport({ picture = true, tileable = true } = {}) {
  const listeners = {};
  const style = new Map();
  const el = {
    clientLeft: 0, clientTop: 0, clientWidth: 820, clientHeight: 620,
    style: { setProperty: (k, v) => style.set(k, v) },
    classList: { add() {}, remove() {} },
    addEventListener: (type, fn) => { listeners[type] = fn; },
    removeEventListener() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 820, height: 620 }),
    setPointerCapture() {},
  };
  let afarChanges = 0;
  const view = createPictureView(el, {
    measure: () => (picture ? { w: area.w * Number(style.get('--zoom') || 1), h: area.h * Number(style.get('--zoom') || 1), aspect: 1 } : null),
    tiles: () => tileable,
    changed: () => { afarChanges++; },
  });
  const turn = (deltaY) => {
    let prevented = false;
    listeners.wheel({ deltaY, deltaMode: 0, clientX: 410, clientY: 310, preventDefault: () => { prevented = true; } });
    return prevented;
  };
  return {
    view, turn, listeners, changes: () => afarChanges,
    zoom: () => Number(style.get('--zoom') || 1),
    pan: () => [parseFloat(style.get('--pan-x') || '0'), parseFloat(style.get('--pan-y') || '0')],
  };
}

test('the wheel zooms in and back out to the whole picture; only then, and only for a tileable picture, does it go on to show it from afar', () => {
  const now = { t: 1000 };
  const realNow = performance.now;
  performance.now = () => now.t;
  try {
    const v = viewport();
    v.turn(-100);
    v.turn(-100);
    assert.ok(v.zoom() > 1.4 && !v.view.afar);
    // Out: down to the whole picture first, where the wheel pauses - turned on at once, it does not go on to afar.
    for (let i = 0; i < 10 && v.zoom() > 1; i++) v.turn(100);
    assert.equal(v.zoom(), 1);
    now.t += HOLD_MS - 50;
    v.turn(100);
    assert.equal(v.view.afar, false, 'held: the turn that zoomed out does not tile the picture as well');
    now.t += 100;
    v.turn(100);
    assert.equal(v.view.afar, true, 'a turn out at the whole picture, after the pause');
    assert.equal(v.changes(), 1);
    v.turn(100);
    assert.equal(v.view.afar, true);
    // Back: the picture alone again, and no zooming in for HOLD_MS.
    v.turn(-100);
    assert.equal(v.view.afar, false);
    now.t += HOLD_MS - 50;
    v.turn(-100);
    assert.equal(v.zoom(), 1, 'held: the turn that came back does not zoom in as well');
    now.t += 100;
    v.turn(-100);
    assert.ok(v.zoom() > 1);
    // Each pause holds its own way only: back at the whole picture from afar, a turn out tiles at once; and zoomed
    // out to it, a turn in zooms at once.
    const quick = viewport();
    quick.turn(100);
    quick.turn(-100);
    quick.turn(100);
    assert.equal(quick.view.afar, true);
    quick.turn(-100);
    now.t += HOLD_MS + 1;
    quick.turn(-100);
    quick.turn(100);
    assert.equal(quick.zoom(), 1);
    quick.turn(-100);
    assert.ok(quick.zoom() > 1 && !quick.view.afar);
    // Not tileable: the whole picture is as far out as it goes.
    const plain = viewport({ tileable: false });
    plain.turn(100);
    assert.equal(plain.view.afar, false);
    assert.equal(plain.zoom(), 1);
    // No picture: the wheel is not the view's.
    const none = viewport({ picture: false });
    assert.equal(none.turn(-100), false);
    assert.equal(none.zoom(), 1);
  } finally {
    performance.now = realNow;
  }
});

test('the middle button pans the picture within reach; other buttons are not the view\'s', () => {
  const v = viewport();
  v.turn(-300);                                      // about 1.8 times: room to pan
  let stopped = false;
  const press = (button) => v.listeners.pointerdown({
    button, pointerType: 'mouse', pointerId: 1, clientX: 400, clientY: 300,
    preventDefault() {}, stopPropagation() { stopped = true; },
  });
  press(0);
  assert.equal(stopped, false, 'the left button paints and slices as before');
  press(1);
  assert.equal(stopped, true);
  const zoom = v.zoom();
  assert.ok(zoom > 1.7);
  // Dragged 60 px up: the picture follows. Dragged far right: its left edge stops at the area's.
  v.listeners.pointermove({ pointerId: 1, clientX: 400, clientY: 240 });
  assert.deepEqual(v.pan(), [0, -60]);
  v.listeners.pointermove({ pointerId: 1, clientX: 5000, clientY: 240 });
  assert.ok(Math.abs(v.pan()[0] - (600 * zoom - 800) / 2) < 1e-6);
  v.listeners.pointerup({ pointerId: 1 });
  v.listeners.pointermove({ pointerId: 1, clientX: 0, clientY: 0 });   // let go: no more panning
  assert.ok(Math.abs(v.pan()[0] - (600 * zoom - 800) / 2) < 1e-6);
});
