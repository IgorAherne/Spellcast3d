// Tests of the slicer's rules (static/js/stages/slicer.js): where a pointer is on the picture, the slice a
// drag makes, where a moved slice may go, the presets, which view comes next, and changing, removing and
// undoing views; and of where its row of buttons is on the page (the workbench's header).
//   node --test code/tests/web/slicer.test.mjs
// (code/tests/test_web.py runs every code/tests/web/*.test.mjs when Node.js is installed)
import assert from 'node:assert/strict';
import { test } from 'node:test';

// What a browser page has, as far as these modules touch it when they load (actions.js listens for the
// page being hidden or closed) and as far as the slicer builds one: elements with children, classes and text.
class FakeNode {}
class FakeEl extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.className = '';
    this.style = { setProperty() {} };
    this.listeners = {};
    this.hidden = false;
  }

  setAttribute() {}
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  animate() {}
  click() { for (const fn of this.listeners.click || []) fn({}); }
  get textContent() { return this.children.map((c) => (typeof c === 'string' ? c : c.textContent)).join(''); }
  get classList() {
    const list = () => this.className.split(/\s+/).filter(Boolean);
    const toggle = (c, on = !list().includes(c)) => { this.className = [...list().filter((x) => x !== c), ...(on ? [c] : [])].join(' '); };
    return { contains: (c) => list().includes(c), toggle, add: (c) => toggle(c, true), remove: (c) => toggle(c, false) };
  }

  /** Every element in it (itself first) with the class `name`. */
  byClass(name) {
    const inside = this.children.flatMap((c) => (c instanceof FakeEl ? c.byClass(name) : []));
    return this.classList.contains(name) ? [this, ...inside] : inside;
  }
}
globalThis.Node = FakeNode;
globalThis.addEventListener ??= () => {};
globalThis.document = {
  addEventListener: () => {}, removeEventListener: () => {}, hidden: false,
  // An icon is made in a <template>: its content is the <svg>.
  createElement: (tag) => (tag === 'template' ? { content: { firstChild: new FakeEl('svg') } } : new FakeEl(tag)),
  createElementNS: (ns, tag) => new FakeEl(tag),
};

const {
  MIN_SIDE, PRESET_EDGE, PRESET_GAP, SHEET_ORDER, VIEWS, createSlicer, dragRect, dropPicture, dropView, firstFree, fitRect, history, importSides, moveRect,
  needsFront, nextView, presetRects, presetViews, putView, relabel, sameViews, sortViews, toImage, usable,
} = await import(new URL('../../spellcast/static/js/stages/slicer.js', import.meta.url));
const { VIEW_ORDER } = await import(new URL('../../spellcast/static/js/stages/rules.js', import.meta.url));

const SHEET = { w: 1024, h: 512 };
const v = (view, src = 'images/0001.png', rect = null) => ({ view, src, rect });

test('the four views, in Pixal3D\'s orbit order, with the names the page shows', () => {
  assert.deepEqual(VIEW_ORDER, ['front', 'left', 'back', 'right']);
  // The arrow points where the object looks in the picture: before "Left", after "Right". Never "Side".
  assert.deepEqual(VIEW_ORDER.map((k) => VIEWS[k].label), ['Front', '← Left', 'Back', 'Right →']);
  assert.equal(VIEWS.left.title, 'Left view: the object faces left');
  assert.equal(VIEWS.right.title, 'Right view: the object faces right');
  assert.equal(VIEW_ORDER.some((k) => /side/i.test(`${VIEWS[k].label} ${VIEWS[k].short} ${VIEWS[k].title}`)), false);
  assert.equal(new Set(VIEW_ORDER.map((k) => VIEWS[k].color)).size, 4, 'a colour of its own each');
});

test('a point on screen is a pixel of the picture, fitted into its box like object-fit: contain', () => {
  // A 1024 × 512 sheet in a 600 × 400 box at (100, 50): scaled to 600 × 300, 50 px bars above and below.
  const box = { left: 100, top: 50, width: 600, height: 400 };
  assert.deepEqual(fitRect(box, SHEET), { left: 100, top: 100, width: 600, height: 300, scale: 600 / 1024 });
  assert.deepEqual(toImage(100, 100, box, SHEET), { x: 0, y: 0 });
  assert.deepEqual(toImage(400, 250, box, SHEET), { x: 512, y: 256 });
  assert.deepEqual(toImage(700, 400, box, SHEET), { x: 1024, y: 512 });
  assert.deepEqual(toImage(40, 60, box, SHEET), { x: 0, y: 0 }, 'outside the picture: its nearest edge');
  assert.deepEqual(toImage(900, 900, box, SHEET), { x: 1024, y: 512 });
  // A tall picture in a wide box: bars left and right.
  const tall = { w: 832, h: 1216 };
  const f = fitRect({ left: 0, top: 0, width: 1000, height: 608 }, tall);
  assert.equal(f.height, 608);
  assert.equal(f.left, (1000 - 416) / 2);
  assert.deepEqual(toImage(f.left + 208, 304, { left: 0, top: 0, width: 1000, height: 608 }, tall), { x: 416, y: 608 });
});

test('a drag makes a slice in whole pixels inside the picture, whichever way it goes', () => {
  assert.deepEqual(dragRect({ x: 10.4, y: 20.6 }, { x: 300.5, y: 200 }, SHEET), [10, 21, 291, 179]);
  assert.deepEqual(dragRect({ x: 300, y: 200 }, { x: 10, y: 20 }, SHEET), [10, 20, 290, 180], 'up and to the left');
  assert.deepEqual(dragRect({ x: 1000, y: 500 }, { x: 1024, y: 512 }, SHEET), [1000, 500, 24, 12]);
  assert.equal(usable([1000, 500, 24, 12]), false, 'smaller than the hub takes');
  assert.equal(usable([0, 0, MIN_SIDE, MIN_SIDE]), true);
  assert.equal(usable(null), false);
});

test('with Shift a drag makes a square, as big as the longer side, never past the picture', () => {
  assert.deepEqual(dragRect({ x: 100, y: 100 }, { x: 300, y: 150 }, SHEET, true), [100, 100, 200, 200]);
  assert.deepEqual(dragRect({ x: 300, y: 300 }, { x: 250, y: 100 }, SHEET, true), [100, 100, 200, 200], 'up and to the left');
  assert.deepEqual(dragRect({ x: 100, y: 400 }, { x: 500, y: 450 }, SHEET, true), [100, 400, 112, 112],
    'as far as the bottom edge lets it grow');
  const [x, y, w, h] = dragRect({ x: 10.7, y: 3.2 }, { x: 250.2, y: 90 }, SHEET, true);
  assert.equal(w, h);
  assert.ok(Number.isInteger(x) && Number.isInteger(y) && x + w <= SHEET.w && y + h <= SHEET.h);
});

test('a slice moved by its button keeps its size, in whole pixels, and stops at the picture\'s edges', () => {
  const rect = [100, 50, 200, 120];
  assert.deepEqual(moveRect(rect, 30.4, -20.6, SHEET), [130, 29, 200, 120]);
  assert.deepEqual(moveRect(rect, 0, 0, SHEET), rect, 'not moved: the same slice');
  assert.deepEqual(moveRect(rect, -500, -500, SHEET), [0, 0, 200, 120], 'the top-left corner');
  assert.deepEqual(moveRect(rect, 5000, 5000, SHEET), [824, 392, 200, 120], 'the bottom-right corner');
  assert.deepEqual(moveRect(rect, 5000, 10, SHEET), [824, 60, 200, 120], 'along the right edge');
  assert.deepEqual(moveRect([0, 0, 1024, 512], 40, -40, SHEET), [0, 0, 1024, 512], 'as big as the picture: nowhere to go');
  assert.equal(usable(moveRect([0, 0, MIN_SIDE, MIN_SIDE], 9999, 9999, SHEET)), true);
});

test('the presets cut the picture into four cells that keep clear of its edges and of each other: quadrants front and back over ← left and right →, a row front, back, ← left, right →', () => {
  // An image generator often frames a sheet and draws a cross (or bars) between its views: the cells stay 2 % of the
  // picture away from its edges and 2 % apart, so a quadrant ends 1 % short of the middle.
  assert.deepEqual([PRESET_EDGE, PRESET_GAP], [0.02, 0.02]);
  const big = { w: 1000, h: 1000 };
  assert.deepEqual(presetRects('quad', big), [[20, 20, 470, 470], [510, 20, 470, 470], [20, 510, 470, 470], [510, 510, 470, 470]]);
  assert.deepEqual(presetRects('row', big), [[20, 20, 225, 960], [265, 20, 225, 960], [510, 20, 225, 960], [755, 20, 225, 960]]);
  assert.deepEqual(presetRects('quad', SHEET), [[20, 10, 482, 241], [522, 10, 482, 241], [20, 261, 482, 241], [522, 261, 482, 241]]);
  // Any size: inside the picture, apart from each other, and big enough for the hub.
  for (const size of [{ w: 1001, h: 333 }, { w: 1664, h: 2432 }, { w: 256, h: 256 }]) {
    for (const kind of ['quad', 'row']) {
      const rects = presetRects(kind, size);
      for (const [x, y, w, h] of rects) {
        assert.ok(x >= size.w * 0.015 && y >= size.h * 0.015 && x + w <= size.w * 0.985 && y + h <= size.h * 0.985, `${kind}: off the edges`);
        assert.ok(usable([x, y, w, h]));
      }
      const row = kind === 'row' ? rects : rects.slice(0, 2);
      row.slice(1).forEach(([x], i) => assert.ok(x - (row[i][0] + row[i][2]) >= size.w * 0.015, `${kind}: a gap between neighbours`));
    }
  }
  // Both in a sheet's reading order, as the picture generator is asked for one: front, back, ← left, right →.
  const cells = (kind) => Object.fromEntries(presetViews(kind, 'images/0001.png', SHEET).map((x) => [x.view, x.rect.slice(0, 2)]));
  assert.deepEqual(cells('quad'), { front: [20, 10], left: [20, 261], back: [522, 10], right: [522, 261] });
  assert.deepEqual(Object.keys(cells('row')).sort((a, b) => cells('row')[a][0] - cells('row')[b][0]), SHEET_ORDER);
  assert.deepEqual(presetViews('quad', 'images/0001.png', SHEET).map((x) => x.view), VIEW_ORDER, 'kept in the views\' order');
  assert.ok(presetViews('row', 'images/0001.png', SHEET).every((x) => x.src === 'images/0001.png'));
});

test('a view has one slice: drawing it again, on any picture, replaces it', () => {
  let views = [v('front', 'images/0001.png', [0, 0, 100, 100])];
  views = putView(views, v('back', 'images/0001.png', [100, 0, 100, 100]));
  views = putView(views, v('left', 'images/0002.png', [0, 0, 50, 50]));
  assert.deepEqual(views.map((x) => x.view), ['front', 'left', 'back'], 'kept in the views\' order');
  views = putView(views, v('front', 'images/0002.png', [10, 10, 60, 60]));
  assert.equal(views.filter((x) => x.view === 'front').length, 1);
  assert.deepEqual(views[0], v('front', 'images/0002.png', [10, 10, 60, 60]));
  assert.deepEqual(dropView(views, 'left').map((x) => x.view), ['front', 'back']);
  assert.deepEqual(dropPicture(views, 'images/0002.png').map((x) => x.view), ['back'], 'the slices of one picture go');
  assert.deepEqual(sortViews([v('right'), v('front'), v('back')]).map((x) => x.view), ['front', 'back', 'right']);
});

test('after a slice, the next view without one is picked, in the buttons\' order (Front, Back, ← Left, Right →); turning slicing on picks the first free one', () => {
  assert.deepEqual(SHEET_ORDER, ['front', 'back', 'left', 'right'], 'a sheet\'s reading order: the buttons\' and the quadrants\'');
  assert.equal(nextView([v('front')], 'front'), 'back');
  assert.equal(nextView([v('front'), v('back')], 'back'), 'left');
  assert.equal(nextView([v('front'), v('left'), v('right')], 'right'), 'back', 'goes round');
  assert.equal(nextView([v('front'), v('left'), v('back'), v('right')], 'back'), 'back', 'all done: stays');
  assert.equal(firstFree([]), 'front');
  assert.equal(firstFree([v('front'), v('left')]), 'back');
  assert.equal(firstFree([v('front'), v('back')]), 'left');
  assert.equal(firstFree([v('front'), v('left'), v('back'), v('right')]), 'front');
});

test('a slice is never smaller than asked and never outside the picture, wherever the drag begins', () => {
  const least = 30;
  // Begun beside the picture (the press is brought to its edge) and dragged along that edge, or barely at all.
  assert.deepEqual(dragRect({ x: 0, y: 100 }, { x: 0, y: 300 }, SHEET, false, least), [0, 100, 30, 200], 'along the left edge: it grows inwards');
  assert.deepEqual(dragRect({ x: 1024, y: 512 }, { x: 1024, y: 512 }, SHEET, false, least), [994, 482, 30, 30], 'in the corner');
  assert.deepEqual(dragRect({ x: 500, y: 200 }, { x: 495, y: 196 }, SHEET, false, least), [470, 170, 30, 30], 'up and to the left: it grows that way');
  assert.deepEqual(dragRect({ x: 10, y: 5 }, { x: 4, y: 2 }, SHEET, false, least), [0, 0, 30, 30], 'pushed back in from the corner');
  assert.deepEqual(dragRect({ x: 100, y: 100 }, { x: 400, y: 300 }, SHEET, false, least), [100, 100, 300, 200], 'a bigger drag is as it was');
  assert.deepEqual(dragRect({ x: 1024, y: 0 }, { x: 1024, y: 0 }, SHEET, true, least), [994, 0, 30, 30], 'a square too');
  assert.deepEqual(dragRect({ x: 100, y: 400 }, { x: 500, y: 450 }, SHEET, true, least), [100, 400, 112, 112]);
  for (const [a, b] of [[{ x: 0, y: 0 }, { x: 0, y: 0 }], [{ x: 1024, y: 300 }, { x: 900, y: 300 }], [{ x: 512, y: 512 }, { x: 700, y: 512 }]]) {
    for (const square of [false, true]) {
      const [x, y, w, h] = dragRect(a, b, SHEET, square, least);
      assert.ok(w >= least && h >= least && x >= 0 && y >= 0 && x + w <= SHEET.w && y + h <= SHEET.h, JSON.stringify([a, b, square, x, y, w, h]));
    }
  }
});

test('a picture is sliced in its own order, whatever the prop\'s other pictures hold: Front first, then Back', () => {
  const sheet = 'images/0005.png';
  // An earlier picture kept a Front slice: the new sheet still starts at its own Front (it was handed "Back" for
  // its front figure and "← Left" for its back one, and putting the second right then swapped the first away).
  const earlier = [v('front', 'images/0003.png', [0, 0, 64, 64])];
  assert.equal(firstFree(earlier, sheet), 'front');
  const first = putView(earlier, v('front', sheet, [0, 0, 64, 64]));
  assert.deepEqual(first.map((x) => [x.view, x.src]), [['front', sheet]], 'the view moves to the new sheet');
  assert.equal(nextView(first, 'front', sheet), 'back');
  assert.equal(nextView([...first, v('back', 'images/0003.png', [64, 0, 64, 64])], 'front', sheet), 'back',
    'a Back sliced on another sheet does not skip it');
  // Pictures brought in one per side are not sheets: a side that is another picture's whole stays with that picture
  // while another side is free here (the first slice on a second picture took the Front away, and no Front was left).
  const perSide = [v('front', 'images/upload_001.png')];
  assert.equal(firstFree(perSide, sheet), 'back');
  assert.equal(nextView(putView(perSide, v('back', sheet, [0, 0, 64, 64])), 'back', sheet), 'left');
  assert.equal(firstFree([v('front', 'images/upload_001.png'), v('back', sheet), v('left', sheet), v('right', sheet)], sheet), 'front',
    'the only side left to slice here: taken then');
  const imported = VIEW_ORDER.map((view, i) => v(view, `images/upload_00${i + 1}.png`));
  assert.equal(firstFree(imported, sheet), 'front');
  const one = putView(imported, v('front', sheet, [0, 0, 64, 64]));
  assert.equal(nextView(one, 'front', sheet), 'back');
  const two = putView(one, v('back', sheet, [64, 0, 64, 64]));
  assert.equal(nextView(two, 'back', sheet), 'left');
  assert.equal(firstFree(two, sheet), 'left');
  const all = VIEW_ORDER.map((view, i) => v(view, sheet, [i * 64, 0, 64, 64]));
  assert.equal(nextView(all, 'right', sheet), 'right', 'all on this sheet: stays');
  assert.equal(firstFree(all, sheet), 'front');
});

test('a chip changes its slice\'s view: the slice that had that view takes the old one', () => {
  const views = [v('front', 'a', [0, 0, 20, 20]), v('left', 'a', [20, 0, 20, 20]), v('right', 'a', [40, 0, 20, 20])];
  const swapped = relabel(views, 'left', 'right');
  assert.deepEqual(swapped.map((x) => [x.view, x.rect[0]]), [['front', 0], ['left', 40], ['right', 20]]);
  assert.deepEqual(relabel(views, 'left', 'back').map((x) => [x.view, x.rect[0]]), [['front', 0], ['back', 20], ['right', 40]]);
  assert.equal(relabel(views, 'left', 'left'), views);
});

test('several views need a front; the same views in another order are the same', () => {
  assert.equal(needsFront([]), false);
  assert.equal(needsFront([v('back')]), false, 'one view is used as it is');
  assert.equal(needsFront([v('back'), v('left')]), true);
  assert.equal(needsFront([v('back'), v('front')]), false);
  assert.equal(sameViews([v('back'), v('front')], [v('front'), v('back')]), true);
  assert.equal(sameViews([v('front', 'a', [0, 0, 20, 20])], [v('front', 'a', [0, 0, 20, 21])]), false);
});

test('Ctrl+Z goes back one change at a time', () => {
  const past = history(3);
  assert.equal(past.undo(), null, 'nothing to undo');
  const states = [[], [v('front')], [v('front'), v('left')], [v('front'), v('left'), v('back')]];
  for (const s of states) past.record(s);
  assert.equal(past.size, 3, 'the oldest are forgotten past the limit');
  assert.deepEqual(past.undo(), states[3]);
  assert.deepEqual(past.undo(), states[2]);
  assert.deepEqual(past.undo(), states[1]);
  assert.equal(past.undo(), null);
});

test('pictures imported together: the sides their names say, the others in name order, the front first', () => {
  // [file name, side, said by the name] in upload order
  const sides = (names) => importSides(names).map((x) => [names[x.index], x.side, x.told]);
  // A file picker gives the focused file first; an Explorer drag the one under the pointer.
  assert.deepEqual(sides(['bear_right.png', 'bear_back.png', 'bear_front.png', 'bear_left.png']), [
    ['bear_front.png', 'front', true], ['bear_left.png', 'left', true], ['bear_back.png', 'back', true],
    ['bear_right.png', 'right', true]]);
  assert.deepEqual(sides(['Knight-Back.PNG', 'C:\\art\\knight front.jpg']), [
    ['C:\\art\\knight front.jpg', 'front', true], ['Knight-Back.PNG', 'back', true]], 'a front and a back: no side view');
  assert.deepEqual(sides(['rear.webp', 'side.png', 'front.png']), [
    ['front.png', 'front', true], ['side.png', 'left', true], ['rear.webp', 'back', true]], '"side" takes a free side');
  // Nothing said: name order, numbers as numbers ("2" before "10").
  assert.deepEqual(sides(['shot10.png', 'shot2.png', 'shot1.png']), [
    ['shot1.png', 'front', false], ['shot2.png', 'left', false], ['shot10.png', 'back', false]]);
  // Angles around the object, when the names give several; one number alone is no angle.
  assert.deepEqual(sides(['azim270.png', 'azim000.png', 'azim180.png', 'azim090.png']).map((x) => x[1]),
    ['front', 'left', 'back', 'right']);
  assert.deepEqual(sides(['azim180.png', 'azim000.png']).map((x) => [x[0], x[1]]), [['azim000.png', 'front'], ['azim180.png', 'back']]);
  assert.deepEqual(sides(['view_1.png', 'view_0.png']).map((x) => [x[0], x[1], x[2]]),
    [['view_0.png', 'front', false], ['view_1.png', 'left', false]], '0 and 1: a count, not angles');
  // Mixed: the named ones keep their sides; the others fill the free ones in name order.
  assert.deepEqual(sides(['b.png', 'dragon_back.png', 'a.png']), [
    ['a.png', 'front', false], ['b.png', 'left', false], ['dragon_back.png', 'back', true]]);
  // A side said twice: the first in name order has it, the other is just a picture; so is a fifth one.
  assert.deepEqual(sides(['front_2.png', 'front_1.png', 'x.png']), [
    ['front_1.png', 'front', true], ['x.png', 'left', false], ['front_2.png', null, false]]);
  assert.deepEqual(sides(['1.png', '2.png', '3.png', '4.png', '5.png']).map((x) => x[1]), ['front', 'left', 'back', 'right', null]);
  // One picture is no view; a word inside another word is no side ("frontier", "backpack").
  assert.deepEqual(sides(['bear_back.png']), [['bear_back.png', null, false]]);
  assert.deepEqual(sides(['backpack.png', 'frontier.png']).map((x) => [x[0], x[1], x[2]]),
    [['backpack.png', 'front', false], ['frontier.png', 'left', false]]);
});

test('Ctrl+Z after an import of several pictures brings back the views and the picture it had', () => {
  const past = history();
  const sheet = [v('front', 'images/0001.png', [0, 0, 20, 20]), v('left', 'images/0001.png', [20, 0, 20, 20])];
  past.record({ views: sheet, chosen: 'images/0001.png' });
  assert.deepEqual(past.undo(), { views: sheet, chosen: 'images/0001.png' });
});

test('another picture taken up while slicing starts at its own first view', async () => {
  // Front and Back were drawn on one sheet, so "← Left" is picked; the next sheet must not be handed "← Left" for
  // its front figure.
  const { store } = await import(new URL('../../spellcast/static/js/store.js', import.meta.url));
  const first = 'images/0001.png';
  store.project = { id: 'Props', name: 'Props', ui: {}, props: [{ id: 'p_7', images: {
    chosen: first, views: [v('front', first, [0, 0, 64, 64]), v('back', first, [64, 0, 64, 64])] } }] };
  const host = new FakeEl('div');
  const slicer = createSlicer('p_7', { toolbar: host });
  const loaded = () => Object.assign(new FakeEl('img'), { complete: true, naturalWidth: SHEET.w, naturalHeight: SHEET.h });
  slicer.picture(loaded(), first);
  const [views, acts] = host.children[0].children;
  const picked = () => views.children.find((b) => b.classList.contains('on')).textContent;
  acts.children[2].click();                        // "Slice into views"
  assert.equal(picked(), '← Left', 'this sheet has its Front and Back');
  slicer.picture(loaded(), 'images/0002.png');
  assert.equal(picked(), 'Front', 'a sheet without slices: its Front first');
  slicer.picture(loaded(), first);
  assert.equal(picked(), '← Left');
  slicer.destroy();
});

test('the strip\'s chips say the side in words', () => {
  assert.deepEqual(VIEW_ORDER.map((k) => VIEWS[k].short), ['Front', '← Left', 'Back', 'Right →']);
});

test('the slicer\'s buttons are one row in the header it is given, while a picture shows and none is being made', () => {
  const host = new FakeEl('div');
  const slicer = createSlicer('p_1', { toolbar: host });
  assert.deepEqual(host.children, [], 'no picture: nothing in the header');
  const picture = Object.assign(new FakeEl('img'), { complete: true, naturalWidth: SHEET.w, naturalHeight: SHEET.h });
  slicer.picture(picture, 'images/0001.png');
  assert.equal(host.children.length, 1, 'one row');
  const [bar] = host.children;
  const [views, acts] = bar.children;
  assert.equal(acts.className, 'sl-acts', 'the buttons are one group: it goes under the views as a whole');
  const [quad, row, toggle, bin, copy] = acts.children;
  assert.deepEqual(views.children.map((b) => b.textContent), ['Front', 'Back', '← Left', 'Right →'], 'a sheet\'s reading order');
  assert.deepEqual([quad, row, toggle, bin, copy].map((b) => b.textContent), ['4 quadrants', '4 in a row', 'Slice into views', '', 'Copy'],
    'Copy is the last');
  assert.equal(views.hidden, true, 'the four views show while slicing only');
  assert.equal(bin.hidden, true, 'the bin shows when the picture has slices');
  toggle.click();
  assert.equal(views.hidden, false, '"Slice into views" was pressed');
  // Only the slices and what the views lack lie over the picture.
  assert.equal(slicer.el.hidden, false);
  assert.deepEqual(slicer.el.byClass('sl-bar'), []);
  assert.equal(slicer.el.byClass('sl-hints').length, 1);
  slicer.update({ busy: true });
  assert.deepEqual(host.children, [], 'a picture is being made');
  slicer.update({ busy: false });
  assert.deepEqual(host.children, [bar], 'the same row again');
  // The next picture is not loaded yet: the row stays, so the header keeps its height.
  slicer.picture(new FakeEl('img'), 'images/0002.png');
  assert.equal(slicer.el.hidden, true);
  assert.deepEqual(host.children, [bar]);
  slicer.picture(null, null);
  assert.deepEqual(host.children, []);
  slicer.picture(picture, 'images/0001.png');
  slicer.destroy();
  assert.deepEqual(host.children, [], 'gone with the slicer');
});

test('a slicer without a header to be in has its row over the picture', () => {
  const slicer = createSlicer('p_1');
  assert.equal(slicer.el.byClass('sl-bar').length, 1);
  assert.equal(slicer.el.byClass('sl-toggle').length, 1);
  slicer.destroy();
});

test('the bin takes the picture\'s slices away and ends slicing: the views to draw next go with them', async () => {
  const { store } = await import(new URL('../../spellcast/static/js/store.js', import.meta.url));
  store.project = { id: 'Props', name: 'Props', props: [{ id: 'p_9', images: { views: [] } }], ui: {} };
  globalThis.fetch = async () => ({ status: 200, text: async () => '{}' });
  const host = new FakeEl('div');
  const slicer = createSlicer('p_9', { toolbar: host });
  slicer.picture(Object.assign(new FakeEl('img'), { complete: true, naturalWidth: SHEET.w, naturalHeight: SHEET.h }), 'images/0001.png');
  const [views, acts] = host.children[0].children;
  const [quad, , , bin] = acts.children;
  quad.click();
  assert.deepEqual([views.hidden, bin.hidden], [false, false], '4 quadrants: sliced, the views to draw next shown');
  bin.click();
  assert.deepEqual([views.hidden, bin.hidden], [true, true]);
  assert.deepEqual(store.project.props[0].images.views, []);
  slicer.destroy();
  await new Promise((r) => setTimeout(r, 350));   // the views' save goes out
});
