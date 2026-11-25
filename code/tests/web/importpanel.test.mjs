// The Import tab of the Images workbench (no browser): which slot shows what for a prop's pictures and views,
// which area is in use (Single view or Multi-view), what filling and emptying a slot does to the views and
// the chosen picture, and the panel itself on a fake page (two areas, four slots, a "Paste" each, no Ctrl+V).
//   node --test code/tests/web/importpanel.test.mjs
// (code/tests/test_web.py runs every code/tests/web/*.test.mjs when Node.js is installed)
import assert from 'node:assert/strict';
import { test } from 'node:test';

// ---- a page, as far as the panel uses one: elements with attributes, children, classes and text ----

class FakeNode {}
class FakeEl extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attrs = {};
    this.className = '';
    this.style = {};
    this.listeners = {};
    this.hidden = false;
    this.disabled = false;
  }

  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes.filter((n) => n != null); }
  get textContent() { return this.children.map((c) => (typeof c === 'string' ? c : c.textContent)).join(''); }
  set textContent(v) { this.children = [String(v)]; }
  get classList() {
    const el = this;
    const list = () => el.className.split(/\s+/).filter(Boolean);
    return {
      contains: (c) => list().includes(c),
      add: (c) => { if (!list().includes(c)) el.className = [...list(), c].join(' '); },
      remove: (c) => { el.className = list().filter((x) => x !== c).join(' '); },
      toggle: (c, on = !list().includes(c)) => {
        el.className = on ? [...new Set([...list(), c])].join(' ') : list().filter((x) => x !== c).join(' ');
        return on;
      },
    };
  }

  /** Every element in it (itself first) with the class `name`. */
  byClass(name) {
    const out = this.classList.contains(name) ? [this] : [];
    for (const c of this.children) if (c instanceof FakeEl) out.push(...c.byClass(name));
    return out;
  }
}
class FakeTemplate extends FakeEl {
  set innerHTML(html) { this.html = html; }
  get content() { return { firstChild: new FakeEl('svg') }; }
}
const pageListeners = [];   // what anything asked the page itself to listen for
globalThis.Node = FakeNode;
globalThis.addEventListener ??= () => {};
globalThis.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} };
globalThis.document = {
  addEventListener: (type) => pageListeners.push(type), removeEventListener: () => {}, hidden: false, activeElement: null,
  documentElement: { dataset: {} },
  createElement: (tag) => (tag === 'template' ? new FakeTemplate(tag) : new FakeEl(tag)),
  createElementNS: (ns, tag) => new FakeEl(tag),
};

const STATIC = new URL('../../spellcast/static/js/', import.meta.url);
const load = (path) => import(new URL(path, STATIC));

const { store } = await load('store.js');
const { hubJob } = await import('./hub_job.mjs');
const { effectiveViews } = await load('stages/rules.js');
const {
  SLOTS, createImportPanel, cropBox, emptySlot, fillSlot, inUse, needsFront, newPicture, singleOf, slotsOf, waiting,
} = await load('stages/importpanel.js');

const A = 'images/0001.png';
const B = 'images/0002.png';
const C = 'images/0003.png';
const SHEET = 'images/0004.png';
const whole = (view, src) => ({ view, src, rect: null });

test('the slots: Front, Back, ← Left, Right →, and which way the object faces in the last two (the slicer’s words)', () => {
  assert.deepEqual(SLOTS.map((s) => s.label), ['Front', 'Back', '← Left', 'Right →']);
  assert.deepEqual(SLOTS.map((s) => s.view), ['front', 'back', 'left', 'right']);
  assert.equal(SLOTS[2].hint, 'The object faces left');
  assert.equal(SLOTS[3].hint, 'The object faces right');
  assert.equal(SLOTS.some((s) => /side/i.test(`${s.label} ${s.hint}`)), false, 'never "Side"');
});

test('a slot shows the view of its side: a whole picture, or a slice of a sheet', () => {
  assert.deepEqual(slotsOf({}).map((s) => s.src), [null, null, null, null], 'a prop without views: four empty slots');
  const images = {
    candidates: [A, B, SHEET], chosen: A,
    views: [whole('front', A), whole('left', B), { view: 'back', src: SHEET, rect: [512, 0, 512, 512] }],
  };
  const slots = slotsOf(images);
  assert.deepEqual(slots.map((s) => [s.view, s.src, s.rect]), [
    ['front', A, null], ['back', SHEET, [512, 0, 512, 512]], ['left', B, null], ['right', null, null],
  ]);
  assert.equal(slots[0].label, 'Front');
  // A view on a picture the prop no longer has shows nothing.
  assert.equal(slotsOf({ ...images, candidates: [A, SHEET] })[2].src, null);
});

test('Single view or Multi-view: the area the 3D model is made from now', () => {
  assert.equal(inUse({}), null, 'no picture yet: neither');
  assert.equal(inUse({ candidates: [A], chosen: A }), 'single');
  assert.deepEqual(singleOf({ candidates: [A], chosen: A }), { src: A, rect: null });
  const two = { candidates: [A, B, C], chosen: A, views: [whole('front', A), whole('left', B)] };
  assert.equal(inUse(two), 'multi');
  assert.equal(singleOf(two), null, 'several views are used: no single picture');
  // Another picture chosen: it is made alone, the views wait.
  const aside = { ...two, chosen: C };
  assert.equal(inUse(aside), 'single');
  assert.deepEqual(singleOf(aside), { src: C, rect: null });
  assert.deepEqual(waiting(aside), { count: 2, chosen: A }, 'using them again chooses the front picture');
  assert.equal(waiting(two), null);
  // One view alone is one picture (its slice of it).
  const one = { candidates: [SHEET], chosen: SHEET, views: [{ view: 'front', src: SHEET, rect: [0, 0, 512, 512] }] };
  assert.equal(inUse(one), 'single');
  assert.deepEqual(singleOf(one), { src: SHEET, rect: [0, 0, 512, 512] });
  assert.equal(waiting(one), null);
});

test('filling a slot: a view of the whole picture for that side, and the views are used', () => {
  // The first slot filled (the upload made B the chosen picture).
  let images = { candidates: [A, B], chosen: B, views: [] };
  let next = fillSlot(images, 'left', B);
  assert.deepEqual(next, { views: [whole('left', B)], chosen: B });
  // Front added: the front picture is the chosen one.
  images = { candidates: [A, B, C], chosen: C, views: next.views };
  next = fillSlot(images, 'front', C);
  assert.deepEqual(next.views, [whole('front', C), whole('left', B)], 'in the views’ own order');
  assert.equal(next.chosen, C);
  // Another side: the front stays chosen although the upload chose the new picture.
  images = { candidates: [A, B, C, SHEET], chosen: SHEET, views: next.views };
  next = fillSlot(images, 'right', SHEET);
  assert.equal(next.chosen, C);
  assert.equal(effectiveViews({ ...images, ...next }).length, 3, 'all three are used');
  // A side filled again: its earlier view is replaced (also a slice made in the slicer).
  const sliced = { candidates: [A, SHEET], chosen: A, views: [{ view: 'front', src: SHEET, rect: [0, 0, 512, 512] }, { view: 'back', src: SHEET, rect: [512, 0, 512, 512] }] };
  next = fillSlot(sliced, 'back', A);
  assert.deepEqual(next.views, [{ view: 'front', src: SHEET, rect: [0, 0, 512, 512] }, whole('back', A)]);
  assert.equal(next.chosen, SHEET);
  assert.equal(needsFront({ ...sliced, ...next }), false);
});

test('emptying a slot drops its view; the views left stay used', () => {
  const images = { candidates: [A, B, C], chosen: A, views: [whole('front', A), whole('left', B), whole('back', C)] };
  let next = emptySlot(images, 'left');
  assert.deepEqual(next, { views: [whole('front', A), whole('back', C)], chosen: A });
  // The front emptied: one of the views left is chosen, and the front is asked for.
  next = emptySlot(images, 'front');
  assert.deepEqual(next.views, [whole('left', B), whole('back', C)]);
  assert.equal(next.chosen, B);
  assert.equal(needsFront({ ...images, ...next }), true);
  assert.equal(effectiveViews({ ...images, ...next }).length, 2);
  // The last one emptied: the chosen picture stays, alone.
  assert.deepEqual(emptySlot({ candidates: [A], chosen: A, views: [whole('front', A)] }, 'front'), { views: [], chosen: A });
  // The views were waiting (another picture is made alone): emptying a slot leaves that picture chosen.
  assert.equal(emptySlot({ ...images, candidates: [A, B, C, SHEET], chosen: SHEET }, 'left').chosen, SHEET);
  // All four on one sheet (4 quadrants): the sheet stays chosen.
  const quad = ['front', 'left', 'back', 'right'].map((view, i) => ({ view, src: SHEET, rect: [i * 10, 0, 10, 10] }));
  assert.equal(emptySlot({ candidates: [SHEET], chosen: SHEET, views: quad }, 'front').chosen, SHEET);
});

test('the picture an upload added, and where a slice’s picture goes in its small frame', () => {
  assert.equal(newPicture([A], { candidates: [A, B], chosen: B }), B);
  assert.equal(newPicture([A], { candidates: [A], chosen: A }), A, 'nothing new: the chosen one');
  assert.equal(newPicture([], {}), null);
  // The right half of a 1024 x 512 sheet: the picture is twice the frame's width, moved one frame left.
  assert.deepEqual(cropBox([512, 0, 512, 512], { w: 1024, h: 512 }), { width: 200, height: 100, left: -100, top: -0 });
});

// ---- the panel on the fake page ----------------------------------------------------------------------

function open(images) {
  store.connected = true;
  store.jobs = [];
  store.jobErrors = {};
  store.status = null;
  store.bust = {};
  store.openStamp = 7;
  store.project = { id: 'Props', name: 'Props', props: [{ id: 'p_1', name: 'Barrel', images }] };
  return createImportPanel('p_1');
}
const text = (el) => el.textContent;

test('the panel: Single view over Multi-view, four slots with a Paste each, the one in use marked', () => {
  const panel = open({ candidates: [A, B, SHEET], chosen: A, views: [whole('front', A), { view: 'left', src: SHEET, rect: [0, 0, 64, 64] }] });
  const areas = panel.el.byClass('ip-area');
  assert.deepEqual(areas.map((a) => text(a.byClass('ip-head')[0])), ['Single view', 'Multi-viewIn use']);
  assert.equal(areas[1].classList.contains('on'), true);
  assert.equal(areas[0].classList.contains('on'), false);
  const slots = panel.el.byClass('ip-slot');
  assert.deepEqual(slots.map((s) => text(s.byClass('ip-slot-name')[0].children[0])), ['Front', 'Back', '← Left', 'Right →']);
  assert.deepEqual(slots.map((s) => s.classList.contains('filled')), [true, false, true, false]);
  assert.deepEqual(slots.map((s) => s.byClass('ip-x').length), [1, 0, 1, 0], 'an × on the filled ones');
  assert.deepEqual(slots.map((s) => s.byClass('ip-slice').length), [0, 0, 1, 0], 'a slice of a sheet says so');
  assert.equal(panel.el.byClass('ip-paste').length, 5, 'Single view’s and one per slot');
  assert.match(text(slots[2]), /The object faces left/);
  assert.match(text(slots[3]), /The object faces right/);
  assert.match(text(areas[1]), /Front plus at least one more/);
  assert.equal(panel.el.byClass('ip-thumb').length, 2, 'the two filled slots; Single view shows none while views are used');
  assert.equal(pageListeners.includes('paste'), false, 'no Ctrl+V here');
  assert.equal(pageListeners.includes('keydown'), false);

  // Another picture chosen: Single view is in use and shows it; the views wait and can be used again.
  store.project.props[0].images = { ...store.project.props[0].images, chosen: B };
  panel.update();
  assert.deepEqual(areas.map((a) => a.classList.contains('on')), [true, false]);
  assert.equal(areas[0].byClass('ip-thumb').length, 1);
  assert.match(text(areas[1]), /Not used now/);
  assert.match(text(areas[1]), /Use these views/);
  panel.destroy();
});

test('the panel waits while the prop’s pictures are being made', () => {
  const panel = open({ candidates: [A], chosen: A, views: [] });
  const buttons = () => [...panel.el.byClass('ip-zone'), ...panel.el.byClass('ip-slot-zone'), ...panel.el.byClass('ip-paste')];
  assert.equal(buttons().length, 10);
  assert.equal(buttons().some((b) => b.disabled), false);
  assert.equal(panel.el.byClass('ip-status')[0].hidden, true);
  store.jobs = [hubJob({ id: 'j1', stage: 'images', status: 'running', prop_ids: ['p_1'], current_prop: 'p_1', progress: {} })];
  panel.update();
  assert.equal(buttons().every((b) => b.disabled), true);
  assert.equal(panel.el.byClass('ip-status')[0].hidden, false);
  assert.match(text(panel.el.byClass('ip-status')[0]), /being made/);
  store.jobs = [];
  panel.update();
  assert.equal(buttons().some((b) => b.disabled), false);
  panel.destroy();
});
