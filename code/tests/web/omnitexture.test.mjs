// The Omni-Texture stage on a fake page, with a stand-in viewer and graphics card (standin_viewer3d.mjs,
// standin_card.mjs): its tools - Projection's marker and clone stamp, its own mask marker at the column's foot - and
// the hand paint they leave, which is in its files before the page has the hub make the steps again (Render Masked, a
// fade, a picture taken off, a job): the hub takes no paint while that is under way. A mask stroke is never saved, and
// the mask outlives a model made again. The view moves while the mask marker is up, the box beside the mask with
// it; "Render Masked" sends the view as it is at the press.
//   node --test code/tests/web/omnitexture.test.mjs
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import { strokeAt } from './standin_card.mjs';

// ---- a page, as far as these modules use one: elements with a parent, attributes, children, classes, text ----

class FakeNode {}
class FakeEl extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attrs = {};
    this.className = '';
    this.style = { setProperty(k, v) { this[k] = v; } };
    this.dataset = {};
    this.listeners = {};
    this.hidden = false;
    this.disabled = false;
    this.parentNode = null;
    this.offsetWidth = this.offsetHeight = 0;
  }

  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter((f) => f !== fn); }
  fire(type, event = {}) { for (const fn of this.listeners[type] || []) fn({ preventDefault() {}, stopPropagation() {}, target: this, ...event }); }
  click() { this.fire('click'); }
  showModal() { this.open = true; }   // (a <dialog>, as far as the page's windows use one)
  close() { this.open = false; this.fire('close'); }
  remove() {
    const siblings = this.parentNode?.children;
    if (siblings) siblings.splice(siblings.indexOf(this), 1);
    this.parentNode = null;
  }

  adopt(node) {
    if (node instanceof FakeEl) {
      node.remove();
      node.parentNode = this;
    }
    return node;
  }

  append(...nodes) { for (const n of nodes) this.children.push(this.adopt(n)); }
  prepend(...nodes) { this.children.unshift(...nodes.map((n) => this.adopt(n))); }
  replaceChildren(...nodes) {
    for (const c of [...this.children]) if (c instanceof FakeEl) c.remove();
    this.children = [];
    this.append(...nodes);
  }

  replaceWith(node) {
    const parent = this.parentNode;
    if (!parent) return;
    parent.children.splice(parent.children.indexOf(this), 1, parent.adopt(node));
    this.parentNode = null;
  }

  get firstChild() { return this.children[0] ?? null; }
  get isConnected() { return true; }
  get textContent() { return this.children.map((c) => (typeof c === 'string' ? c : c.textContent)).join(''); }
  set textContent(v) { this.replaceChildren(String(v)); }
  get title() { return this.attrs.title ?? ''; }
  set title(v) { this.attrs.title = String(v); }
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

  animate() { return { finished: Promise.resolve(), playState: 'finished', cancel() {}, finish() {} }; }
  getBoundingClientRect() { return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }; }
  focus() {}
  blur() {}
  matches() { return false; }
  // (a canvas, as far as the mask's picture is made on one)
  getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData() {} }; }
  toBlob(done) { done({ arrayBuffer: async () => new ArrayBuffer(8) }); }

  /** Every element in it (itself first) that `pick` accepts. */
  all(pick) {
    const out = pick(this) ? [this] : [];
    for (const c of this.children) if (c instanceof FakeEl) out.push(...c.all(pick));
    return out;
  }

  byClass(name) { return this.all((e) => e.classList.contains(name)); }
  querySelectorAll(sel) {
    const classes = sel.trim().split(/\s+/).at(-1).split('.').filter(Boolean);
    return this.all((e) => e !== this && classes.every((c) => e.classList.contains(c)));
  }

  querySelector(sel) { return this.querySelectorAll(sel)[0] ?? null; }
}
class FakeTemplate extends FakeEl {
  set innerHTML(html) { this.html = html; }
  get content() { return { firstChild: new FakeEl('svg') }; }
}
const pageKeys = [];
globalThis.Node = FakeNode;
globalThis.addEventListener ??= () => {};
globalThis.document = {
  addEventListener: (type, fn) => { if (type === 'keydown') pageKeys.push(fn); },
  removeEventListener: (type, fn) => { if (pageKeys.includes(fn)) pageKeys.splice(pageKeys.indexOf(fn), 1); },
  hidden: false, activeElement: null, documentElement: { dataset: {} }, body: new FakeEl('body'),
  querySelector: () => null,
  getElementById: () => new FakeEl('div'),
  createElement: (tag) => (tag === 'template' ? new FakeTemplate(tag) : new FakeEl(tag)),
  createElementNS: (ns, tag) => new FakeEl(tag),
};
// The page's polls and toasts wait a second or more: here they never come.
const realTimeout = globalThis.setTimeout;
globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 1000 ? 0 : realTimeout(fn, ms, ...rest));

/** A key pressed on the page, as far as its listeners let it go. */
function keyDown(key, mods = {}) {
  let stopped = false;
  const e = { type: 'keydown', key, ...mods, preventDefault() {}, stopImmediatePropagation() { stopped = true; } };
  for (const fn of [...pageKeys]) {
    if (stopped) break;
    fn(e);
  }
}

// viewer3d.js is the stand-in viewer here.
const VIEWER = new URL('./standin_viewer3d.mjs', import.meta.url).href;
registerHooks({
  resolve: (specifier, context, next) => {
    const found = next(specifier, context);
    return found.url.endsWith('/static/js/viewer3d.js') ? { url: VIEWER, shortCircuit: true } : found;
  },
});
const { viewer } = await import(VIEWER);
const load = (path) => import(new URL(`../../spellcast/static/js/${path}`, import.meta.url));
const { store } = await load('store.js');
const { undoList, undoStamp } = await load('undo.js');
const { default: OMNI } = await load('stages/omnitexture.js');
const { hubJob } = await import('./hub_job.mjs');

// ---- a prop with an Omni-Texture and one repaint on it, and a hub that holds the hand paint's saves ----------------

const RESULT = 'omnitexture/ot_001/omnitexture.glb';
const REPAINT = 'omnitexture/ot_001/retouch_001/omnitexture.glb';
const MADE_AGAIN = 'omnitexture/ot_001/remove_001/omnitexture.glb';
const API = '/api/projects/Props/props/p_1';

const prop = (glb = REPAINT, steps = [RESULT, REPAINT]) => ({
  id: 'p_1', name: 'Barrel', images: { chosen: 'images/0001.png', candidates: ['images/0001.png'], views: [] },
  mesh: { glb: 'mesh/mesh.glb', settings: {} },
  omnitexture: {
    settings: { chosen: true }, glb, steps, source_mesh: 'mesh/mesh.glb', stats: {},
    views: ['omnitexture/ot_001/views/0-front/painted.png', 'omnitexture/ot_001/views/1-back/painted.png'],
    repaints: [{ glb: REPAINT, picture: 'omnitexture/ot_001/retouch_001/picture.png', under: RESULT, fade: 1 }],
  },
  retopo: { obj: null, settings: {} },
});
/** A second prop, chosen and waiting for its Omni-Texture: "Omni-Texture the chosen" has it to paint. */
const waiting = () => ({
  id: 'p_2', name: 'Crate', images: { chosen: 'images/0001.png', candidates: ['images/0001.png'], views: [] },
  mesh: { glb: 'mesh/mesh.glb', settings: {} }, omnitexture: { settings: { chosen: true } }, retopo: { obj: null, settings: {} },
});

/** A prop the artist never OmniTextured (and has not chosen): its 3D model alone. */
const plain = () => ({
  id: 'p_1', name: 'Guy', images: { chosen: 'images/0001.png', candidates: ['images/0001.png'], views: [] },
  mesh: { glb: 'mesh/mesh.glb', settings: {} }, omnitexture: { settings: { chosen: false } }, retopo: { obj: null, settings: {} },
});
/** ... and with the result of its own texture the hub begins for it: no views, no steps yet. */
const begun = () => ({
  ...plain(),
  omnitexture: { settings: { chosen: false }, glb: RESULT, steps: [], source_mesh: 'mesh/mesh.glb', stats: null, views: [], repaints: [] },
});

// job: what the newest job was asked with; asked: the newest change of any kind
const hub = { calls: [], hold: null, answer: null, job: null, asked: null };
const answer = (body) => ({ status: 200, text: async () => JSON.stringify(body) });
globalThis.fetch = async (path, init = {}) => {
  const method = init.method || 'GET';
  const body = typeof init.body === 'string' ? JSON.parse(init.body) : null;
  hub.calls.push(`${method} ${path.split('?')[0]}${body?.prepare ? ' (prepare)' : ''}`);
  if (method !== 'GET') hub.asked = body;
  if (path.startsWith('/api/jobs')) {
    if (method === 'POST') hub.job = body;
    return answer(method === 'POST' ? { id: 'j_1', stage: 'omnitexture', status: 'queued', prop_ids: body.prop_ids, props: {} } : []);
  }
  if (path.includes('/omnitexture/paint')) {
    await hub.hold;
    return answer(null);
  }
  if (path.includes('/omnitexture/mask')) return answer({ mask: 'omnitexture/masks/0001.png' });
  if (path.includes('/omnitexture/')) return answer(hub.answer ? hub.answer(path, body) : { prop: store.project.props[0], renamed: {} });
  return answer({});
};

/** What the page asked the hub to change, in order (its reads and a fade's readying left out). */
const changes = () => hub.calls.filter((c) => !c.startsWith('GET') && !c.endsWith('(prepare)'));
const settle = async () => { for (let i = 0; i < 8; i++) await new Promise((resolve) => realTimeout(resolve, 0)); };

let opened = 0;
/** The stage on the first prop, its result on screen with the tools bound, no tool up; the hand paint's saves held. */
async function mount(props = [prop()]) {
  OMNI.unmount();
  store.connected = true;
  store.jobs = [];
  store.jobErrors = {};
  store.status = {
    tools: { qwen: { state: 'stopped', installed: true, supported: true, uses_gpu: true } },
    stage_tools: { images: ['qwen'], omnitexture: ['qwen'] }, install: { running: false, components: {} },
  };
  store.project = { id: 'Props', name: 'Props', props, ui: {} };
  store.bust = {};
  store.openStamp = ++opened;          // a model of its own for each test
  hub.calls = [];
  hub.answer = null;
  hub.job = null;
  lookFrom(FRONT);
  let release;
  hub.hold = new Promise((resolve) => { release = resolve; });
  const host = new FakeEl('div');
  const ctx = { selectedId: 'p_1', stageId: 'omnitexture', project: store.project, select() {}, goto() {}, moving: false };
  OMNI.mount(host, ctx);
  await settle();
  const label = (node) => node.getAttribute('aria-label');
  const tools = host.byClass('pt-tools')[0];
  // The column's buttons, top to bottom (the stage's own tool stands with the x that wipes it, its lens with the
  // glass of its slider).
  const buttons = () => tools.children.flatMap((b) => (b.className === 'pt-own' ? b.children : [b]))
    .filter((b) => b.tagName !== 'SPAN');
  const tool = (name) => buttons().find((b) => label(b) === name);
  tool('Orbit').click();               // (the brush is the page's: a tool may still be up from the view before)
  return { host, ctx, release, tools, buttons, tool, label, update: () => OMNI.update(ctx) };
}

// Where the artist looks from (the stand-in camera's place): as every test begins, and after going round the model.
const FRONT = [0, 0, 2];
const SIDE = [2, 0.5, 0];
function lookFrom(place) {
  viewer.camera.position.fromArray(place);
  viewer.camera.updateMatrixWorld();
}
const lookingFrom = () => viewer.camera.position.toArray();

/** A stroke of the marker on the base colour, its save due. */
function paintBase(stage, rect = { x: 8, y: 8, w: 4, h: 4 }) {
  stage.tool('Marker of any colour').click();
  strokeAt(viewer, rect);
}

/** The step-making call `then` waits for the paint's save: none of it is asked for until that is in the files. */
async function waitsForThePaint(stage, then) {
  await settle();
  assert.deepEqual(changes(), [`PUT ${API}/omnitexture/paint`], 'the hand paint goes first ...');
  stage.release();
  await settle();
  assert.deepEqual(changes(), [`PUT ${API}/omnitexture/paint`, ...then], '... and only once it is in the files, the rest');
}

test('the Omni-Texture result has Projection\'s tools in the column, Orbit on top, its mask marker at the foot with the x that discards the mask, the hint left of the glass', async () => {
  const stage = await mount();
  const WIPE = 'Discard the mask (Ctrl+Z brings it back)';
  assert.deepEqual(stage.buttons().map(stage.label),
    ['Orbit', 'Clone stamp', 'Marker of any colour', WIPE, 'Mask marker', 'Inverse perspective']);
  // The mask marker is a tool like the stamp above it, with an icon of its own: not one more disc of colour.
  const [stamp, mask] = [stage.tool('Clone stamp'), stage.tool('Mask marker')];
  assert.deepEqual([mask.className, mask.children.length], [stamp.className, 1]);
  const [hint, glass] = stage.host.byClass('pt-row')[0].children;
  assert.equal(hint.className, 'pt-hint');
  assert.equal(glass.className, 'pt-bar');
  assert.equal(glass.byClass('mk-render').length, 0, 'the glass holds the brush\'s settings alone');
  const cover = stage.host.byClass('mk-cover')[0];
  const [ask] = cover.byClass('ot-ask');
  assert.equal(stage.tool(WIPE).hidden, true);
  stage.tool('Mask marker').click();
  assert.equal(cover.hidden, false, 'the black outside the square Render Masked sends');
  assert.equal(stage.tool(WIPE).hidden, false, 'the x beside the marker while it is up');
  await settle();
  assert.equal(ask.hidden, true, 'nothing masked: no box');
  // Masked, a box beside the mask holds "Render Masked" - off the model (the stand-in's is 30 .. 70 px across), on
  // the room to its left -; the x discards the mask, and the box goes with it.
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  await settle();
  assert.equal(ask.hidden, false);
  assert.ok(ask.byClass('mk-render').length === 1 && parseFloat(ask.style.left) < 30 - 16, ask.style.left);
  stage.tool(WIPE).click();
  assert.equal(ask.hidden, true, 'nothing masked any more: nothing to render');
  assert.equal(await viewer.current.userData.paint.readOverlay().then((read) => Math.max(...read.data)), 0);
  // Masked again, then the marker put down: the box goes (the side panel's "Render Masked" stands in).
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  await settle();
  assert.equal(ask.hidden, false);
  stage.tool('Orbit').click();
  assert.deepEqual([cover.hidden, ask.hidden, stage.tool(WIPE).hidden], [true, true, true]);
  OMNI.unmount();
});

test('the box beside the mask takes words of the artist\'s own, if any: they go with the mask, and Enter renders', async () => {
  const stage = await mount();
  stage.release();
  stage.tool('Mask marker').click();
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  await settle();
  const [ask] = stage.host.byClass('ot-ask');
  const [field] = ask.byClass('mk-ask-input');
  ask.byClass('mk-render')[0].click();
  await settle();
  assert.equal('words' in hub.job.params.retouch, false, 'none typed: none sent');
  store.jobs = [];                                   // (the repaint done)
  stage.update();
  await settle();
  stage.tool('Mask marker').click();
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  await settle();
  field.value = '  rusty bolts along the edge ';
  field.fire('input');
  field.fire('keydown', { key: 'Enter', shiftKey: false, isComposing: false });
  await settle();
  assert.equal(hub.job.params.retouch.words, 'rusty bolts along the edge');
  OMNI.unmount();
});

test('a mask stroke is never saved, and a stroke of the marker leaves the mask as it was', async () => {
  let stage = await mount();
  stage.release();
  stage.tool('Mask marker').click();
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  await settle();
  assert.equal(stage.host.byClass('ot-ask')[0].hidden, false, 'something masked: "Render Masked" beside it');
  OMNI.unmount();                                  // what is left to save goes as the view goes
  await settle();
  assert.deepEqual(changes(), [], 'nothing saved of the mask');

  stage = await mount();
  stage.release();
  stage.tool('Mask marker').click();
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  const session = viewer.current.userData.paint;
  const mask = (await session.readOverlay()).data;
  paintBase(stage, { x: 10, y: 10, w: 8, h: 8 });
  assert.deepEqual((await session.readOverlay()).data, mask, 'the mask as it was');
  stage.tool('Mask marker').click();
  await settle();
  assert.equal(stage.host.byClass('ot-ask')[0].hidden, false);
  OMNI.unmount();
  await settle();
  assert.deepEqual(changes(), [`PUT ${API}/omnitexture/paint`], 'the marker\'s stroke');
});

test('a picture taken off waits until the hand paint is in the files; the mask on the model comes back with it', async () => {
  const stage = await mount();
  stage.tool('Mask marker').click();
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  const mask = (await viewer.current.userData.paint.readOverlay()).data;
  paintBase(stage);
  const shownFirst = viewer.shown.at(-1);
  hub.answer = (path) => (path.endsWith('/remove') ? { prop: (store.project.props[0] = prop(MADE_AGAIN, [RESULT, REPAINT, MADE_AGAIN])) }
    : store.project.props[0] = prop());
  stage.host.byClass('ot-x')[0].click();
  await waitsForThePaint(stage, [`POST ${API}/omnitexture/remove`]);
  stage.update();                                   // the prop shows the model made again ...
  await settle();
  assert.notEqual(viewer.shown.at(-1), shownFirst);
  viewer.forget(shownFirst);                        // ... the viewer lets go of the one before ...
  keyDown('z', { ctrlKey: true });                  // ... and Ctrl+Z shows it again
  await settle();
  assert.deepEqual(changes().at(-1), `POST ${API}/omnitexture/step`);
  stage.update();
  await settle();
  assert.equal(viewer.shown.at(-1), shownFirst);
  assert.deepEqual((await viewer.current.userData.paint.readOverlay()).data, mask, 'its mask, stroke by stroke');
  OMNI.unmount();
});

test('a prop that was never OmniTextured has the tools on its 3D model: the first one taken up begins a result of its own texture, a model of its own that alone is painted', async () => {
  const stage = await mount([plain()]);
  const onScreen = viewer.shown.at(-1);
  assert.ok(onScreen.includes('mesh/mesh.glb'), onScreen);
  assert.equal(stage.host.byClass('pt-layer')[0].hidden, false, 'the tools are there, though the prop is not chosen');
  assert.deepEqual(changes(), [], 'looking at it asks the hub for nothing');
  const marker = stage.tool('Marker of any colour');
  hub.answer = (path) => (path.endsWith('/own') ? { prop: begun() } : { prop: store.project.props[0] });
  marker.click();
  assert.equal(marker.classList.contains('on'), false, 'not up before the model is the result\'s to paint');
  await settle();
  assert.deepEqual(changes(), [`POST ${API}/omnitexture/own`]);
  stage.update();                                   // the prop has its result now
  await settle();
  const step = viewer.shown.at(-1);
  assert.ok(step.includes(RESULT), 'the step is shown as its own model, never the 3D model\'s under another name');
  assert.equal(marker.classList.contains('on'), true, 'the tool asked for is up');
  assert.equal(stage.host.all((e) => e.tagName === 'BUTTON' && e.textContent === 'Painted').length, 1,
    'the switch names what the result is');
  // From here on it is painted and masked like any result: the hand paint goes into the new step.
  strokeAt(viewer, { x: 8, y: 8, w: 4, h: 4 });
  stage.release();
  stage.tool('Mask marker').click();
  strokeAt(viewer, { x: 20, y: 20, w: 6, h: 6 });
  stage.host.byClass('mk-render')[0].click();
  await settle();
  assert.deepEqual(changes().slice(1), [`PUT ${API}/omnitexture/paint`, `PUT ${API}/omnitexture/mask`, 'POST /api/jobs']);
  // "Original" shows the model the 3D stage made: nothing of the paint or of the mask is on it.
  stage.host.all((e) => e.tagName === 'BUTTON' && e.textContent === 'Original')[0].click();
  await settle();
  assert.equal(viewer.shown.at(-1), onScreen);
  assert.equal(viewer.current.userData.paint?.painted('base') ?? false, false);
  assert.equal(viewer.current.userData.paint?.painted('overlay') ?? false, false);
  OMNI.unmount();
});

test('a result of the prop\'s own texture on a 3D model made again since shows nowhere: not in the viewport, not among the painted views', async () => {
  const stale = () => ({ ...begun(), omnitexture: {
    ...begun().omnitexture, stale: true, steps: [RESULT, REPAINT],  glb: REPAINT,
    repaints: [{ glb: REPAINT, picture: 'omnitexture/ot_001/retouch_001/picture.png', under: RESULT, fade: 1 }] } });
  const stage = await mount([stale()]);
  assert.ok(viewer.shown.at(-1).includes('mesh/mesh.glb'), 'the 3D model the prop has now');
  assert.deepEqual(stage.host.byClass('ot-repaint'), [], 'no repaint of the old model to point at or take off');
  assert.deepEqual(stage.host.byClass('ot-x').filter((x) => !x.hidden), [], 'no × is offered');
  // The tools are there, on the 3D model: the first one taken up begins anew.
  hub.answer = (path) => (path.endsWith('/own') ? { prop: begun() } : { prop: store.project.props[0] });
  stage.tool('Marker of any colour').click();
  await settle();
  assert.deepEqual(changes(), [`POST ${API}/omnitexture/own`]);
  OMNI.unmount();
});

test('the hub refusing to begin the result puts the brush down and says why', async () => {
  const stage = await mount([plain()]);
  const said = [];
  const warn = console.warn;
  console.warn = (e) => said.push(e?.message || String(e));
  const real = globalThis.fetch;
  globalThis.fetch = async (path, init) => (String(path).endsWith('/own')
    ? { status: 409, text: async () => JSON.stringify({ detail: 'This prop is being OmniTextured. Wait until it is done.' }) }
    : real(path, init));
  try {
    const marker = stage.tool('Marker of any colour');
    marker.click();
    await settle();
    assert.equal(marker.classList.contains('on'), false);
    assert.equal(stage.tool('Orbit').classList.contains('on'), true);
    assert.deepEqual(said, ['This prop is being OmniTextured. Wait until it is done.']);
    marker.click();                                 // asked again: the hub is asked again
    await settle();
    assert.equal(said.length, 2);
  } finally {
    globalThis.fetch = real;
    console.warn = warn;
  }
  OMNI.unmount();
});

test('a tile\'s box chooses the prop for Omni-Texture; a click on the tile no longer does', async () => {
  await mount([plain()]);
  const box = () => OMNI.tileCheck(store.project.props[0]);
  assert.deepEqual(box(), { on: false, title: 'Choose it for Omni-Texture' });
  assert.equal(OMNI.pickTile, undefined, 'a click on the tile only opens the prop');
  assert.equal(OMNI.tileMark, undefined, 'no golden outline');
  // A tick rings the button it is for, once ("Omni-Texture the chosen"); taking the tick away rings nothing.
  const rung = [];
  const { animate } = FakeEl.prototype;
  FakeEl.prototype.animate = function (frames, timing) { rung.push([this.classList.contains('ot-go'), timing?.iterations]); return animate.call(this); };
  OMNI.checkTile('p_1');
  assert.equal(store.project.props[0].omnitexture.settings.chosen, true);
  assert.deepEqual(box(), { on: true, title: 'Chosen for Omni-Texture: click to leave it out' });
  assert.deepEqual(rung, [[true, 1]]);
  OMNI.checkTile('p_1');
  assert.equal(box().on, false);
  assert.equal(rung.length, 1);
  FakeEl.prototype.animate = animate;
  // A prop whose Omni-Texture is being made stays chosen; Omni-Texture that can't run here offers no box.
  OMNI.checkTile('p_1');
  store.jobs = [hubJob({ id: 'j_1', stage: 'omnitexture', status: 'running', prop_ids: ['p_1'], progress: { fraction: 0.2, message: '' }, errors: {} })];
  OMNI.checkTile('p_1');
  assert.equal(box().on, true);
  store.jobs = [];
  // A tileable texture has no 3D model to paint: no box for it.
  const [object] = store.project.props;
  assert.equal(OMNI.tileCheck({ ...object, images: { ...object.images, settings: { tileable: true } } }), null);
  assert.notEqual(OMNI.tileCheck(object), null);
  store.status.stage_tools.omnitexture = [];
  assert.equal(OMNI.tileCheck(store.project.props[0]), null);
  await new Promise((resolve) => realTimeout(resolve, 650));
  assert.deepEqual(changes(), [`PATCH ${API}`], 'saved like any setting, a moment later');
  OMNI.unmount();
});

test('Alt + click on a painted view or on a repaint shows what Qwen-Image was given; a plain click on a repaint picks it for the fade, on a view does nothing', async () => {
  const stage = await mount();
  const asked = () => hub.calls.filter((c) => c.includes('request.json'));
  const windows = () => document.body.all((e) => e.classList.contains('ot-inputs')).length;
  const [view] = stage.host.byClass('ot-view');
  const [repaint] = stage.host.byClass('ot-repaint');
  assert.ok(view.title.endsWith('Alt + click: what Qwen-Image was given for it.') && repaint.title.endsWith('Alt + click: what Qwen-Image was given for it.'));
  view.fire('click', { altKey: false });
  await settle();
  assert.deepEqual([asked(), windows()], [[], 0], 'a plain click on a view opens nothing');
  repaint.fire('click', { altKey: false });
  await settle();
  assert.deepEqual([asked(), repaint.classList.contains('picked')], [[], true], 'a plain click on a repaint picks it for the edge fade');
  const real = globalThis.fetch;
  const record = { prompt: 'restore the image', pictures: ['picture 1: 1-viewport.png', 'picture 2: 2-depth.png', 'picture 3: 3-example.png'],
    seed: 7, steps: 8, strength: 0.9, model: 'turbo',
    retouch: { example: 'front', tries: [{ seed: 6, left: 30 }, { seed: 7, left: 2 }] } };
  globalThis.fetch = async (path, init = {}) => {
    hub.calls.push(`${init.method || 'GET'} ${String(path).split('?')[0]}`);
    return String(path).includes('request.json') ? { ok: true, status: 200, json: async () => record } : { ok: true, status: 200 };
  };
  try {
    repaint.fire('click', { altKey: true });
    await settle();
    assert.deepEqual(asked(), ['GET /files/Props/props/p_1/omnitexture/ot_001/retouch_001/request.json'], 'the repaint\'s own folder');
    assert.equal(repaint.classList.contains('picked'), true, 'the Alt + click did not unpick it');
    const [shown] = document.body.all((e) => e.classList.contains('ot-inputs'));
    const said = shown.all((e) => e.tagName === 'FIGCAPTION').map((e) => e.textContent);
    assert.deepEqual(said, [
      'Picture 1, also what it starts from: your view of the model, what you masked in magenta',
      'Picture 2: the depth map from there',
      'Picture 3, the example: the prop’s own picture of its front',
      'Qwen-Image’s painting', 'Its lighting taken off', 'What went onto the model']);
    assert.ok(shown.all((e) => e.classList.contains('hint')).some((e) => e.textContent.includes('the second picture')), 'it says a picture was made again');
    view.fire('click', { altKey: true });
    await settle();
    assert.equal(asked().length, 2, 'a painted view\'s record too');
    // Masked from behind the side its picture shows: the picture went mirrored, and the caption says so. Older
    // records: the example was the model from the nearest camera around it, and before that your own view.
    for (const [retouch, said] of [[{ example: 'front', mirrored: true }, 'the prop’s own picture of its front, mirrored: you look from behind it'],
      [{ example: 'front', mirrored: false }, 'the prop’s own picture of its front'],
      [{ example_view: 'front left' }, 'the model from the front left, the nearest camera around it'],
      [{}, 'the model without the mask']]) {
      record.retouch = { ...retouch, tries: [] };
      repaint.fire('click', { altKey: true });
      await settle();
      const older = document.body.all((e) => e.classList.contains('ot-inputs')).at(-1);
      assert.ok(older.all((e) => e.tagName === 'FIGCAPTION').some((e) => e.textContent === `Picture 3, the example: ${said}`), said);
    }
    // A repaint of today is drawn anew where the view is masked and held to the view elsewhere (its record's
    // "redrawn"): picture 1 is worded so, and what was drawn anew is shown after the pictures it saw.
    record.redrawn = 'redrawn.png';
    repaint.fire('click', { altKey: true });
    await settle();
    const held = document.body.all((e) => e.classList.contains('ot-inputs')).at(-1)
      .all((e) => e.tagName === 'FIGCAPTION').map((e) => e.textContent);
    assert.equal(held[0], 'Picture 1, also what it is held to outside the mask: your view of the model, what you masked in magenta');
    assert.equal(held[3], 'Not a picture it sees: what it draws anew (white); everywhere else its painting is held to picture 1');
    delete record.redrawn;
    // A painted view of today: the depth map, then the prop's own picture of its front as the example. A sheet
    // starts from the model with its own texture, which is no picture Qwen-Image sees: listed after them.
    Object.assign(record, { pictures: ['picture 1: 1-render.png', 'picture 2: prop: mesh/views/0_front.png'],
      shown: 'depth', example: 'front', start_image: '1-render.png', painted_where: '2-mask.png' });
    const captions = async () => {
      view.fire('click', { altKey: true });
      await settle();
      return document.body.all((e) => e.classList.contains('ot-inputs')).at(-1)
        .all((e) => e.tagName === 'FIGCAPTION').map((e) => e.textContent);
    };
    let told = await captions();
    assert.ok(told[0].startsWith('Picture 1, also what it starts from: the depth map'));
    assert.equal(told[1], 'Picture 2, the example: the prop’s own picture of its front');
    assert.ok(!told.some((text) => text.startsWith('Not a picture it sees')));
    record.start_image = 'model.png';
    record.ref_sides = [1152, 512];           // a sheet: how large Qwen-Image's own copy of each picture is
    told = await captions();
    assert.ok(told[0].startsWith('Picture 1: the depth map') && told[0].endsWith('(Qwen-Image sees it at 1152 px)'));
    assert.equal(told[1], 'Picture 2, the example: the prop’s own picture of its front (Qwen-Image sees it at 512 px)');
    assert.ok(told[2].startsWith('Not a picture it sees: what the painting starts from, one step in'));
  } finally {
    globalThis.fetch = real;
  }
  OMNI.unmount();
});

test('the eye in a picture\'s middle hides it on the model and shows it again: a repaint alone, the painted views all together; Alt + click there is the picture\'s own', async () => {
  const stage = await mount();
  stage.release();
  const eyes = () => stage.host.byClass('ot-eye');
  const off = () => stage.host.byClass('ot-tile').map((tile) => tile.classList.contains('off'));
  const shown = (omnitexture) => ({ prop: (store.project.props[0] = { ...prop(), omnitexture: { ...prop().omnitexture, ...omnitexture } }),
    renamed: { [REPAINT]: MADE_AGAIN } });
  const repaint = (hidden) => ({ glb: MADE_AGAIN, picture: 'omnitexture/ot_001/remove_001/picture.png', under: RESULT, fade: 1, hidden });
  assert.deepEqual(off(), [false, false, false], 'two views and a repaint, all on the model');
  assert.ok(eyes().slice(0, 2).every((eye) => stage.label(eye).startsWith('Hide the painted views - all of them')));
  assert.equal(stage.label(eyes()[2]), 'Hide this repaint: the model shows what is under it');

  // The repaint's eye: it alone is asked hidden; its tile stays, dim under the crossed eye, and is no longer picked.
  hub.answer = () => shown({ glb: MADE_AGAIN, steps: [RESULT, MADE_AGAIN], repaints: [repaint(true)] });
  eyes()[2].fire('click', { altKey: false });
  await settle();
  assert.deepEqual([changes(), hub.asked], [[`POST ${API}/omnitexture/hide`], { glb: REPAINT, repaint: REPAINT, hidden: true }]);
  stage.update();
  await settle();
  assert.deepEqual(off(), [false, false, true]);
  assert.equal(stage.label(eyes()[2]), 'Show this repaint on the model again');
  const [picture] = stage.host.byClass('ot-repaint');
  picture.fire('click', { altKey: false });
  assert.equal(picture.classList.contains('picked'), false, 'no edge to fade while it is hidden');
  assert.ok(picture.title.includes('hidden: it is not on the model'));
  hub.answer = () => shown({ glb: MADE_AGAIN, steps: [RESULT, MADE_AGAIN], repaints: [repaint(false)] });
  eyes()[2].fire('click', { altKey: false });
  await settle();
  assert.deepEqual(hub.asked, { glb: MADE_AGAIN, repaint: MADE_AGAIN, hidden: false }, 'shown again');

  // A view's eye: all of them, whichever is clicked - and no view is taken off while they are hidden.
  const base = { glb: MADE_AGAIN, steps: [MADE_AGAIN], repaints: [] };
  const views = prop().omnitexture.views.map((view) => ({ folder: view.slice(0, view.lastIndexOf('/')), picture: view, layer: null }));
  hub.answer = () => shown({ ...base, bases: [{ glb: MADE_AGAIN, views, hidden: true, shown: RESULT }] });
  eyes()[1].fire('click', { altKey: false });
  await settle();
  assert.deepEqual(hub.asked, { glb: MADE_AGAIN, views: true, hidden: true });
  stage.update();
  await settle();
  assert.deepEqual(off(), [true, true]);
  assert.ok(eyes().every((eye) => stage.label(eye) === 'Show the painted views on the model again'));
  assert.deepEqual(stage.host.byClass('ot-tile').flatMap((tile) => tile.byClass('ot-x')), [], 'no × on a hidden view');
  eyes()[0].fire('click', { altKey: false });
  await settle();
  assert.deepEqual(hub.asked, { glb: MADE_AGAIN, views: true, hidden: false });

  // Alt + click on an eye hides nothing: it shows what Qwen-Image was given, as on the picture around it.
  const asked = changes().length;
  const real = globalThis.fetch;
  globalThis.fetch = async (path) => {
    hub.calls.push(`GET ${String(path).split('?')[0]}`);
    return { ok: true, status: 200, json: async () => ({ prompt: '', pictures: [] }) };
  };
  try {
    eyes()[0].fire('click', { altKey: true });
    await settle();
  } finally {
    globalThis.fetch = real;
  }
  assert.equal(changes().length, asked, 'nothing asked of the hub');
  assert.ok(hub.calls.at(-1).endsWith('/omnitexture/ot_001/views/0-front/request.json'));
  OMNI.unmount();
});

test('a picture\'s share shows on the model while the pointer is anywhere on its tile, its eye and its × too', async () => {
  const stage = await mount([{ ...prop(), omnitexture: { ...prop().omnitexture,
    layers: ['omnitexture/ot_001/views/0-front/layer.webp'] } }]);
  const [front, back] = stage.host.byClass('ot-tile');
  const layers = [];
  const setLayer = viewer.setLayer;
  viewer.setLayer = (url) => layers.push(url && url.split('?')[0]);
  try {
    front.fire('pointerenter');
    front.fire('pointerleave');
    back.fire('pointerenter');            // (a view without a layer of its own: nothing to show)
    back.fire('pointerleave');
  } finally {
    viewer.setLayer = setLayer;
  }
  assert.deepEqual(layers, ['/files/Props/props/p_1/omnitexture/ot_001/views/0-front/layer.webp', null]);
  OMNI.unmount();
});

test('a fade waits until the hand paint is in the files', async () => {
  const stage = await mount();
  paintBase(stage);
  stage.host.byClass('ot-repaint')[0].click();      // the repaint picked for the slider
  const range = stage.host.byClass('ot-fade-range')[0];
  range.value = 2;
  range.fire('change');
  await waitsForThePaint(stage, [`POST ${API}/omnitexture/fade`]);
  OMNI.unmount();
});

test('"Paint it again" and "Omni-Texture the chosen" wait until the hand paint is in the files', async () => {
  let stage = await mount();
  paintBase(stage);
  stage.host.all((e) => e.tagName === 'BUTTON' && e.textContent === 'Paint it again')[0].click();
  await waitsForThePaint(stage, ['POST /api/jobs']);
  OMNI.unmount();

  stage = await mount([prop(), waiting()]);
  paintBase(stage);
  stage.host.byClass('ot-go')[0].click();
  await waitsForThePaint(stage, ['POST /api/jobs']);
  OMNI.unmount();
});

test('"Render Masked" sends the mask once the hand paint is in the files', async () => {
  const stage = await mount();
  stage.tool('Mask marker').click();
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  paintBase(stage);
  stage.host.byClass('mk-render')[0].click();
  await waitsForThePaint(stage, [`PUT ${API}/omnitexture/mask`, 'POST /api/jobs']);
  assert.equal(stage.host.byClass('mk-cover')[0].hidden, true, 'the mask marker put down');
  OMNI.unmount();
});

test('the × beside "Painted views" takes the whole Omni-Texture off: the prop shows its 3D model, Ctrl+Z brings the result back and Ctrl+Y takes it off again', async () => {
  const stage = await mount();
  stage.release();
  const steps = [RESULT, REPAINT, ''];
  const whole = stage.host.all((e) => stage.label(e) === 'Take the whole Omni-Texture off this model')[0];
  assert.equal(whole.hidden, false, 'beside the heading of the painted views');
  hub.answer = (path, body) => (path.endsWith('/remove') ? { prop: (store.project.props[0] = prop('', steps)) }
    : (store.project.props[0] = prop(body.glb, steps)));
  whole.click();
  await settle();
  assert.deepEqual([changes(), hub.asked], [[`POST ${API}/omnitexture/remove`], { glb: REPAINT, whole: true }]);
  stage.update();
  await settle();
  assert.ok(viewer.shown.at(-1).includes('mesh/mesh.glb'), 'the 3D model with its own texture');
  assert.deepEqual([stage.host.byClass('ot-view').length, whole.hidden], [0, true], 'nothing painted is listed any more');
  keyDown('z', { ctrlKey: true });
  await settle();
  assert.deepEqual([changes().at(-1), hub.asked], [`POST ${API}/omnitexture/step`, { glb: REPAINT }]);
  stage.update();
  await settle();
  assert.ok(viewer.shown.at(-1).includes(REPAINT), 'the result again, as it was');
  assert.equal(whole.hidden, false);
  // Taken off once more, after a change elsewhere on the page: it is the newest change again, Ctrl+Z's first.
  const elsewhere = undoStamp();
  const forget = undoList(() => elsewhere);
  whole.click();
  await settle();
  stage.update();
  await settle();
  assert.ok(viewer.shown.at(-1).includes('mesh/mesh.glb'));
  keyDown('z', { ctrlKey: true });
  await settle();
  assert.deepEqual([changes().at(-1), hub.asked], [`POST ${API}/omnitexture/step`, { glb: REPAINT }]);
  forget();
  stage.update();
  await settle();
  keyDown('y', { ctrlKey: true });
  await settle();
  assert.deepEqual([changes().at(-1), hub.asked], [`POST ${API}/omnitexture/step`, { glb: '' }]);
  OMNI.unmount();
});

test('the view moves while the mask marker is up as with any tool; the box leaves while it does and takes its place again once the view rests', async () => {
  const stage = await mount();
  stage.release();
  stage.tool('Mask marker').click();
  const [first, keys] = stage.host.byClass('pt-hint')[0].children.map((line) => line.textContent);
  assert.ok(first.startsWith('Mask marker: paint over what to make again') && keys.includes('orbit'), `${first} / ${keys}`);
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  await settle();
  const [ask] = stage.host.byClass('ot-ask');
  assert.equal(ask.hidden, false);
  // The artist turns the model (Alt + drag, the wheel: the viewer's controls say so as they go).
  const sights = viewer.renderer.reads;
  lookFrom(SIDE);
  viewer.controls.dispatchEvent(new Event('change'));
  assert.equal(ask.hidden, true, 'out of the way while the view moves');
  await settle();
  assert.equal(ask.hidden, true, 'still moving, for all the page knows');
  viewer.controls.dispatchEvent(new Event('change'));
  await new Promise((resolve) => realTimeout(resolve, 260));
  await settle();
  assert.equal(ask.hidden, false, 'at rest: back, where the model and the mask now leave room');
  assert.equal(viewer.renderer.reads, sights + 1, 'looked at once, when the view came to rest');
  // The lens bent while masking moves the view too.
  stage.tool('Inverse perspective').click();
  assert.equal(ask.hidden, true);
  await new Promise((resolve) => realTimeout(resolve, 260));
  await settle();
  assert.equal(ask.hidden, false);
  stage.tool('Inverse perspective').click();
  // With the marker down no box follows the view.
  stage.tool('Orbit').click();
  viewer.controls.dispatchEvent(new Event('change'));
  await new Promise((resolve) => realTimeout(resolve, 260));
  assert.equal(ask.hidden, true);
  OMNI.unmount();
});

test('a tool taken up in the lit view shows the base colour; "Render Masked" puts the marker down', async () => {
  const stage = await mount();
  stage.release();
  assert.equal(viewer.view, 'all');
  stage.tool('Mask marker').click();
  assert.equal(viewer.view, 'basecolor', 'the mask marker too: what is painted over shows flat');
  // A job takes the prop (one started elsewhere): the tools go until it is through.
  store.jobs = [hubJob({ id: 'j_2', stage: 'omnitexture', status: 'running', prop_ids: ['p_1'], progress: { fraction: 0.2, message: '' }, errors: {} })];
  stage.update();
  store.jobs = [];
  stage.update();
  await settle();
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  stage.host.byClass('mk-render')[0].click();
  await settle();
  assert.equal(stage.tool('Mask marker').classList.contains('on'), false);
  assert.equal(viewer.view, 'basecolor', 'the view stays where the tool left it');
  OMNI.unmount();
});

test('"No Facets" in the box, left of "Render Masked": on, the repaint asks for a depth map without the model\'s facets, and stays on for the next', async () => {
  const stage = await mount();
  stage.release();
  stage.tool('Mask marker').click();
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  await settle();
  const [ask] = stage.host.byClass('ot-ask');
  const [toggle, render] = ask.byClass('mk-ask-row')[0].children;
  assert.deepEqual([toggle.textContent, toggle.getAttribute('aria-pressed'), render.textContent], ['No Facets', 'false', 'Render Masked']);
  assert.equal(ask.byClass('mk-ask-input')[0].getAttribute('rows'), '3');
  render.click();
  await settle();
  assert.equal('smooth' in hub.job.params.retouch, false, 'off: the depth map as the model is');
  store.jobs = [];
  stage.update();
  await settle();
  stage.tool('Mask marker').click();
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  await settle();
  toggle.click();
  assert.equal(toggle.getAttribute('aria-pressed'), 'true');
  render.click();
  await settle();
  assert.equal(hub.job.params.retouch.smooth, true);
  // The artist's setting, as the lens is: the next view of the stage has it on still.
  OMNI.unmount();
  const again = await mount();
  const next = again.host.byClass('ot-ask')[0].byClass('mk-ask-row')[0].children[0];
  assert.equal(next.getAttribute('aria-pressed'), 'true');
  next.click();                                      // (off again for the tests that follow)
  OMNI.unmount();
});

test('"Inverse perspective" under the mask marker bends the view, its slider says how far, "Render Masked" sends that lens, and it goes with the view', async () => {
  const stage = await mount();
  stage.release();
  const bend = stage.tool('Inverse perspective');
  const [glass] = stage.host.byClass('mk-lens');
  const [slider] = glass.children;
  assert.deepEqual([viewer.camera.perspective, bend.getAttribute('aria-pressed'), glass.hidden], [1, 'false', true]);
  bend.click();
  assert.deepEqual([viewer.camera.perspective, bend.getAttribute('aria-pressed'), glass.hidden], [-1, 'true', false],
    'half way along its slider: as inverse as a camera is perspective');
  slider.value = 25;
  slider.fire('input');
  assert.equal(viewer.camera.perspective, 0, 'a quarter of the way: no perspective left');
  slider.value = 100;
  slider.fire('input');
  assert.equal(viewer.camera.perspective, -3);
  // The lens stays the artist's while the mask marker is up.
  stage.tool('Mask marker').click();
  assert.deepEqual([bend.disabled, slider.disabled], [false, false]);
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  stage.host.byClass('mk-render')[0].click();
  await settle();
  const { tan, wide } = hub.job.params.retouch;
  const half = Math.tan((viewer.camera.fov * Math.PI) / 360);
  assert.ok(Math.abs(tan + 3 * half) < 1e-12 && Math.abs(wide - half * 2 * 4) < 1e-12, 'the bent lens goes with the camera');
  store.jobs = [];                                   // (the repaint done)
  stage.update();
  await settle();
  assert.equal(bend.disabled, false, 'the repaint done: the lens is the artist\'s again');
  // Off again: a camera's own view.
  bend.click();
  assert.deepEqual([viewer.camera.perspective, glass.hidden], [1, true]);
  bend.click();
  assert.equal(viewer.camera.perspective, -3, 'on again: as far as it was');
  // Another prop's view, or another stage's, starts from the usual view; the setting stays for the next visit here.
  OMNI.unmount();
  assert.equal(viewer.camera.perspective, 1);
  const again = await mount();
  assert.equal(viewer.camera.perspective, -3, 'back in the stage: the lens as the artist left it');
  again.tool('Inverse perspective').click();
  OMNI.unmount();
});

test('"Render Masked" sends the view as it is at the press, wherever the mask was painted from; taking the marker up again leaves the view where it is', async () => {
  const stage = await mount();
  stage.release();
  const mask = stage.tool('Mask marker');
  mask.click();
  strokeAt(viewer, { x: 8, y: 8, w: 6, h: 6 });
  stage.tool('Orbit').click();
  lookFrom(SIDE);                                    // the marker down, the artist goes round the model
  mask.click();
  assert.deepEqual(lookingFrom(), SIDE, 'the marker up again: the view stays where it is');
  stage.tool('Orbit').click();
  stage.host.byClass('mk-render')[0].click();
  await settle();
  assert.deepEqual(hub.job.params.retouch.eye, SIDE, 'the camera Qwen-Image paints for is the one at the press');
  assert.deepEqual(lookingFrom(), SIDE);
  OMNI.unmount();
});
