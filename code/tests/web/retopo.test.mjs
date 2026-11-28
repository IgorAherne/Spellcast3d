// The Retopo stage on a fake page, with a stand-in viewer (standin_viewer3d.mjs): the tool gates per method, the
// sizes a prop asks for (the boxes on its tile), "Retopo All" in the viewport's corner (one job per method; going
// down to Retopo starts none), the clean meshes at the bottom of the viewport and the pick, the call to action held
// back until every prop has a picked clean mesh, the tool loaded as the stage opens, Instant Meshes' viewer a click
// away, a prop's status for its method, the clean meshes each method keeps, TriFlow's terms (the chip, the export
// note), and the tool lists of Settings > Diagnostics and the install card (the status line's words:
// shell_logic.test.mjs).
//   node --test code/tests/web/retopo.test.mjs
// (code/tests/test_web.py runs every code/tests/web/*.test.mjs when Node.js is installed)
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { test } from 'node:test';

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

  append(...nodes) { for (const n of nodes.flat()) this.children.push(this.adopt(n)); }
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
  get lastChild() { return this.children.at(-1) ?? null; }
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
  matches() { return false; }

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
globalThis.Node = FakeNode;
globalThis.window = globalThis;
globalThis.addEventListener ??= () => {};
globalThis.removeEventListener ??= () => {};
globalThis.location = { href: 'http://127.0.0.1:7770/', origin: 'http://127.0.0.1:7770', hostname: '127.0.0.1' };
globalThis.document = {
  addEventListener: () => {}, removeEventListener: () => {},
  hidden: false, activeElement: null, documentElement: { dataset: {} }, body: new FakeEl('body'),
  querySelector: () => null,
  getElementById: () => new FakeEl('div'),
  createElement: (tag) => (tag === 'template' ? new FakeTemplate(tag) : new FakeEl(tag)),
  createElementNS: (ns, tag) => new FakeEl(tag),
};
// The page's polls and toasts wait a second or more: here they never come.
const realTimeout = globalThis.setTimeout;
globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 1000 ? 0 : realTimeout(fn, ms, ...rest));

// viewer3d.js is the stand-in viewer here; what each model was shown with is kept.
const VIEWER = new URL('./standin_viewer3d.mjs', import.meta.url).href;
registerHooks({
  resolve: (specifier, context, next) => {
    const found = next(specifier, context);
    return found.url.endsWith('/static/js/viewer3d.js') ? { url: VIEWER, shortCircuit: true } : found;
  },
});
const { viewer } = await import(VIEWER);
const shownWith = [];
const showModel = viewer.show.bind(viewer);
viewer.show = (url, kind, opts = {}) => {
  shownWith.push({ url, keepView: !!opts.keepView });
  return showModel(url, kind, opts);
};

const STATIC = new URL('../../spellcast/static/js/', import.meta.url);
const load = (path) => import(new URL(path, STATIC));

const { on, store, TOOL_NAMES, TOOL_ROLES, STAGE_TOOLS, stageTool } = await load('store.js');
const { flushSettings } = await load('actions.js');
const { carryDown, jobState, liveDeps, toolGate } = await load('stages/common.js');
const {
  RETOPO_KEEP_VERTICES, RETOPO_SIZES, RETOPO_VERTICES, retopoFresh, retopoSizeName, retopoSwitched, retopoVertices,
} = await load('stages/rules.js');
const { licenseOf, nonCommercial } = await load('licenses.js');
const { default: RETOPO, methodFollowers, pickMethod, settingsChanged, waitingHint } = await load('stages/retopo.js');
const { default: EXPORT } = await load('stages/rig.js');
const { placementOrNone } = await load('stages/placebar.js');
const { toastSaved } = await load('stages/preview.js');
const { TOOLS: DIAGNOSTICS } = await load('shell/settings.js');
const { NAMES: INSTALL_NAMES, ORDER: INSTALL_ORDER } = await load('shell/install.js');
const { boxNumber, boxNumbers, boxText } = await load('shell/proplist.js');
const { hubJob } = await import('./hub_job.mjs');

const IDS = ['qwen', 'forge', 'pixal3d', 'imesh', 'lato2', 'triflow'];
const tools = (extra = {}) => Object.fromEntries(IDS.map((t) => [t, { state: 'stopped', installed: true, supported: true, ...extra[t] }]));

/** A prop with a 3D model, set to `method` (unset: none picked, so TriFlow, the default) and `sizes`. */
const prop = (id, method, extra = {}, sizes = [2]) => ({
  id, name: id, images: { chosen: 'images/0001.png' }, mesh: { glb: 'mesh/mesh.glb', source: 'images/0001.png' },
  retopo: { obj: null, source: null, method: null, variants: [], settings: { ...(method ? { method } : {}), sizes } },
  projection: { glb: null }, ...extra,
});

/** Its clean meshes, made with `madeWith` from its 3D model: {size: vertices}; `picked`: the size it uses, picked. */
function made(p, madeWith, meshes, { picked = null, ...more } = {}) {
  const file = madeWith === 'imesh' ? 'retopo' : madeWith;
  const variants = Object.entries(meshes).map(([size, vertices]) => ({
    size: Number(size), obj: `retopo/${size}k/${file}.obj`, stats: { vertices, faces: 2 * vertices, has_uv: true },
  }));
  const used = variants.find((v) => v.size === (picked ?? p.retopo.settings.sizes.find((s) => meshes[s]))) || variants[0];
  p.retopo = { ...p.retopo, obj: used.obj, stats: used.stats, method: madeWith, source: 'mesh/mesh.glb', stale: false,
    accepted: picked !== null, variants, ...more };
  return p;
}

function reset(props, status = { tools: tools() }) {
  store.connected = true;
  store.jobs = [];
  store.jobErrors = {};
  store.status = status;
  store.project = { id: 'Props', name: 'Props', props, ui: {} };
  store.bust = {};
}

// What the page asked of the hub (its polls left out), and how the hub answers: a job queued, a clean mesh picked.
const hub = { calls: [], bodies: [], saved: null };      // saved(prop): the prop as the hub has it after a save
const answer = (body) => ({ status: 200, text: async () => JSON.stringify(body) });
globalThis.fetch = async (path, init = {}) => {
  const method = init.method || 'GET';
  const body = typeof init.body === 'string' ? JSON.parse(init.body) : null;
  if (method === 'GET') return answer(path.startsWith('/api/jobs') ? store.jobs : {});
  hub.calls.push(`${method} ${path}`);
  hub.bodies.push(body);
  if (path === '/api/jobs') return answer(hubJob({ id: `j${hub.calls.length}`, status: 'queued', progress: {}, ...body }));
  const pick = /\/props\/([^/]+)\/retopo\/pick$/.exec(path);
  if (pick) {
    const p = structuredClone(store.project.props.find((x) => x.id === pick[1]));
    const variant = p.retopo.variants.find((v) => v.size === body.size);
    Object.assign(p.retopo, { obj: variant.obj, stats: variant.stats, accepted: true });
    return answer(p);
  }
  if (path === '/api/imesh/session') return answer({ session_id: 's1', viewer_url: 'http://127.0.0.1:7776/viewer?session=s1' });
  const patched = /\/props\/([^/]+)$/.exec(path);
  const kept = patched ? store.project.props.find((x) => x.id === patched[1]) : {};
  return answer(patched && hub.saved ? hub.saved(structuredClone(kept)) : kept);
};
const settle = async () => { for (let i = 0; i < 8; i++) await new Promise((resolve) => realTimeout(resolve, 0)); };

let nudged = 0;
/** The stage on screen with `props`, the first selected (the shell's context: refresh draws it again). */
async function mount(props, selectedId = props[0]?.id ?? null) {
  RETOPO.unmount();
  reset(props, { tools: tools(), install: { components: {} }, gpu: { vram_total_mb: 12288 } });
  hub.calls = [];
  hub.bodies = [];
  shownWith.length = 0;
  const host = new FakeEl('div');
  const ctx = {
    selectedId, stageId: 'retopo', project: store.project, select() {}, goto() {}, moving: false,
    settled: Promise.resolve(), nudgeCta: () => { nudged += 1; }, refresh: async () => RETOPO.update(ctx),
  };
  RETOPO.mount(host, ctx);
  await settle();
  const variants = () => host.byClass('rt-variant');
  return { host, ctx, variants, go: () => host.byClass('rt-go')[0], update: () => RETOPO.update(ctx) };
}

// ---- the tools ---------------------------------------------------------------------------------------------

test('each retopology method has its own gate: one that can\'t run here leaves only its props out', () => {
  reset([], { tools: tools({ triflow: { supported: false, state: 'unsupported' } }), install: { components: {} } });
  assert.equal(toolGate('retopo', 'triflow').kind, 'unsupported');
  assert.equal(toolGate('retopo', 'triflow').text, "TriFlow can't run on this computer.");
  assert.equal(toolGate('retopo', 'lato2').kind, 'ok');
  assert.equal(toolGate('retopo').kind, 'ok', 'the stage\'s default tool: Instant Meshes');
  assert.equal(liveDeps().open('retopo', 'triflow'), false);
  assert.equal(liveDeps().open('retopo', 'lato2'), true);
  assert.equal(liveDeps().open('mesh'), true);
  // Installing: jobs wait for it, and the words name the tool; the default tool keeps its role's name.
  reset([], {
    tools: tools({ lato2: { state: 'installing', installed: false }, imesh: { state: 'installing', installed: false } }),
    install: { components: { lato2: { status: 'running', fraction: 0.45 }, imesh: { status: 'running', fraction: 0.1 } } },
  });
  assert.deepEqual([toolGate('retopo', 'lato2').kind, toolGate('retopo', 'lato2').text], ['installing', 'LATO.2 is installing… 45%']);
  assert.equal(toolGate('retopo', 'imesh').text, 'The retopology tool is installing… 10%');
  assert.equal(liveDeps().open('retopo', 'lato2'), true);
  // A hub that does not know the tool yet: the page does not stand in its way.
  reset([], { tools: { imesh: { state: 'ready' } } });
  assert.equal(toolGate('retopo', 'lato2').kind, 'unknown');
});

test('the tool of a prop\'s stage work: its method in Retopo, the stage\'s own elsewhere', () => {
  assert.equal(stageTool('retopo', prop('a', 'triflow')), 'triflow');
  assert.equal(stageTool('retopo', prop('a', 'imesh')), 'imesh');
  assert.equal(stageTool('retopo', prop('a')), 'triflow', 'no method picked: TriFlow, the default');
  assert.equal(stageTool('retopo'), 'imesh');
  assert.equal(stageTool('mesh', prop('a', 'triflow')), 'pixal3d');
  assert.deepEqual(STAGE_TOOLS.retopo, ['triflow', 'lato2', 'imesh'], 'the default first');
});

// ---- the sizes on each tile ---------------------------------------------------------------------------

test('the boxes on a prop\'s tile hold its sizes; what is typed there is saved, each size once, three at most', async () => {
  reset([prop('a', 'lato2', {}, [4, 22, 6])]);
  const offer = RETOPO.tileSizes(store.project.props[0]);
  assert.deepEqual(offer, { values: [4, 22, 6], count: 3, min: 0.1, max: 99, decimals: 2, unit: 'k verts', title: offer.title });
  assert.deepEqual(offer, { values: [4, 22, 6], ...RETOPO_SIZES, unit: 'k verts', title: offer.title });
  assert.match(offer.title, /\(0\.25 is 250\)/, 'it says a size can be under a thousand');
  assert.deepEqual(boxNumbers(['4', '', '07'], RETOPO_SIZES), [4, 7], 'an empty box says nothing');
  assert.deepEqual(boxNumbers(['0', '100', '99', '0,25'], RETOPO_SIZES), [99, 0.25]);
  hub.calls = [];
  RETOPO.sizeTile('a', [4, 4, 9]);
  assert.deepEqual(store.project.props[0].retopo.settings.sizes, [4, 9], 'at once, in the page');
  await flushSettings();
  assert.deepEqual([hub.calls, hub.bodies.at(-1)], [['PATCH /api/projects/Props/props/a'], { retopo: { settings: { sizes: [4, 9] } } }]);
  RETOPO.sizeTile('a', boxNumbers(['0,3', '.3', '2'], RETOPO_SIZES));
  assert.deepEqual(store.project.props[0].retopo.settings.sizes, [0.3, 2], '"0,3" and ".3" are the same size');
  await flushSettings();
  assert.deepEqual(hub.bodies.at(-1), { retopo: { settings: { sizes: [0.3, 2] } } });
  RETOPO.sizeTile('a', []);
  assert.deepEqual(store.project.props[0].retopo.settings.sizes, [], 'every box erased');
  await flushSettings();
});

test('a size is typed with up to two decimals after a dot or a comma: ".3", ",3", "0.3" and "0,3" are 300 vertices', () => {
  const kept = (text) => boxText(text, RETOPO_SIZES);
  const typed = ['.3', ',3', '0.3', '0,3', '.25', ',25', '0.25', '0,25', '12.25', '99', '7'];
  assert.deepEqual(typed.map(kept), typed, 'kept as typed');
  assert.deepEqual(['123', '123.4', '0.255', '1.2.3', '1,2.3', '..5', 'a.5', '2 k', '-1', '1e3', '½'].map(kept),
    ['12', '12.4', '0.25', '1.23', '1,23', '.5', '.5', '2', '1', '13', ''], 'what else is typed is dropped');
  assert.deepEqual(typed.map((text) => boxNumber(text, RETOPO_SIZES)), [0.3, 0.3, 0.3, 0.3, 0.25, 0.25, 0.25, 0.25, 12.25, 99, 7]);
  assert.deepEqual(['', '0', '0.', '.', ',', '00', '0.05', '0.09', '99.5', '100'].map((text) => boxNumber(text, RETOPO_SIZES)),
    Array(10).fill(null), 'not (yet) a size from 0.1 to 99: as if the box were empty');
  assert.deepEqual([boxNumber('0.1', RETOPO_SIZES), boxNumber('99', RETOPO_SIZES), boxNumber('99.00', RETOPO_SIZES)], [0.1, 99, 99]);
  // Boxes without decimals keep to digits.
  assert.deepEqual(['1.5', '0,3', '123'].map((text) => boxText(text, { max: 99 })), ['15', '03', '12']);
});

test('a size is written without trailing zeros and with a "k": the boxes under the model, the panel, the tile', async () => {
  assert.deepEqual([0.1, 0.25, 0.3, 2, 12.5, 99].map(retopoSizeName), ['0.1k', '0.25k', '0.3k', '2k', '12.5k', '99k']);
  const p = made(prop('a', 'triflow', {}, [0.25, 2]), 'triflow', { 0.25: 262, 2: 2050 }, { picked: 0.25 });
  const { host, variants } = await mount([p]);
  assert.deepEqual(variants().map((b) => [b.byClass('rt-variant-size')[0].textContent, b.classList.contains('picked')]),
    [['0.25k', true], ['2k', false]]);
  assert.equal(variants()[0].title, '0.25k vertices (262 made). Picked: the next stages use this one.');
  assert.equal(shownWith.at(-1).url.split('?')[0], '/files/Props/props/a/retopo/0.25k/triflow.obj');
  assert.match(host.byClass('sv-subtitle')[0].textContent, /^0\.25k · 262\svertices/);
  assert.deepEqual(RETOPO.status(store.project.props[0]), { state: 'done', message: 'Clean mesh of 0.25k picked' });
  variants()[1].click();
  await settle();
  assert.deepEqual(hub.bodies.at(-1), { size: 2 }, 'picked by its number');
  store.project.props[0].retopo.settings.sizes = [0.25, 2, 0.5];
  RETOPO.update({ selectedId: 'a', stageId: 'retopo', project: store.project, select() {}, goto() {}, moving: false });
  assert.equal(RETOPO.status(store.project.props[0]).message, 'Clean mesh of 2k picked · 0.5k not made yet');
  assert.equal(host.all((e) => e.tagName === 'BUTTON' && /^Make /.test(e.textContent))[0].textContent, 'Make 0.5k');
});

test('boxes brought down to one size that is made: the hub picks it at the save, and the views hear of it', async () => {
  const p = made(prop('a', 'triflow', {}, [4, 22]), 'triflow', { 4: 4100, 22: 21_800 });
  const { variants, update } = await mount([p]);
  const heard = [];
  const off = on('prop', (now) => heard.push([now.retopo.obj, now.retopo.accepted]));
  hub.saved = (now) => {
    const only = now.retopo.variants.find((v) => v.size === 22);
    Object.assign(now.retopo, { obj: only.obj, stats: only.stats, accepted: true });
    return now;
  };
  try {
    RETOPO.sizeTile('a', [22]);
    await flushSettings();
    assert.deepEqual(heard, [['retopo/22k/triflow.obj', true]]);
    update();
    assert.deepEqual(variants().map((b) => [b.byClass('rt-variant-size')[0].textContent, b.classList.contains('picked')]),
      [['22k', true]]);
    // A save that changes nothing of the kind stays quiet: the workbench shows those values already.
    hub.saved = null;
    RETOPO.sizeTile('a', [22, 9]);
    await flushSettings();
    assert.equal(heard.length, 1);
  } finally {
    hub.saved = null;
    off();
  }
});

// ---- Retopo All, the clean meshes, the pick ----------------------------------------------------------------

test('going down to Retopo starts no retopology; Retopo All queues one job per method for the props that lack a size', async () => {
  const props = [prop('a'), prop('b', 'triflow', {}, [4, 22]), prop('c', 'lato2'), prop('d', 'lato2'), prop('e', 'imesh'),
    made(prop('f', 'lato2'), 'lato2', { 2: 2000 }, { picked: 2 }),
    prop('g', 'triflow', { mesh: { glb: null }, images: { chosen: null } })];
  const { go } = await mount(props, null);
  await carryDown('retopo');
  assert.deepEqual(hub.calls, [], 'the props have their 3D models (or nothing to make one from): nothing to queue on the way');
  // The tool of its first job loads as the stage opens, taking the card from a tool idling on it.
  assert.deepEqual(RETOPO.warmWish(), { tool: 'triflow', displace: true });
  assert.ok(!RETOPO.rowAction, 'no button over the call to action: Retopo All is in the viewport');
  const button = go();
  assert.deepEqual([button.textContent, button.disabled], ['Retopo All5', false], 'its count: the props to make');
  assert.equal(button.title, 'Retopologizes 5 props (2 with TriFlow, then 2 with LATO.2, then 1 with Instant Meshes). '
    + 'Each gets a clean mesh of every size in the boxes on its tile that it lacks.');
  button.click();
  await settle();
  assert.deepEqual(hub.bodies.map((b) => [b.prop_ids, b.tool]), [[['a', 'b'], 'triflow'], [['c', 'd'], 'lato2'], [['e'], 'imesh']]);
  // While TriFlow's group runs, the props of the tools after it say what they wait for.
  store.jobs = hub.bodies.map((b, i) => hubJob({ id: `j${i}`, stage: 'retopo', status: i ? 'queued' : 'running',
    current_prop: i ? null : 'a', progress: {}, ...b }));
  assert.equal(jobState('retopo', props[2]).message, 'Waits for TriFlow to finish');
  assert.equal(jobState('retopo', props[4]).message, 'Waits for TriFlow to finish');
  assert.equal(jobState('retopo', props[1]).message, 'Waiting for its turn', 'its own group: its turn');
  assert.equal(RETOPO.warmWish(), null, 'everything is queued: nothing to load ahead');
  RETOPO.update({ selectedId: null, stageId: 'retopo', project: store.project, select() {}, goto() {}, moving: false });
  assert.deepEqual([button.textContent, button.disabled], ['Retopologizing…', true]);
});

test('with nothing left to make, Retopo All asks before making every clean mesh again', async () => {
  const props = [made(prop('a', 'lato2'), 'lato2', { 2: 2000 }, { picked: 2 })];
  const { go } = await mount(props);
  const button = go();
  assert.deepEqual([button.textContent, button.disabled], ['Retopo All again1', false]);
  assert.equal(button.title, 'Retopologizes 1 prop. Every clean mesh of theirs is made anew.');
  button.click();
  await settle();
  const dialog = document.body.byClass('modal').at(-1);
  assert.match(dialog.textContent, /Every prop has its clean meshes already/);
  dialog.querySelectorAll('.btn').find((b) => b.textContent === 'Make them again').click();
  await settle();
  assert.deepEqual(hub.bodies.map((b) => [b.prop_ids, b.tool]), [[['a'], 'lato2']]);
});

test('the boxes under the model, one per size: a click shows that clean mesh where the camera is and picks it', async () => {
  const p = made(prop('a', 'triflow', {}, [4, 22, 6]), 'triflow', { 4: 4100, 22: 21_800 });
  const { variants, update } = await mount([p]);
  const words = () => variants().map((b) => [b.byClass('rt-variant-size')[0].textContent, b.disabled,
    b.classList.contains('picked'), b.byClass('rt-variant-check').length]);
  assert.deepEqual(words(), [['4k', false, false, 0], ['22k', false, false, 0], ['6k', true, false, 0]],
    'none picked yet; 6k not made');
  assert.equal(variants()[2].title, '6k vertices: not made yet. Retopo All makes it.');
  assert.equal(shownWith.at(-1).url.split('?')[0], '/files/Props/props/a/retopo/4k/triflow.obj', 'the one the prop uses');
  assert.ok(RETOPO.status(p).message.startsWith('Ready to pick'));
  const nudgedBefore = nudged;
  variants()[1].click();
  await settle();
  assert.deepEqual(shownWith.at(-1), { url: shownWith.at(-1).url, keepView: true }, 'the camera stays');
  assert.equal(shownWith.at(-1).url.split('?')[0], '/files/Props/props/a/retopo/22k/triflow.obj');
  assert.deepEqual([hub.calls.at(-1), hub.bodies.at(-1)], ['POST /api/projects/Props/props/a/retopo/pick', { size: 22 }]);
  assert.deepEqual(words()[1], ['22k', false, true, 1], 'picked: its green check');
  assert.equal(nudged, nudgedBefore + 1, 'every prop is ready now: the call to action asks for attention');
  // Clicked again: shown, nothing asked. A disabled one (not made) can't be picked.
  hub.calls = [];
  variants()[1].click();
  await settle();
  assert.deepEqual(hub.calls, []);
  update();
  assert.deepEqual(RETOPO.status(store.project.props[0]), { state: 'done', message: 'Clean mesh of 22k picked · 6k not made yet' });
  // Another number typed for a size that is not made yet: its box says the new one.
  store.project.props[0].retopo.settings.sizes = [4, 22, 45];
  update();
  assert.deepEqual(words().map(([size]) => size), ['4k', '22k', '45k']);
});

test('a size outside the vertices its method places is made with the nearest count, and the page says so', async () => {
  assert.deepEqual(RETOPO_VERTICES, { triflow: [250, 99_000], lato2: [200, 10_000], imesh: [100, 100_000] });
  assert.deepEqual([0.1, 0.24, 0.25, 2, 99].map((size) => retopoVertices('triflow', size)), [250, 250, 250, 2000, 99_000]);
  assert.deepEqual([0.1, 0.2, 0.25, 10, 22].map((size) => retopoVertices('lato2', size)), [200, 200, 250, 10_000, 10_000]);
  assert.deepEqual([0.1, 0.25, 2.01, 12.25, 99].map((size) => retopoVertices('imesh', size)),
    [100, 250, 2010, 12_250, 99_000], 'whole counts (a thousand times 2.01 is 2009.9999999999998)');
  // LATO.2, under its least and over its most: the boxes under the model and the panel's quiet line.
  const p = made(prop('a', 'lato2', {}, [0.1, 4, 22]), 'lato2', { 4: 4000, 22: 10_000 });
  const { host, variants, update } = await mount([p]);
  const cap = () => host.byClass('rt-cap')[0];
  assert.equal(variants()[0].title, '0.1k vertices: not made yet. Retopo All makes it. LATO.2 makes 0.2k vertices at '
    + 'least: this size is made with 0.2k.');
  assert.equal(variants()[1].title, '4k vertices (4,000 made). Click to see it and pick it.');
  assert.match(variants()[2].title, /\. LATO\.2 makes 10k vertices at most: this size is made with 10k\.$/);
  assert.equal(cap().textContent, 'LATO.2 places 200 vertices at least, so 0.1k is made with that many. LATO.2 places '
    + '10,000 vertices at most, so 22k is made with that many.');
  assert.equal(cap().hidden, false);
  // TriFlow: two triangles a vertex, 500 at least.
  Object.assign(store.project.props[0].retopo.settings, { method: 'triflow', sizes: [0.1, 0.15, 22] });
  update();
  assert.equal(cap().textContent, 'TriFlow places 250 vertices at least, so 0.1k and 0.15k are made with that many.');
  assert.match(variants()[1].title, /^0\.15k vertices: not made yet\. Retopo All makes it\. TriFlow makes 0\.25k vertices at least: this size is made with 0\.25k\.$/);
  assert.equal(variants()[2].title, '22k vertices: not made yet. Retopo All makes it.');
  // Instant Meshes makes every size as asked.
  store.project.props[0].retopo.settings.method = 'imesh';
  update();
  assert.deepEqual([cap().hidden, cap().textContent], [true, '']);
  assert.ok(variants().every((b) => !/ at (least|most)/.test(b.title)));
});

test('the pointer over High or Low rings the Wireframe toggle under the viewport', async () => {
  const { host } = await mount([made(prop('a', 'triflow'), 'triflow', { 2: 2000 }, { picked: 2 })]);
  const [wire] = host.all((e) => e.tagName === 'BUTTON' && e.textContent === 'Wireframe');
  const sides = host.all((e) => e.tagName === 'BUTTON' && ['High', 'Low'].includes(e.textContent));
  const rung = [];
  wire.animate = (frames, timing) => rung.push(frames.map((f) => Object.keys(f)[0]).join('>') + ` ${timing.duration}`);
  assert.deepEqual(sides.map((b) => b.textContent), ['High', 'Low']);
  for (const side of sides) for (const fn of side.listeners.pointerenter) fn();
  const once = ['boxShadow>boxShadow 700', 'transform>transform>transform 340'];
  assert.deepEqual(rung, [...once, ...once], 'for each: the ring Images gives the Resolution, and the toggle swells for a moment');
});

test('a prop picked shows its Low side with the edges on again, whatever was chosen on the prop before; the lights of "All" reach further here', async () => {
  const { viewer } = await import(VIEWER);
  const a = made(prop('a', 'triflow'), 'triflow', { 2: 2000 }, { picked: 2 });
  const b = made(prop('b', 'triflow'), 'triflow', { 2: 2000 }, { picked: 2 });
  const { host } = await mount([a, b]);
  const wire = () => host.all((e) => e.tagName === 'BUTTON' && e.textContent === 'Wireframe')[0];
  const settle = async () => { for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0)); };
  const pick = async (id) => {
    RETOPO.update({ selectedId: id, stageId: 'retopo', project: store.project, select() {}, goto() {}, moving: false });
    await settle();
  };
  await settle();
  assert.equal(viewer.lightReach, 1.95, 'Retopo\'s clay lit by the same lights from further');
  assert.equal(wire().getAttribute('aria-pressed'), 'true');
  wire().click();
  assert.equal(wire().getAttribute('aria-pressed'), 'false', 'the artist turned it off');
  await pick('b');
  assert.equal(wire().getAttribute('aria-pressed'), 'true', 'another prop: on again');
  wire().click();
  await pick('a');
  assert.equal(wire().getAttribute('aria-pressed'), 'true', 'and back: on again');
});

test('the call to action goes on to Projection only once every prop has a picked, up-to-date clean mesh', async () => {
  const ready = made(prop('a', 'lato2'), 'lato2', { 2: 2000 }, { picked: 2 });
  const pick = made(prop('b', 'lato2', {}, [2, 4]), 'lato2', { 2: 2000, 4: 4000 });
  const none = prop('c', 'triflow');
  await mount([ready, pick, none]);
  const gate = RETOPO.ctaEnabled();
  assert.deepEqual(gate, { enabled: false, blocked: true, hint: 'Every prop needs a picked clean mesh first: 1 prop needs a '
    + 'pick (click a size under the model), 1 prop needs its clean meshes (Retopo All).' });
  assert.equal(RETOPO.ctaHover(), gate.hint, 'its hover says why');
  made(none, 'triflow', { 2: 2100 }, { picked: 2 });
  pick.retopo.accepted = true;
  store.jobs = [hubJob({ id: 'j1', stage: 'retopo', tool: 'triflow', status: 'queued', prop_ids: ['c'], progress: {} })];
  assert.match(RETOPO.ctaEnabled().hint, /1 prop is being retopologized\.$/);
  store.jobs = [];
  assert.deepEqual(RETOPO.ctaEnabled(), { enabled: true });
  assert.match(RETOPO.ctaHover(), /^Bakes the detail and colours/);
  assert.equal(waitingHint([]), '', 'no props: nothing holds the call to action');
  // The props the stage is about hold it back: one without a 3D model has nothing to make clean - unless its 3D model
  // is being made -, and a tileable texture never has.
  const flat = { ...prop('d', 'triflow'), mesh: { glb: null } };
  const texture = { ...prop('e', 'triflow'), images: { chosen: 'images/0001.png', settings: { tileable: true } } };
  store.project.props.push(flat, texture);
  assert.deepEqual(RETOPO.ctaEnabled(), { enabled: true });
  store.jobs = [hubJob({ id: 'j2', stage: 'mesh', tool: 'pixal3d', status: 'running', prop_ids: ['d'], progress: {} })];
  assert.match(RETOPO.ctaEnabled().hint, /^Every prop needs a picked clean mesh first: 1 prop needs its clean meshes/);
  store.jobs = [];
  // A texture has no size boxes on its tile; a prop without a 3D model keeps them, for when it has one.
  assert.deepEqual([RETOPO.tileSizes(texture), RETOPO.tileSizes(flat)?.values], [null, [2]]);
});

test('Instant Meshes\' viewer opens from the prop\'s panel and goes back to its clean meshes', async () => {
  const p = made(prop('a', 'imesh'), 'imesh', { 2: 2000 }, { picked: 2 });
  const { host } = await mount([p]);
  const button = host.all((e) => e.tagName === 'BUTTON' && e.textContent === 'Shape it by hand in the viewer')[0];
  assert.ok(button && !button.disabled);
  button.click();
  await settle();
  assert.ok(hub.calls.includes('POST /api/imesh/session'), 'its session opens');
  assert.equal(host.byClass('sv-imesh-frame').length, 1);
  assert.equal(host.byClass('rt-variants').length, 0);
  host.byClass('rt-back')[0].click();
  await settle();
  assert.deepEqual([host.byClass('sv-imesh-frame').length, host.byClass('rt-variants').length], [0, 1]);
});

// ---- a prop's status, its settings, the clean meshes each method keeps ---------------------------------------

test('a prop\'s status in Retopo follows the method it is set to and the pick', () => {
  reset([], { tools: tools({ triflow: { supported: false, state: 'unsupported', reason: 'TriFlow needs a newer graphics card.' } }) });
  const st = (p) => [RETOPO.status(p).state, RETOPO.status(p).message];
  assert.deepEqual(st(made(prop('a', 'lato2', {}, [2, 4]), 'lato2', { 2: 2000, 4: 4000 })),
    ['review', 'Ready to pick: click a size under the model']);
  assert.deepEqual(st(made(prop('a', 'lato2'), 'lato2', { 2: 2000 }, { picked: 2 })), ['done', 'Clean mesh of 2k picked']);
  assert.deepEqual(st(made(prop('a', 'lato2'), 'imesh', { 2: 2000 }, { picked: 2 })),
    ['stale', 'Made with Instant Meshes, not LATO.2: make it again'], 'another method picked: to do again');
  assert.deepEqual(st(made(prop('a'), 'lato2', { 2: 2000 })), ['stale', 'Made with LATO.2, not TriFlow: make it again'],
    'no method picked: TriFlow, the default');
  assert.deepEqual(st(made(prop('a', 'lato2'), 'lato2', { 2: 2000 }, { picked: 2, stale: true })),
    ['stale', 'Made from an older 3D model: make it again']);
  assert.deepEqual(st(prop('a', 'triflow')), ['unreachable', 'TriFlow needs a newer graphics card.'], 'its tool can\'t run here');
  assert.deepEqual(st(prop('a', 'lato2')), ['idle', 'Not retopologized yet']);
  assert.deepEqual(st(prop('a', 'lato2', {}, [])), ['idle', 'Type a size in the boxes on its picture']);
});

test('a clean mesh made with other settings than the prop has now says so; an older result without them does not', () => {
  const with_ = (method, stats, settings = {}) => made(prop('a', method, {}), method, { 2: 2000 },
    { picked: 2, stats, settings: { method, sizes: [2], ...settings } });
  assert.equal(settingsChanged(with_('lato2', { seed: 0 })), false, 'the defaults');
  assert.equal(settingsChanged(with_('lato2', { seed: 0 }, { lato2_seed: 3 })), true);
  assert.equal(settingsChanged(with_('lato2', {}, { lato2_seed: 3 })), false, 'made before it was kept');
  const tri = { target_faces: 4000, quad_ratio: 0.95, seed: 0 };
  assert.equal(settingsChanged(with_('triflow', tri)), false);
  assert.equal(settingsChanged(with_('triflow', tri, { triflow_quad_ratio: 0.5 })), true);
  assert.equal(settingsChanged(with_('triflow', tri, { triflow_seed: 7 })), true);
  // another method picked: that is "to do" again anyway (retopoFresh), not "settings changed"
  const other = with_('triflow', tri);
  other.retopo.settings.method = 'lato2';
  assert.equal(settingsChanged(other), false);
});

test('each method keeps its clean meshes while another is picked: back at its method, they are there again', () => {
  assert.equal(RETOPO_KEEP_VERTICES, 50_000);
  const pick = (p, method) => ({ ...p, retopo: { ...retopoSwitched(p.retopo, method), settings: { ...p.retopo.settings, method } } });
  let p = made(prop('a', 'triflow', {}, [2, 4]), 'triflow', { 2: 2100, 4: 4000 }, { picked: 2, session_id: 's1' });
  p = pick(p, 'lato2');
  assert.deepEqual([p.retopo.obj, p.retopo.method, p.retopo.accepted, p.retopo.variants, p.retopo.stats],
    [null, null, false, [], undefined], 'none for LATO.2 yet');
  assert.equal(p.retopo.session_id, 's1', 'what is not the clean meshes stays');
  assert.deepEqual(p.retopo.kept.triflow.variants.map((v) => v.size), [2, 4]);
  p = made(p, 'lato2', { 2: 1900 });
  assert.equal(retopoFresh(p), true);
  p = pick(p, 'triflow');
  assert.deepEqual([p.retopo.obj, p.retopo.stats.vertices, p.retopo.accepted, retopoFresh(p), p.retopo.variants.length],
    ['retopo/2k/triflow.obj', 2100, true, true, 2], 'TriFlow\'s clean meshes, their numbers and their pick again');
  assert.deepEqual(Object.keys(p.retopo.kept), ['lato2']);
  p = pick(p, 'imesh');
  assert.deepEqual([p.retopo.obj, Object.keys(p.retopo.kept).sort()], [null, ['lato2', 'triflow']]);
  // One of 50,000 vertices or more: not kept. They stay, "to do" again, until another method's replace them.
  const big = made(prop('b', 'triflow', {}, [2, 50]), 'triflow', { 2: 2000, 50: 50_000 });
  assert.equal(retopoSwitched(big.retopo, 'lato2'), big.retopo);
  assert.equal(retopoFresh(pick(big, 'lato2')), false);
  const back = made(prop('x', 'lato2'), 'lato2', { 2: 1900 }).retopo;
  const replaced = retopoSwitched({ ...big.retopo, kept: { lato2: back } }, 'lato2');
  assert.deepEqual([replaced.obj, replaced.kept], ['retopo/2k/lato2.obj', undefined]);
  // A project from before: the hub has made its clean mesh the one variant; one without a count is not kept.
  const old = { obj: 'retopo/retopo.obj', source: 'mesh/mesh.glb', settings: { method: 'imesh' },
    variants: [{ size: 2, obj: 'retopo/retopo.obj', stats: null }] };
  assert.equal(retopoSwitched(old, 'imesh'), old);
  assert.equal(retopoSwitched(old, 'triflow'), old, 'it does not say how many vertices: it stays');
});

test('a method picked on one prop is the default from then on: the props without a clean mesh take it, the hub keeps it for new props, a toast says so', async () => {
  const toasts = new FakeEl('div');
  const byId = document.getElementById;
  document.getElementById = (id) => (id === 'toasts' ? toasts : byId(id));
  try {
    const done = made(prop('done', 'triflow'), 'triflow', { 2: 2000 }, { picked: 2 });
    const { host } = await mount([prop('a', 'triflow'), prop('b', 'triflow'), prop('c'), done, { ...prop('draft', 'triflow'), draft: true }]);
    assert.deepEqual(methodFollowers('a', 'lato2').map((p) => p.id), ['b', 'c', 'draft'], 'not the one that has its clean meshes');
    const button = (label) => host.all((e) => e.tagName === 'BUTTON' && e.textContent.startsWith(label))[0];
    button('LATO.2').click();
    await flushSettings();
    await settle();
    const method = (id) => store.project.props.find((p) => p.id === id).retopo.settings.method;
    assert.deepEqual(['a', 'b', 'c', 'done', 'draft'].map(method), ['lato2', 'lato2', 'lato2', 'triflow', 'lato2']);
    assert.deepEqual(store.project.ui.retopo_picked, { a: true });
    const sent = hub.calls.map((call, i) => [call, hub.bodies[i]]);
    assert.deepEqual(sent.find(([call]) => call === 'PATCH /api/settings')[1], { retopo: { default_method: 'lato2' } });
    assert.deepEqual(sent.filter(([call]) => /\/props\/[^/]+$/.test(call)).map(([call, body]) => [call.split('/').pop(), body.retopo.settings.method]).sort(),
      [['a', 'lato2'], ['b', 'lato2'], ['c', 'lato2'], ['draft', 'lato2']]);
    assert.equal(toasts.lastChild.byClass('toast-text')[0].textContent,
      'LATO.2 is now the default for all the props: new props start with it, and 2 props here without a clean mesh were '
      + 'switched to it. The props that have their clean meshes, or a method you picked for them, keep theirs.');
    // A prop set apart by hand stays apart: 'a' was picked, so a pick on 'b' leaves it.
    pickMethod('b', 'imesh');
    assert.deepEqual(['a', 'b', 'c', 'done'].map(method), ['lato2', 'imesh', 'imesh', 'triflow']);
    // The same method again changes nothing and says nothing.
    const before = toasts.children.length;
    pickMethod('b', 'imesh');
    await settle();
    assert.equal(toasts.children.length, before + 1, 'only the pick before it spoke');
  } finally {
    document.getElementById = byId;
  }
});

test('the Move / Rotate bar: where the prop stands in its saved files - off at first on Retopo, a drag is saved, Back takes it away', async () => {
  const placed = { move: [0, 0.5, 0], turn: [0, 0, 0, 1] };
  const a = made(prop('a', 'triflow', { placement: placed }), 'triflow', { 2: 2000 }, { picked: 2 });
  const { host } = await mount([a]);
  const bar = host.byClass('sv-place')[0];
  const button = (label) => bar.all((e) => e.tagName === 'BUTTON' && e.getAttribute('aria-label') === label)[0];
  assert.equal(bar.hidden, false, 'with the model on screen');
  assert.deepEqual(viewer.placements.at(-1), placed, 'the model is shown where the prop stands');
  assert.equal(viewer.gizmo, null, 'no handles until asked: the topology is what is looked at here');
  assert.equal(button('Back as it was made').disabled, false);
  button('Move').click();
  await settle();
  assert.deepEqual([viewer.gizmo, button('Move').getAttribute('aria-pressed'), button('Rotate').getAttribute('aria-pressed')],
    ['move', 'true', 'false']);
  hub.calls = [];
  hub.bodies = [];
  const turned = { move: [0.1, 0.5, 0], turn: [0, 0.7071068, 0, 0.7071068] };
  viewer.drag(turned);
  await settle();
  const saves = () => hub.bodies.filter((body, i) => hub.calls[i] === 'PATCH /api/projects/Props/props/a');
  assert.deepEqual(saves(), [{ placement: turned }]);
  assert.deepEqual(store.project.props[0].placement, turned);
  button('Rotate').click();
  await settle();
  assert.equal(viewer.gizmo, 'turn');
  button('Rotate').click();
  await settle();
  assert.equal(viewer.gizmo, null, 'pressed again: the handles go');
  // On the floor puts the handles up (the floor shows with them) and saves where the model then stands.
  button('Stand it on the floor').click();
  await settle();
  assert.equal(viewer.gizmo, 'move');
  assert.deepEqual(saves().at(-1), { placement: { move: [0, 0.5, 0], turn: turned.turn } });
  button('Back as it was made').click();
  await settle();
  assert.deepEqual([saves().at(-1), store.project.props[0].placement, button('Back as it was made').disabled],
    [{ placement: null }, null, true], 'a placement that moves nothing is none');
  assert.equal(placementOrNone({ move: [0, 0, 0], turn: [0, 0, 0, -1] }), null);
  // The handles stay as the artist left them on this page, for the next prop too: taken away here.
  button('Move').click();
  await settle();
  assert.equal(viewer.gizmo, null);
});

test('the Export page shows the finished model where it stands in its saved files, the Move handles up from the start', async () => {
  const placed = { move: [0, 0.47, 0], turn: [0, 0, 0, 1] };
  const a = made(prop('a', 'triflow', { placement: placed, projection: { glb: 'projection/bake_001/projection.glb' } }),
    'triflow', { 2: 2000 }, { picked: 2 });
  RETOPO.unmount();
  reset([a], { tools: tools(), install: { components: {} } });
  const host = new FakeEl('div');
  const ctx = { selectedId: 'a', stageId: 'rig', project: store.project, select() {}, goto() {}, moving: false, settled: Promise.resolve() };
  EXPORT.mount(host, ctx);
  await settle();
  try {
    assert.match(viewer.shown.at(-1), /projection\.glb/);
    assert.deepEqual([viewer.placements.at(-1), viewer.gizmo], [placed, 'move']);
    assert.match(host.byClass('sv-place-hint')[0].textContent, /the grid is the floor of the saved files/);
    // Another page's model is shown as it was made, without handles.
    EXPORT.unmount();
    await mount([prop('b', 'triflow', { placement: placed })]);
    assert.deepEqual([viewer.placements.at(-1), viewer.gizmo], [placed, null]);
  } finally {
    EXPORT.unmount();
  }
});

// ---- TriFlow's terms, the tool lists ---------------------------------------------------------------------------

test('TriFlow is free for non-commercial use only: its terms, with or without the hub\'s', () => {
  reset([], { tools: tools() });
  assert.equal(nonCommercial('triflow'), true);
  assert.equal(nonCommercial('lato2'), false);
  assert.equal(nonCommercial('imesh'), false);
  const own = licenseOf('triflow');
  assert.match(own.url, /^https:\/\/github\.com\/IgorAherne\/triflow\/blob\/873a2cdf68b0a1ec243635bbf9170c87556b2f2b\/LICENSE$/);
  assert.ok(own.links.some((l) => /MeshLib/.test(l.label)), 'MeshLib\'s terms go with it');
  assert.ok(own.links.every((l) => /\/MeshInspector\/MeshLib\/blob\/v[\d.]+\/LICENSE$/.test(l.url)),
    'its licence file at the version installed, not the repository\'s front page');
  reset([], { tools: tools({ triflow: { license: { name: 'ADPNCL 1.0', url: 'https://example.org/LICENSE', noncommercial: true } } }) });
  assert.equal(licenseOf('triflow').url, 'https://example.org/LICENSE', 'the hub\'s link when it has one');
  assert.ok(licenseOf('triflow').links.length, 'the MeshLib link stays');
  reset([], { tools: tools({ lato2: { license: { name: 'MIT', url: 'https://example.org/MIT', noncommercial: false } } }) });
  assert.equal(licenseOf('lato2'), null, 'nothing to flag');
});

test('a save says where it went, and what the hub warned of - nothing more of TriFlow\'s terms (said where it is picked)', () => {
  const toasts = new FakeEl('div');
  const { getElementById } = document;
  document.getElementById = (id) => (id === 'toasts' ? toasts : null);
  try {
    toastSaved('Barrel.glb', { folder: 'exports/Demo/Barrel', opened: true });
    toastSaved('2 props', { folder: 'exports/Demo', opened: true, warnings: ['Crate: its FBX could not be made.'] });
    assert.deepEqual(toasts.children.map((t) => t.textContent), [
      'Saved Barrel.glb to exports\\Demo\\Barrel', 'Saved 2 props to exports\\Demo. Crate: its FBX could not be made.']);
  } finally {
    document.getElementById = getElementById;
  }
});

test('Diagnostics, the install card and the status line know every tool', () => {
  assert.deepEqual(DIAGNOSTICS, IDS);
  assert.deepEqual(INSTALL_ORDER, ['hub', 'imesh', 'qwen', 'forge', 'pixal3d', 'lato2', 'triflow']);
  for (const id of IDS) {
    assert.ok(TOOL_NAMES[id] && TOOL_ROLES[id], `${id} has a name and a role`);
    assert.ok(INSTALL_NAMES[id], `${id} has a line in the install card`);
  }
  assert.equal(TOOL_NAMES.lato2, 'LATO.2');
  assert.equal(TOOL_NAMES.triflow, 'TriFlow');
  assert.equal(TOOL_NAMES.qwen, 'Qwen-Image 2.1');
});
