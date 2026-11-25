// The 3D stage's Shape in the page (no browser): Hi3D or Pixal3D per prop (mesh.settings.geometry; a prop of before the
// choice keeps Pixal3D's), saved like the other 3D settings, "Powered by Hi3D 1.0" linked to Hi3D's site while Hi3D is
// picked or made the model shown (its licence asks for both), and what the panel says while Hi3D's AI models are not
// on this PC (/api/status "hi3d"): their download on request (the button, or the question asked when Hi3D is picked),
// its progress, then Hi3D's convolutions measured.
//   node --test code/tests/web/mesh.test.mjs
// (code/tests/test_web.py runs every code/tests/web/*.test.mjs when Node.js is installed)
import assert from 'node:assert/strict';
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
    this.parentNode = null;
  }

  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter((f) => f !== fn); }
  fire(type, event = {}) { for (const fn of this.listeners[type] || []) fn({ preventDefault() {}, stopPropagation() {}, target: this, ...event }); }
  click() { this.fire('click'); }

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
  replaceChildren(...nodes) {
    for (const c of [...this.children]) if (c instanceof FakeEl) c.remove();
    this.children = [];
    this.append(...nodes);
  }

  insertBefore(node, ref) {
    this.adopt(node);
    this.children.splice(ref ? this.children.indexOf(ref) : this.children.length, 0, node);
    return node;
  }

  replaceWith(node) {
    const parent = this.parentNode;
    if (!parent) return;
    parent.insertBefore(node, this);
    this.remove();
  }

  get firstChild() { return this.children[0] ?? null; }
  get lastChild() { return this.children.at(-1) ?? null; }
  get firstElementChild() { return this.children.find((c) => c instanceof FakeEl) ?? null; }
  get nextElementSibling() {
    const siblings = this.parentNode?.children || [];
    return siblings.slice(siblings.indexOf(this) + 1).find((c) => c instanceof FakeEl) ?? null;
  }

  get isConnected() { return this === document.body || !!this.parentNode?.isConnected; }
  contains(node) { return node === this || this.children.some((c) => c instanceof FakeEl && c.contains(node)); }
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
  showModal() { this.open = true; }
  close() { this.open = false; this.fire('close'); }

  /** Every element in it (itself first) that `pick` accepts. */
  all(pick) {
    const out = pick(this) ? [this] : [];
    for (const c of this.children) if (c instanceof FakeEl) out.push(...c.all(pick));
    return out;
  }

  byClass(name) { return this.all((e) => e.classList.contains(name)); }
  /** The elements in it that have every class of the last part of a selector like ".modal-actions .btn". */
  querySelectorAll(sel) {
    const classes = sel.trim().split(/\s+/).at(-1).split('.').filter(Boolean);
    return this.all((e) => e !== this && classes.every((c) => e.classList.contains(c)));
  }

  querySelector(sel) { return this.querySelectorAll(sel)[0] ?? null; }
  /** This element shows: neither it nor anything it is in is hidden. */
  get showing() { return !this.hidden && (this.parentNode ? this.parentNode.showing : true); }
}
class FakeTemplate extends FakeEl {
  set innerHTML(html) { this.html = html; }
  get content() {
    const svg = new FakeEl('svg');
    svg.className = /class="([^"]*)"/.exec(this.html)?.[1].trim() ?? '';
    return { firstChild: svg };
  }
}
const pageListeners = {};
const toasts = new FakeEl('div');
globalThis.Node = FakeNode;
globalThis.addEventListener ??= () => {};
globalThis.innerWidth = 1280;
globalThis.innerHeight = 800;
globalThis.document = {
  addEventListener: (type, fn) => { (pageListeners[type] ||= []).push(fn); },
  removeEventListener: () => {}, hidden: false, activeElement: null, documentElement: { dataset: {} },
  body: new FakeEl('body'),
  querySelector: () => null,
  getElementById: (id) => (id === 'toasts' ? toasts : null),
  createElement: (tag) => (tag === 'template' ? new FakeTemplate(tag) : new FakeEl(tag)),
  createElementNS: (ns, tag) => new FakeEl(tag),
};

const STATIC = new URL('../../spellcast/static/', import.meta.url);
const load = (path) => import(new URL(`js/${path}`, STATIC));

const { store } = await load('store.js');
const { GEOMETRIES, madeWith, meshGeometry } = await load('stages/rules.js');
const { HI3D_SITE, POWERED_BY_HI3D, SHAPES, agreeToHi3d, hi3dNote, default: MESH } = await load('stages/mesh.js');

const IDS = ['qwen', 'pixal3d', 'imesh', 'lato2', 'triflow'];
const tools = () => Object.fromEntries(IDS.map((t) => [t, {
  state: 'stopped', message: '', installed: true, supported: true, uses_gpu: t !== 'imesh',
}]));
const HI3D_BYTES = 11_364_913_260;
/** A prop with a picture, set to `geometry` (undefined: none saved), and `made`: its model, made by that geometry. */
const prop = (geometry, made = undefined) => ({
  id: 'p_1', name: 'Barrel', images: { chosen: 'images/0001.png', candidates: ['images/0001.png'], views: [] },
  mesh: {
    settings: geometry === undefined ? {} : { geometry }, glb: made ? 'mesh/mesh.glb' : null,
    source: made ? 'images/0001.png' : null, ...(made ? { geometry: made } : {}),
  },
  retopo: { obj: null, settings: {} },
});
const status = (hi3d = { available: true, total_bytes: HI3D_BYTES }, components = {}) => ({
  tools: tools(), install: { running: false, components }, multiview: { available: true }, hi3d,
});

function reset(props, st = status()) {
  store.connected = true;
  store.jobs = [];
  store.jobErrors = {};
  store.status = st;
  store.project = { id: 'Props', name: 'Props', props, ui: {} };
  store.bust = {};
  store.openStamp = 7;
}

/** The workbench of the 3D stage on `p`; the workbench "moves", so no model is ever read here. */
function mount(p, st = status()) {
  reset([p], st);
  const host = new FakeEl('div');
  const ctx = {
    selectedId: 'p_1', stageId: 'mesh', project: store.project, boxes: store.project.props, select() {}, goto() {},
    moving: true, settled: new Promise(() => {}),
  };
  MESH.mount(host, ctx);
  const [field] = host.byClass('sv-shape');
  const [seg] = field.byClass('seg');
  const [chip] = field.byClass('hi3d');
  const note = field.nextElementSibling;
  return { host, seg, chip, note, update: () => MESH.update(ctx) };
}

test('two shapes, Hi3D first: their words, and Pixal3D\'s for a prop that names none (made before the choice)', () => {
  assert.deepEqual(GEOMETRIES, ['hi3d', 'pixal3d'], 'projects.GEOMETRIES, the default first');
  assert.deepEqual(SHAPES.map((s) => [s.value, s.label, s.title]), [
    ['hi3d', 'Hi3D', 'Hi3D: finer carved detail, slower; from the front picture alone.'],
    ['pixal3d', 'Pixal3D', 'Pixal3D: faster, follows the picture closely.']]);
  assert.equal(meshGeometry(prop('hi3d')), 'hi3d');
  assert.equal(meshGeometry(prop('pixal3d')), 'pixal3d');
  assert.equal(meshGeometry(prop(undefined)), 'pixal3d', 'a prop of before the choice: Pixal3D made its model');
  assert.equal(meshGeometry(prop('trellis')), 'pixal3d');
  assert.equal(meshGeometry({}), 'pixal3d');
  assert.equal(madeWith(prop('pixal3d', 'hi3d')), 'hi3d');
  assert.equal(madeWith(prop('hi3d')), null, 'no model yet');
  assert.equal(madeWith(prop('hi3d', 'nobody')), null);
  assert.equal(POWERED_BY_HI3D, 'Powered by Hi3D 1.0', 'the words of Hi3D\'s licence, section 7');
  assert.equal(HI3D_SITE, 'http://hi3d.ai/', 'the site its Schedule 2 names for the link');
});

test('the Shape comes first among the settings, is saved per prop, and "Powered by Hi3D 1.0" links to Hi3D\'s site', async () => {
  const sent = [];
  globalThis.fetch = async (path, init = {}) => {
    if (init.method && init.method !== 'GET') sent.push([init.method, path, init.body]);
    return { status: 200, text: async () => '{}' };
  };
  const { host, seg, chip, update } = mount(prop('hi3d'));
  const labels = host.byClass('field-label').map((l) => l.firstChild?.textContent ?? l.textContent);
  assert.deepEqual(labels.slice(0, 2), ['Shape', 'Quality'], 'above Quality');
  const [hi3d, pixal3d] = seg.children;
  assert.deepEqual([hi3d.textContent, pixal3d.textContent], ['Hi3D', 'Pixal3D']);
  assert.deepEqual([hi3d.title, pixal3d.title], [SHAPES[0].title, SHAPES[1].title], 'short tooltips');
  assert.ok(hi3d.classList.contains('on') && !pixal3d.classList.contains('on'));
  assert.equal(seg.getAttribute('aria-label'), 'Shape');
  // Hi3D picked: its credit, a link that opens its site outside the page
  assert.equal(chip.hidden, false);
  assert.equal(chip.tagName, 'A');
  assert.equal(chip.textContent, 'Powered by Hi3D 1.0');
  assert.deepEqual([chip.getAttribute('href'), chip.getAttribute('target'), chip.getAttribute('rel')],
    ['http://hi3d.ai/', '_blank', 'noopener noreferrer']);
  // Pixal3D picked: saved with the prop's other 3D settings, and no credit
  pixal3d.click();
  assert.equal(store.project.props[0].mesh.settings.geometry, 'pixal3d');
  assert.ok(pixal3d.classList.contains('on') && !hi3d.classList.contains('on'));
  assert.equal(chip.hidden, true);
  update();
  assert.ok(pixal3d.classList.contains('on'), 'the prop stays on its pick');
  await new Promise((resolve) => setTimeout(resolve, 600));     // the settings' own save (0.5 s) goes out
  const patch = sent.find(([method, path]) => method === 'PATCH' && path.endsWith('/props/p_1'));
  assert.ok(patch, `the pick is saved: ${JSON.stringify(sent)}`);
  assert.deepEqual(JSON.parse(patch[2]).mesh.settings, { geometry: 'pixal3d' });
  MESH.unmount();
});

test('Faster 3D generation sits under More, off until the hub says otherwise, on RTX 30xx and newer only', async () => {
  const asked = [];
  globalThis.fetch = async (path, init = {}) => {
    asked.push([init.method ?? 'GET', path]);
    return { status: 200, text: async () => JSON.stringify({ pixal3d: { coarse_exact: false } }) };
  };
  try {
    for (const [cap, shown] of [[8.6, true], [7.5, false], [6.1, false]]) {
      store.settings = undefined;
      asked.length = 0;
      const { host } = mount(prop('pixal3d'), { ...status(), gpu: { compute_cap: cap } });
      const [sw] = host.all((e) => e.classList.contains('switch') && /Faster 3D generation/.test(e.textContent));
      const [fold] = host.all((e) => e.tagName === 'DETAILS' && e.contains(sw));
      assert.equal(fold.children[0].textContent, 'More', `${cap}`);
      assert.equal(sw.showing, shown, `${cap}: older cards cannot compute the original's numbers either way`);
      assert.equal(!!sw.input.checked, false, 'off until the hub\'s settings are read');
      await new Promise((resolve) => setTimeout(resolve, 0));
      assert.equal(!!sw.input.checked, shown, 'the hub\'s pixal3d.coarse_exact false, asked for where it applies');
      assert.deepEqual(asked, shown ? [['GET', '/api/settings']] : []);
      MESH.unmount();
    }
  } finally {
    store.settings = undefined;
  }
});

test('the credit stays while the model shown was made with Hi3D, whatever the prop is set to now', () => {
  let view = mount(prop('pixal3d', 'hi3d'));
  assert.equal(view.chip.hidden, false, 'made with Hi3D');
  MESH.unmount();
  view = mount(prop('pixal3d', 'pixal3d'));
  assert.equal(view.chip.hidden, true, 'made with Pixal3D, set to Pixal3D');
  MESH.unmount();
  view = mount(prop(undefined));
  assert.ok(view.seg.children[1].classList.contains('on') && view.chip.hidden, 'a prop of before the choice: Pixal3D');
  MESH.unmount();
});

test('Hi3D\'s AI models missing: the panel says so, its button starts their download, and the note follows it', async () => {
  const missing = { available: false, code: true, downloading: false, waiting: false, finishing: false, done_bytes: 0,
    total_bytes: HI3D_BYTES, error: null };
  assert.equal(hi3dNote(prop('hi3d'), status()), null, 'they are here');
  assert.equal(hi3dNote(prop('pixal3d'), status(missing)), null, 'a prop set to Pixal3D needs none');
  assert.equal(hi3dNote(prop('hi3d'), { tools: tools() }), null, 'a hub that says nothing of them: nothing to say');
  assert.deepEqual(hi3dNote(prop('hi3d'), status(missing)), {
    text: 'Hi3D needs a one-time 11.4 GB download of its AI models.', button: 'Download (11.4 GB)', failed: false,
    fraction: null,
  });
  assert.deepEqual(hi3dNote(prop('hi3d'), status({ ...missing, downloading: true, done_bytes: HI3D_BYTES * 0.4 })), {
    text: 'Downloading Hi3D’s AI models (11.4 GB)… 40%', button: null, failed: false, fraction: 0.4,
  }, 'the download runs');
  assert.deepEqual(hi3dNote(prop('hi3d'), status({ ...missing, downloading: true, waiting: true })), {
    text: 'Hi3D’s AI models wait for the installer to finish…', button: null, failed: false, fraction: 0,
  });
  assert.deepEqual(hi3dNote(prop('hi3d'), status({ ...missing, available: true, finishing: true, done_bytes: HI3D_BYTES })), {
    text: 'Hi3D’s AI models are here. Measuring its convolutions on the graphics card (a few minutes; keep heavy programs closed)…',
    button: null, failed: false, fraction: null,
  }, 'the files are there: the hub measures Hi3D\'s convolutions');
  assert.deepEqual(hi3dNote(prop('hi3d'), status({ ...missing, error: 'The drive ran out of space.' })), {
    text: 'The drive ran out of space.', button: 'Try again', failed: true, fraction: null,
  }, 'the download stopped: the hub says why');
  assert.deepEqual(hi3dNote(prop('hi3d'), status({ ...missing, code: false })), {
    text: 'Hi3D is not part of this Spellcast3D yet. Pick Pixal3D.', button: null, failed: false, fraction: null,
  }, 'without its code no download makes it work');
  // picking Hi3D asks nothing while its models are here, or on their way (the question itself needs a browser's dialog)
  reset([prop('pixal3d')], status());
  assert.equal(await agreeToHi3d(), true);
  reset([prop('pixal3d')], status({ ...missing, downloading: true }));
  assert.equal(await agreeToHi3d(), true);

  const posted = [];
  // the status poll keeps asking every few seconds: its slow timers stay off, this test is its poll
  const { setTimeout: realTimeout } = globalThis;
  globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 1000 ? 0 : realTimeout(fn, ms, ...rest));
  globalThis.fetch = async (path, init = {}) => {
    if (init.method === 'POST') posted.push(path);
    return {
      status: 200,
      text: async () => JSON.stringify(path === '/api/hi3d/download' ? { ...missing, downloading: true } : {}),
    };
  };
  const st = status(missing);
  const { note, update } = mount(prop('hi3d'), st);
  assert.equal(note.hidden, false);
  assert.ok(note.textContent.startsWith('Hi3D needs a one-time 11.4 GB download of its AI models.'));
  const [button] = note.all((e) => e.tagName === 'BUTTON');
  assert.equal(button.textContent, 'Download (11.4 GB)');
  button.click();
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.deepEqual(posted, ['/api/hi3d/download'], 'the hub starts the download');
  assert.equal(button.hidden, true, 'the hub\'s answer says it runs');
  store.status.hi3d = { ...missing, downloading: true, done_bytes: HI3D_BYTES / 2 };
  update();
  assert.match(note.textContent, /Downloading Hi3D’s AI models \(11\.4 GB\)… 50%/);
  store.status.hi3d = { ...missing, available: true, finishing: true, done_bytes: HI3D_BYTES };
  update();
  assert.match(note.textContent, /Measuring its convolutions/);
  store.status.hi3d = { ...missing, available: true, done_bytes: HI3D_BYTES };
  update();
  assert.equal(note.hidden, true, 'here now');
  // set to Pixal3D: nothing to say about Hi3D's models
  store.status.hi3d = missing;
  store.project.props[0].mesh.settings.geometry = 'pixal3d';
  update();
  assert.equal(note.hidden, true);
  MESH.unmount();
  await new Promise((resolve) => realTimeout(resolve, 600));    // the status poll it asked for goes out
  globalThis.setTimeout = realTimeout;
});

test('a tileable texture is no object: the 3D stage says it gets no 3D model and offers to make none', () => {
  const texture = (settings) => ({ ...prop('pixal3d'), images: { ...prop('pixal3d').images, settings } });
  const ticked = texture({ tileable: true, qwen_model: 'turbo' });
  const { host } = mount(ticked);
  assert.deepEqual(MESH.status(ticked), { state: 'unreachable', message: 'A tileable texture: it gets no 3D model' });
  assert.match(host.textContent, /A tileable texture.*It gets no 3D model: “Generate 3D” passes it by\. To make one from this picture, untick Tileable in Images\./);
  const make = () => host.all((e) => e.tagName === 'BUTTON' && /^Make 3D/.test(e.textContent)).map((b) => b.hidden);
  assert.deepEqual(make(), [true]);
  // Unticked, or drawn by FLUX.3 (which draws no tileable picture), the prop is an object like any other.
  for (const settings of [{ tileable: false }, { tileable: true, qwen_model: 'flux3' }, {}]) {
    const object = texture(settings);
    assert.equal(MESH.status(object).message, 'Not made into 3D yet', JSON.stringify(settings));
    store.project.props = [object];
    MESH.update({ selectedId: 'p_1', stageId: 'mesh', project: store.project, boxes: store.project.props, select() {}, goto() {}, moving: true, settled: new Promise(() => {}) });
    assert.deepEqual(make(), [false], JSON.stringify(settings));
    assert.match(host.textContent, /Ready to become 3D/);
  }
  // A model it had before the tick stays its own: nothing is taken away, nothing new is made.
  const made = { ...texture({ tileable: true }), mesh: prop('pixal3d', 'pixal3d').mesh };
  assert.equal(MESH.status(made).message, 'Has its 3D model');
  MESH.unmount();
});

test('every setting of the 3D panel is on one line, its control beside its name; the quality reads 1K and 1.5K', () => {
  const { host } = mount(prop('pixal3d'));
  const rows = host.byClass('field-row').map((row) => [row.children[0].textContent,
    row.children.slice(1).map((c) => (c.classList.contains('seg') ? c.all((e) => e.tagName === 'BUTTON').map((b) => b.textContent).join('|')
      : c.classList.contains('field-value') ? 'value' : c.tagName === 'INPUT' ? c.getAttribute('type') : 'seed'))]);
  assert.deepEqual(rows, [
    ['Shape', ['Hi3D|Pixal3D']],
    ['Quality', ['Standard 1K|High 1.5K']],
    ['High-poly detail', ['range', 'value']],
    ['Texture size', ['1K|2K|4K']],
    ['Seed', ['seed']],
  ]);
  for (const row of host.byClass('field-row')) assert.equal(row.children[0].classList.contains('field-label'), true);
  assert.equal(host.byClass('field-value')[0].textContent, '300k tris');
  // Hi3D's credit keeps a line of its own under the Shape, inside its field.
  const [shape] = host.byClass('sv-shape');
  assert.deepEqual(shape.children.map((c) => c.className.split(' ')[0]), ['field-row', 'chip']);
  MESH.unmount();
});
