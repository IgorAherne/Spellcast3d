// The image generator in the page (no browser): which one runs (the one spellcast3d-config.ini picks, as the
// hub says in /api/status), the Images gate when that file has a mistake, the controls each image
// generator has and the fields a prop's workbench shows for it (Qwen-Image 2.1: no "Avoid", its own steps),
// the example pictures (several for Qwen-Image 2.1, their routes, "Paste (Ctrl+V)"), Qwen's "Research licence"
// chip and line, the row of a prop's pictures (the mark of the chosen one, the × and its Ctrl+Z), and the words
// of the status line, the install card and Diagnostics.
//   node --test code/tests/web/images.test.mjs
// (code/tests/test_web.py runs every code/tests/web/*.test.mjs when Node.js is installed)
import assert from 'node:assert/strict';
import { test } from 'node:test';

// ---- a page, as far as these modules use one: elements with attributes, children, classes and text ----

class FakeNode {
  getBoundingClientRect() { return { left: 0, top: 0, width: 0, height: 0 }; }   // nothing is laid out here
}
class FakeEl extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attrs = {};
    this.className = '';
    this.style = { setProperty(k, v) { this[k] = v; } };
    this.listeners = {};
    this.hidden = false;
  }

  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  get firstChild() { return this.children[0] ?? null; }
  get textContent() { return this.children.map((c) => (typeof c === 'string' ? c : c.textContent)).join(''); }
  set textContent(v) { this.children = [String(v)]; }
  get title() { return this.attrs.title ?? ''; }
  set title(v) { this.attrs.title = String(v); }
  get options() { return this.children.filter((c) => c instanceof FakeEl && c.tagName === 'OPTION'); }
  animate() { return { finished: Promise.resolve(), cancel() {} }; }
  decode() { return Promise.resolve(); }      // (an <img>: its picture is there)
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

  click() { for (const fn of this.listeners.click || []) fn({ preventDefault() {}, stopPropagation() {} }); }
  /** Every element in it (itself first) that `pick` accepts. */
  all(pick) {
    const out = pick(this) ? [this] : [];
    for (const c of this.children) if (c instanceof FakeEl) out.push(...c.all(pick));
    return out;
  }

  byClass(name) { return this.all((e) => e.classList.contains(name)); }
  /** The first element in it with the class a selector like ".field-label" names (all the page asks for here). */
  querySelector(sel) { return this.byClass(sel.slice(1))[0] ?? null; }
  /** Every element in it (itself first) that shows - it and what it is in are not hidden - and that `pick` accepts. */
  showing(pick) {
    if (this.hidden) return [];
    const out = pick(this) ? [this] : [];
    for (const c of this.children) if (c instanceof FakeEl) out.push(...c.showing(pick));
    return out;
  }
}
class FakeTemplate extends FakeEl {
  set innerHTML(html) { this.html = html; }
  get content() { return { firstChild: new FakeEl('svg') }; }
}
globalThis.Node = FakeNode;
globalThis.addEventListener ??= () => {};
globalThis.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} };   // the slicer watches its bar's size
globalThis.document = {
  addEventListener: () => {}, removeEventListener: () => {}, hidden: false, activeElement: null, documentElement: { dataset: {} },
  querySelector: () => null,
  createElement: (tag) => (tag === 'template' ? new FakeTemplate(tag) : new FakeEl(tag)),
  createElementNS: (ns, tag) => new FakeEl(tag),
};

const STATIC = new URL('../../spellcast/static/js/', import.meta.url);
const load = (path) => import(new URL(path, STATIC));

const {
  store, STAGE_TOOL, TOOL_NAMES, configChanged, configProblems, imageTool, jobFor, stageNoun, stageTool,
} = await load('store.js');
const { gateLine, gateOpen, toolGate } = await load('stages/common.js');
const {
  kommunityLicence, kommunityText, licenseChip, licenseLine, licenseOf, nonCommercial, researchLicence, researchText,
} = await load('licenses.js');
const { headline } = await load('shell/statusline.js');
const { hubJob } = await import('./hub_job.mjs');
const { addReferences, removeReferences } = await load('actions.js');
const { TOOLS: DIAGNOSTICS } = await load('shell/settings.js');
const { NAMES: INSTALL_NAMES, ORDER: INSTALL_ORDER } = await load('shell/install.js');
const {
  CONTROLS, FLUX3_NEEDS_ENGINE, MAX_REFS, OPTIONAL_MODELS, QWEN_MODELS, QWEN_REFERENCE_STEPS, QWEN_STEPS,
  QWEN_STEPS_HINT, controlsOf, livePreview, modeOf, modesFor, optionalModelState, previewBox, previewParts, qwenModelOf,
  refsOf, refsToAdd, shiftedCanvas, shownControls, default: IMAGES,
} = await load('stages/images.js');
const { givenFolder } = await load('stages/given.js');

const QWEN_LICENSE = 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE';
const QWEN_STATEMENT = 'https://x.com/QwenDevs/status/2101917379785838660';
const STATEMENT = 'The pictures you make are yours, also for commercial work (Qwen, 21 Sep 2026).';
const OTHERS = ['pixal3d', 'imesh', 'lato2', 'triflow'];

/** The hub's /api/status with the image generator `image` (null: spellcast3d-config.ini has a mistake). */
function status(image, { problems = [], states = {}, license = undefined, staged = true, changed = null } = {}) {
  const tools = Object.fromEntries([...(image ? [image] : []), ...OTHERS].map((t) => [t, {
    state: states[t] || (t === image ? 'ready' : 'stopped'), message: '', installed: true, supported: true, uses_gpu: t !== 'imesh',
    stage: t === image ? 'images' : t === 'pixal3d' ? 'mesh' : t === 'imesh' ? 'retopo' : 'retopo',
  }]));
  if (image === 'qwen' && license !== undefined) tools.qwen.license = license;
  return {
    tools,
    ...(staged ? { stage_tools: { images: image ? [image] : [], mesh: ['pixal3d'], retopo: ['imesh', 'lato2', 'triflow'], projection: ['imesh'] } } : {}),
    config: { image_backend: image, quality: image === 'qwen' ? 'Q4_K_M' : null, problems, changed },
    install: { running: false, components: {} },
  };
}

function reset(st) {
  store.connected = true;
  store.jobs = [];
  store.jobErrors = {};
  store.status = st;
  store.project = { id: 'Props', name: 'Props', props: [{ id: 'p_1', name: 'Barrel', images: { chosen: 'images/0001.png' } }] };
  store.bust = {};
  store.openStamp = 7;
}

test('Images says what it wants loaded - the picked model, the card taken from an idle tool - and a pick asks for the new one at once', async () => {
  const ctx = { selectedId: 'w_1', select() {}, goto() {} };
  const { setTimeout: realTimeout } = globalThis;
  globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 1000 ? 0 : realTimeout(fn, ms, ...rest));   // no polling after
  const sent = [];
  globalThis.fetch = async (path, init = {}) => {
    if (init.method && init.method !== 'GET') sent.push([init.method, path, init.body]);
    return { status: 200, text: async () => (path.endsWith('/warm') ? '{"warmed": true, "state": "stopping"}' : '{}') };
  };
  const warms = () => sent.filter(([, path]) => path.endsWith('/warm'));
  try {
    const st = status('qwen', { states: { qwen: 'starting', pixal3d: 'ready' } });
    st.qwen_full = { available: true, downloading: false, done_bytes: 1, total_bytes: 1, error: null };
    reset(st);
    store.project.props = [{ id: 'w_1', name: 'Barrel', images: { settings: { prompt: 'a barrel', qwen_model: 'full' } } }];
    const host = new FakeEl('div');
    IMAGES.mount(host, ctx);
    await new Promise((r) => realTimeout(r, 0));
    assert.deepEqual(sent, [], 'showing asks nothing by itself: the shell does, once per arrival, with this wish');
    assert.deepEqual(IMAGES.warmWish(), { tool: 'qwen', displace: true, model: 'full', stage: 'images' }, 'the model the prop on screen draws with, in Images\' arithmetic');
    // The full model is still loading and the artist picks the Turbo: the hub is asked for it at once.
    const [seg] = host.byClass('sv-model')[0].byClass('seg');
    seg.children[0].click();
    await new Promise((r) => realTimeout(r, 0));
    assert.deepEqual(warms(), [['POST', '/api/tools/qwen/warm', '{"displace":true,"model":"turbo","stage":"images"}']]);
    assert.equal(store.status.tools.qwen.state, 'stopping', 'the page shows what the hub answered');
    assert.deepEqual(IMAGES.warmWish(), { tool: 'qwen', displace: true, model: 'turbo', stage: 'images' });
    IMAGES.unmount();
    // Forge Neo has no model of the launcher's to name; without an image generator there is nothing to ask.
    reset(status('forge', { states: { forge: 'stopped' } }));
    store.project.props = [{ id: 'w_1', name: 'Barrel', images: { settings: { prompt: 'a barrel' } } }];
    IMAGES.mount(new FakeEl('div'), ctx);
    assert.deepEqual(IMAGES.warmWish(), { tool: 'forge', displace: true });
    IMAGES.unmount();
    reset(status(null));
    assert.equal(IMAGES.warmWish(), null);
  } finally {
    globalThis.setTimeout = realTimeout;
  }
});

test('Images runs on the image generator spellcast3d-config.ini picks: the hub says which', () => {
  reset(null);
  assert.equal(imageTool(), 'qwen', 'before the hub has answered: the default');
  reset(status('qwen'));
  assert.equal(imageTool(), 'qwen');
  assert.equal(STAGE_TOOL.images, 'qwen');
  assert.equal(stageTool('images', store.project.props[0]), 'qwen');
  assert.equal(stageNoun('images'), 'Qwen-Image 2.1', 'named: there are two image generators');
  reset(status('forge'));
  assert.equal(imageTool(), 'forge');
  assert.equal(STAGE_TOOL.images, 'forge', 'every stage view follows (gates, loading words, the status line)');
  assert.equal(stageTool('images', store.project.props[0]), 'forge');
  reset(status('forge', { staged: false }));
  assert.equal(imageTool(), 'forge', 'a hub from before stage_tools: the image generator it has');
  reset(status(null, { problems: ['spellcast3d-config.ini, line 6: "forg" is not an image generator.'] }));
  assert.equal(imageTool(), null, 'a mistake in the file: none');
  assert.equal(STAGE_TOOL.mesh, 'pixal3d');
  assert.equal(stageTool('retopo', { retopo: { settings: { method: 'lato2' } } }), 'lato2', 'Retopo still follows the prop');
});

test('a mistake in spellcast3d-config.ini closes the Images gate: a short headline, what to do, the file', async () => {
  const problem = 'spellcast3d-config.ini, line 6: "forg" is not an image generator Spellcast3D knows (write backend = qwen '
    + 'or backend = forge).';
  reset(status(null, { problems: [problem] }));
  const gate = toolGate('images');
  const fix = 'Fix it, save the file, then close Spellcast3D and start it again.';
  assert.deepEqual([gate.kind, gate.text, gate.detail],
    ['config', 'No image generator: spellcast3d-config.ini has a mistake.', `${problem} ${fix}`]);
  assert.equal(gateOpen(gate), false, 'Generate waits for the file to be fixed');
  assert.equal(toolGate('mesh').kind, 'ok', 'the other stages go on');
  const line = gateLine('images', { config: 'Import images works without it.' });
  line.sync();
  assert.equal(line.hidden, false);
  assert.match(line.className, /\bconfig\b/);
  assert.equal(line.textContent, `No image generator: spellcast3d-config.ini has a mistake. ${problem} ${fix} `
    + 'Import images works without it. Show the file');
  const [small] = line.all((e) => e.tagName === 'SMALL');
  assert.equal(small.textContent, ` ${problem} ${fix} Import images works without it.`, 'what to do, under the headline');
  const calls = [];
  globalThis.fetch = async (path, init = {}) => {
    calls.push([init.method, path, init.body]);
    return { status: 200, text: async () => '{"ok": true, "opened": true}' };
  };
  const [show] = line.all((e) => e.tagName === 'A');
  show.click();
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(calls.at(-1), ['POST', '/api/open-folder', JSON.stringify({ which: 'config' })],
    'File Explorer on Spellcast3D\'s folder with the file selected');
  reset(status(null));
  assert.match(toolGate('images').detail, /backend = qwen or backend = forge/, 'no words from the hub: the page says what to write');
  // An unknown model size: Qwen runs with the default, and the gate line says what was wrong.
  const size = 'spellcast3d-config.ini, line 9: "Q9" is not a Qwen model size; Q4_K_M is used.';
  reset(status('qwen', { problems: [size] }));
  assert.deepEqual(configProblems(), [size]);
  assert.equal(toolGate('images').kind, 'ok');
  line.sync();
  assert.deepEqual([line.hidden, line.textContent], [false, `${size} Show the file`]);
  assert.match(line.className, /\bconfig\b/);
  reset(status('qwen'));
  line.sync();
  assert.equal(line.hidden, true, 'nothing to say');
  reset(status('qwen', { states: { qwen: 'starting' } }));
  line.sync();
  assert.equal(line.textContent, 'Loading Qwen-Image 2.1… (this can take a minute)');
});

test('the file changed while Spellcast3D runs: the Images gate says it quietly', () => {
  const changed = 'spellcast3d-config.ini now picks Forge Neo. Close Spellcast3D and start it again to switch (it installs '
    + 'what that needs; Qwen-Image 2.1 runs until then).';
  reset(status('qwen', { changed }));
  assert.equal(configChanged(), changed);
  const line = gateLine('images');
  line.sync();
  assert.deepEqual([line.hidden, line.textContent], [false, changed], 'no file link: there is nothing to fix');
  assert.match(line.className, /\bconfig\b/);
  reset(status('qwen'));
  assert.equal(configChanged(), '');
  line.sync();
  assert.equal(line.hidden, true);
});

test('Qwen-Image 2.1 installing: the gate and the status line say how much is downloaded', () => {
  const st = status('qwen');
  st.tools.qwen.installed = false;
  st.install = { running: true, current: 'qwen', components: { qwen: {
    status: 'running', fraction: 0.27, step: 'Downloading its AI models (10.4 GB)', bytes_done: 2.9 * 1024 ** 3,
    bytes_total: 10.4 * 1024 ** 3 } } };
  reset(st);
  const line = gateLine('images', { installing: 'Generate waits for it.' });
  line.sync();
  assert.equal(line.textContent, 'Qwen-Image 2.1 is installing… 27% · 2.9 GB of 10.4 GB Generate waits for it.');
  assert.equal(headline().text, 'Installing Qwen-Image 2.1 · 2.9 GB of 10.4 GB');
});

test('each image generator shows its own controls; both share the words, shape, variations and seed', () => {
  const shared = ['prompt', 'shape', 'variations', 'seed'];
  const forge = controlsOf('forge');
  const qwen = controlsOf('qwen');
  for (const c of shared) assert.ok(forge.has(c) && qwen.has(c), `${c} is in both`);
  // "Avoid" is Forge Neo's: Qwen-Image 2.1 is sampled without guidance, where a negative prompt is never read.
  for (const c of ['avoid', 'model', 'steps', 'cfg', 'sampler', 'example', 'follow']) {
    assert.ok(forge.has(c), `Forge Neo has ${c}`);
    assert.ok(!qwen.has(c), `Qwen-Image 2.1 has no ${c}`);
  }
  for (const c of ['tool', 'qwenModel', 'qwenSteps', 'transparent', 'firstBlockCache', 'examples']) {
    assert.ok(qwen.has(c), `Qwen-Image 2.1 has ${c}`);
    assert.ok(!forge.has(c), `Forge Neo has no ${c}`);
  }
  assert.deepEqual([...controlsOf(null)], [...shared, 'examples'],
    'no image generator: the shared controls, the example pictures laid out as for the default one');
  assert.deepEqual(Object.keys(CONTROLS).sort(), ['forge', 'qwen']);
});

test('Qwen-Image 2.1\'s steps: 25 for a prop that has none saved, and the hint names Qwen\'s own 40', () => {
  assert.deepEqual([QWEN_STEPS, QWEN_REFERENCE_STEPS], [25, 40]);
  assert.equal(QWEN_STEPS_HINT, '25 suits most pictures; Qwen’s own reference is 40 (slower, sometimes finer detail).');
});

test('a prop\'s workbench: Qwen-Image 2.1 has no "Avoid" and no guidance; Forge Neo has both, with the words saved', () => {
  const ctx = { selectedId: 'w_1', select() {}, goto() {} };
  // the full model: its own Steps show (with the Turbo they hide, see the next test)
  const settings = { prompt: 'a barrel', negative_prompt: 'blurry, text', qwen_model: 'full' };
  const open = (tool) => {
    reset(status(tool));
    store.project.props = [{ id: 'w_1', name: 'Barrel', images: { settings } }];
    const host = new FakeEl('div');
    IMAGES.mount(host, ctx);
    return host;
  };
  /** A field's label: "Steps" (a slider's label also holds its value). */
  const labelOf = (f) => {
    const [label] = f.querySelector('.field-label').children;
    return typeof label === 'string' ? label : label.textContent;
  };
  const fields = (host) => host.showing((e) => e.classList.contains('field')).map(labelOf);
  const stepsOf = (host) => host.showing((e) => e.classList.contains('field') && labelOf(e) === 'Steps');
  // What is written to the hub.
  const sent = [];
  globalThis.fetch = async (path, init = {}) => {
    if (init.method !== 'GET') sent.push([init.method, path]);
    return { status: 200, text: async () => '{}' };
  };

  const host = open('qwen');
  assert.deepEqual(fields(host), ['Resolution', 'Shape', 'Variations', 'Steps', 'Seed'],
    'Resolution in the panel; Shape and Variations under "More", with Steps and Seed');
  assert.deepEqual(host.showing((e) => e.classList.contains('field-row')).map((row) => [row.children[0].textContent, row.children.length]),
    [['Resolution', 2], ['Shape', 2], ['Variations', 2], ['Seed', 2]], 'each on one line, its control beside its name');
  const [resolution] = host.showing((e) => e.classList.contains('field') && labelOf(e) === 'Resolution');
  const sizes = resolution.byClass('seg-btn');
  assert.deepEqual(sizes.map((b) => b.textContent), ['1K', '1.5K', '2K']);
  assert.deepEqual(sizes.map((b) => b.title.split(':')[0]), ['1024 × 1024 pixels', '1536 × 1536 pixels', '2048 × 2048 pixels'],
    'each says the pixels it gives the Shape picked');
  assert.match(sizes[2].title, /4 times the pixels of 1K/);
  sizes[1].click();
  assert.equal(store.project.props[0].images.settings.resolution, 1.5);
  const [shapes] = host.byClass('sv-shapes');
  assert.deepEqual(shapes.byClass('seg-btn').map((b) => b.title),
    ['Square · 1536 × 1536', 'Portrait · 1248 × 1824', 'Landscape · 1824 × 1248'], 'each Shape says its pixels at that Resolution');
  shapes.byClass('seg-btn')[1].click();
  assert.equal(sizes[2].title.split(':')[0], '1664 × 2432 pixels', 'the Shape stays as it is kept (832 × 1216), 2K doubles it');
  assert.deepEqual([store.project.props[0].images.settings.width, store.project.props[0].images.settings.height], [832, 1216]);
  // The pointer on a Render button rings the Resolution once: what the pictures are about to be rendered at.
  const rung = [];
  const { animate } = FakeEl.prototype;
  FakeEl.prototype.animate = function (frames, timing) { rung.push([this, timing?.iterations]); return animate.call(this); };
  try {
    const seg = resolution.byClass('seg')[0];
    for (const button of host.byClass('sv-make')) for (const fn of button.listeners.pointerenter || []) fn({});
    assert.deepEqual(rung.map(([el, times]) => [el === seg, times]), [[true, 1], [true, 1], [true, 1]], 'each of the three Render buttons, once');
  } finally {
    FakeEl.prototype.animate = animate;
  }
  // Where each control sits: Shape, Variations and Transparent background under "More"; the rest in the panel.
  const fold = (el) => host.all((e) => e.tagName === 'DETAILS').find((d) => d.all((e) => e === el).length)?.children[0].textContent;
  const fieldOf = (name) => host.all((e) => e.classList.contains('field') && labelOf(e) === name)[0];
  const switchOf = (name) => host.all((e) => e.classList.contains('switch')).find((e) => e.textContent.includes(name));
  assert.deepEqual(['Shape', 'Variations', 'Seed'].map((name) => fold(fieldOf(name))), ['More', 'More', 'More']);
  assert.equal(fold(switchOf('Transparent background')), 'More');
  assert.equal(fold(fieldOf('Resolution')), undefined);
  assert.equal(fold(switchOf('Faster image generation')), undefined, 'in the panel, where Transparent background was');
  const [prompt] = host.byClass('sv-prompt');
  const [avoid] = host.byClass('sv-avoid');
  assert.ok(prompt.classList.contains('alone'), 'the words take the room "Avoid" had (stages-images.css)');
  const [steps, ...more] = stepsOf(host);
  assert.equal(more.length, 0, 'one Steps slider: its own');
  assert.equal(steps.querySelector('.field-value').textContent, '25');
  assert.equal(steps.querySelector('.hint').textContent, QWEN_STEPS_HINT);

  // spellcast3d-config.ini picks Forge Neo: "Avoid" is there, with the words the prop has saved.
  store.status = status('forge');
  IMAGES.update(ctx);
  assert.deepEqual(fields(host), ['Avoid', 'Shape', 'Variations', 'Steps', 'Guidance (CFG)', 'Sampler', 'Scheduler', 'Seed'],
    'Forge Neo draws at the Shape\'s own size: no Resolution');
  assert.equal(avoid.value, 'blurry, text');
  assert.ok(!prompt.classList.contains('alone'));
  assert.match(stepsOf(host)[0].querySelector('.hint').textContent, /Lightning/, 'Forge Neo\'s own steps');

  // No image generator (a mistake in the file): laid out as for the default one, Qwen-Image 2.1.
  store.status = status(null, { problems: ['spellcast3d-config.ini, line 6: "forg" is not an image generator.'] });
  IMAGES.update(ctx);
  assert.deepEqual(fields(host), ['Shape', 'Variations', 'Seed']);
  assert.ok(prompt.classList.contains('alone'));

  // Hidden, "Avoid" is only kept: nothing was written, the prop has its words for Forge Neo.
  assert.equal(avoid.value, 'blurry, text');
  assert.equal(store.project.props[0].images.settings.negative_prompt, 'blurry, text');
  assert.deepEqual(sent, [], 'no settings were sent');
  IMAGES.unmount();

  // A prop with its own number of steps shows it.
  settings.qwen_steps = 40;
  const again = open('qwen');
  assert.equal(stepsOf(again)[0].querySelector('.field-value').textContent, '40');
  IMAGES.unmount();
});

test('Qwen-Image 2.1\'s two models: Turbo by default, Full after its download; Steps and Faster drawing only with Full', async () => {
  assert.deepEqual(QWEN_MODELS.map((m) => [m.value, m.label]), [['turbo', 'Turbo\n(fast, 8 steps)'], ['full', 'Full\n(25 steps)'],
    ['flux3', 'FLUX.3 Klein\n(16 steps)']]);
  assert.equal(qwenModelOf({}), 'turbo', 'a prop from before the choice');
  assert.equal(qwenModelOf(undefined), 'turbo');
  assert.equal(qwenModelOf({ qwen_model: 'full' }), 'full');
  assert.equal(qwenModelOf({ qwen_model: 'fast' }), 'turbo');
  for (const name of ['qwenSteps', 'firstBlockCache']) {
    assert.ok(!shownControls('qwen', 'turbo').has(name), `the Turbo has no ${name}: it has its own`);
    assert.ok(shownControls('qwen', 'full').has(name), `the full model has ${name}`);
  }
  assert.ok(shownControls('qwen', 'turbo').has('qwenModel') && !shownControls('forge', 'turbo').has('qwenModel'));
  assert.ok(shownControls('forge', 'turbo').has('steps'), 'Forge Neo\'s own steps never hide');

  // The full model as /api/status says: no word yet (an older hub): a plain choice
  assert.deepEqual(optionalModelState('full',undefined), { state: 'unknown', fraction: null, note: '', title: QWEN_MODELS[1].title });
  const total = 4_199_565_024;
  const missing = optionalModelState('full',{ available: false, downloading: false, done_bytes: 0, total_bytes: total, error: null });
  assert.equal(missing.state, 'missing');
  assert.equal(missing.note, 'Download 4.2 GB');
  assert.match(missing.title, /one-time 4\.2 GB download into models\\qwen-image \(the folder opens\)\. Turbo works meanwhile/);
  const going = optionalModelState('full',{ available: false, downloading: true, done_bytes: total * 0.37, total_bytes: total, error: null });
  assert.deepEqual([going.state, going.note, going.fraction.toFixed(2)], ['downloading', 'Downloading… 37%', '0.37']);
  assert.match(going.title, /^1\.6 GB of 4\.2 GB into models\\qwen-image/);
  const stopped = optionalModelState('full',{ available: false, downloading: false, done_bytes: total / 2, total_bytes: total, error: null });
  assert.equal(stopped.note, 'Continue (50%)', 'pressed again, it continues where it stopped');
  const failed = optionalModelState('full',{ available: false, downloading: false, done_bytes: 0, total_bytes: total, error: 'The drive is full.' });
  assert.deepEqual([failed.state, failed.note, failed.title], ['failed', 'Try again', 'The drive is full.']);
  assert.equal(optionalModelState('full',{ available: true, downloading: false, done_bytes: total, total_bytes: total, error: null }).state,
    'available');

  // The workbench: a new prop on the Turbo, no Steps, no "Faster drawing"; Full's button offers the download
  const ctx = { selectedId: 'w_1', select() {}, goto() {} };
  const st = status('qwen');
  st.qwen_full = { available: false, downloading: false, done_bytes: 0, total_bytes: total, error: null };
  reset(st);
  store.project.props = [{ id: 'w_1', name: 'Barrel', images: { settings: { prompt: 'a barrel' } } }];
  const sent = [];
  globalThis.fetch = async (path, init = {}) => {
    if (init.method !== 'GET') sent.push([init.method, path, init.body]);
    return { status: 200, text: async () => '{}' };
  };
  // the status poll the download asks for (1.5 s) stays off: this test is its poll
  const { setTimeout: realTimeout } = globalThis;
  globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 1000 ? 0 : realTimeout(fn, ms, ...rest));
  const host = new FakeEl('div');
  IMAGES.mount(host, ctx);
  const [box] = host.byClass('sv-model');
  const [seg] = box.byClass('seg');
  const [turbo, full] = seg.children;
  const switches = () => host.showing((e) => e.classList.contains('switch-title')).map((e) => e.textContent);
  const steps = () => host.showing((e) => e.classList.contains('field-label')).map((e) => e.textContent).filter((t) => t.startsWith('Steps'));
  assert.equal(box.hidden, false);
  assert.ok(turbo.classList.contains('on') && !full.classList.contains('on'), 'Turbo, the default');
  assert.equal(turbo.textContent, 'Turbo\n(fast, 8 steps)');
  assert.equal(full.textContent, 'Full\n(25 steps)Download 4.2 GB', 'its download under its name');
  assert.ok(full.classList.contains('sv-get'));
  assert.deepEqual(switches(), ['Faster image generation', 'Transparent background', 'Tileable'], 'no "Faster drawing" with the Turbo');
  assert.deepEqual(steps(), [], 'no Steps with the Turbo');
  const [bar] = box.byClass('progress');
  assert.equal(bar.hidden, true);

  // pressed: the download starts (the hub opens the folder); the prop stays on the Turbo meanwhile
  full.click();
  await new Promise((resolve) => realTimeout(resolve, 20));
  assert.deepEqual(sent.map(([method, path]) => [method, path]), [['POST', '/api/qwen/full/download']]);
  assert.ok(turbo.classList.contains('on') && !full.classList.contains('on'));
  assert.equal(store.project.props[0].images.settings.qwen_model, undefined, 'nothing picked');
  st.qwen_full = { available: false, downloading: true, done_bytes: total / 4, total_bytes: total, error: null };
  IMAGES.update(ctx);
  assert.equal(full.textContent, 'Full\n(25 steps)Downloading… 25%');
  assert.equal(bar.hidden, false);
  assert.equal(full.title.startsWith('1.0 GB of 4.2 GB'), true);

  // there: Full is a choice like Turbo; picked, Steps and "Faster drawing" show
  st.qwen_full = { available: true, downloading: false, done_bytes: total, total_bytes: total, error: null };
  IMAGES.update(ctx);
  assert.equal(full.textContent, 'Full\n(25 steps)');
  assert.equal(bar.hidden, true);
  full.click();
  assert.equal(store.project.props[0].images.settings.qwen_model, 'full');
  assert.ok(full.classList.contains('on'));
  assert.deepEqual(switches(), ['Faster image generation', 'Faster drawing (First-Block Cache)', 'Transparent background', 'Tileable']);
  assert.equal(steps().length, 1);
  turbo.click();
  assert.equal(store.project.props[0].images.settings.qwen_model, 'turbo');
  assert.deepEqual(steps(), []);
  IMAGES.unmount();

  // no word about the full model from the hub yet: the page works, Full can be picked (the hub says if it is missing)
  reset(status('qwen'));
  store.project.props = [{ id: 'w_1', name: 'Barrel', images: { settings: { prompt: 'a barrel', qwen_model: 'full' } } }];
  const plain = new FakeEl('div');
  IMAGES.mount(plain, ctx);
  const [, fullPlain] = plain.byClass('sv-model')[0].byClass('seg')[0].children;
  assert.ok(fullPlain.classList.contains('on') && fullPlain.textContent === 'Full\n(25 steps)');
  // its button says the steps its slider under More is at
  const [slider] = plain.all((e) => e.classList.contains('field') && e.querySelector('.field-label')?.textContent.startsWith('Steps')
    && !e.hidden).map((f) => f.input);
  slider.value = 32;
  for (const fn of slider.listeners.input) fn();
  assert.equal(fullPlain.textContent, 'Full\n(32 steps)');
  IMAGES.unmount();
  await new Promise((resolve) => realTimeout(resolve, 600));     // the settings' own save (0.5 s) goes out
  globalThis.setTimeout = realTimeout;
});

const FLUX3_LICENSE = 'https://huggingface.co/unsloth/flux-3-image-klein-GGUF/blob/12ef13ce920d53f45fe805093910c899668b9e76/LICENSE.md';
const FLUX3_TERMS = 'FLUX.3 Image Klein is under the FLUX Kommunity License: free for non-commercial use; commercial use '
  + 'of the pictures only for creators and companies under US$5M yearly revenue.';

test('FLUX.3 Image Klein, the third model: disabled until the engine draws it, then a download like Full; its own steps, no transparent background, its licence chip', async () => {
  const flux3 = QWEN_MODELS[2];
  assert.equal(flux3.value, 'flux3');
  assert.match(flux3.title, /16 steps\. It follows long, descriptive prompts best\./, 'BFL: full scene captions');
  assert.equal(qwenModelOf({ qwen_model: 'flux3' }), 'flux3');
  for (const name of ['qwenSteps', 'firstBlockCache', 'transparent']) {
    assert.ok(!shownControls('qwen', 'flux3').has(name), `FLUX.3 has no ${name}`);
  }
  for (const name of ['examples', 'qwenModel', 'tool', 'prompt']) assert.ok(shownControls('qwen', 'flux3').has(name));
  assert.ok(shownControls('qwen', 'turbo').has('transparent') && shownControls('qwen', 'full').has('transparent'));
  assert.deepEqual(Object.keys(OPTIONAL_MODELS), ['full', 'flux3']);

  // its state as /api/status "flux3" says: the engine first, then the download as for the full model
  const total = 6_763_437_948;
  const st = (more) => ({ available: false, downloading: false, done_bytes: 0, total_bytes: total, error: null, engine: true, ...more });
  assert.deepEqual(optionalModelState('flux3', st({ engine: false })),
    { state: 'unsupported', fraction: null, note: 'Next update', title: FLUX3_NEEDS_ENGINE });
  assert.equal(FLUX3_NEEDS_ENGINE, 'FLUX.3 Image Klein needs the next update of Spellcast3D’s image engine; it can be picked then.');
  assert.equal(optionalModelState('flux3', st({ engine: false, available: true })).state, 'unsupported',
    'on disk is not enough');
  assert.equal(optionalModelState('flux3', undefined).state, 'unknown');
  const missing = optionalModelState('flux3', st());
  assert.deepEqual([missing.state, missing.note], ['missing', 'Download 6.8 GB']);
  assert.equal(missing.title, 'FLUX.3 Klein is a one-time 6.8 GB download into models\\flux3-klein (the folder opens). '
    + 'Turbo works meanwhile; FLUX.3 Klein can be picked once it is there.');
  assert.match(optionalModelState('flux3', st({ downloading: true, done_bytes: total / 2 })).title,
    /^3\.4 GB of 6\.8 GB into models\\flux3-klein\. Turbo works meanwhile; FLUX\.3 Klein can be picked/);
  assert.equal(optionalModelState('flux3', st({ available: true })).state, 'available');
  assert.deepEqual(optionalModelState('full',st({ engine: undefined })).state, 'missing', 'the full model has no engine gate');

  // the licence: its own chip and line, neither "Non-commercial" nor "Research licence"
  reset(status('qwen'));
  assert.ok(kommunityLicence('flux3') && !nonCommercial('flux3') && !researchLicence('flux3'));
  assert.equal(kommunityText('flux3'), FLUX3_TERMS);
  assert.equal(licenseOf('flux3').url, FLUX3_LICENSE, 'the licence at the revision Spellcast3D downloads');
  const chip = licenseChip('flux3');
  assert.deepEqual([chip.tagName, chip.textContent, chip.title], ['SPAN', 'Kommunity licence', FLUX3_TERMS]);
  assert.ok(chip.classList.contains('kl') && !chip.classList.contains('rl') && !chip.classList.contains('nc'));
  const line = licenseLine('flux3');
  assert.equal(line.textContent, `${FLUX3_TERMS} Licence ↗`);
  assert.equal(line.all((e) => e.tagName === 'A')[0].getAttribute('href'), FLUX3_LICENSE);
  assert.equal(kommunityLicence('qwen'), false);

  // The workbench while the engine cannot draw it: the button is there, disabled, and says why; a click does nothing
  const ctx = { selectedId: 'w_1', select() {}, goto() {} };
  const hub = status('qwen');
  hub.flux3 = st({ engine: false });
  reset(hub);
  store.project.props = [{ id: 'w_1', name: 'Barrel', images: { settings: { prompt: 'a barrel' } } }];
  const sent = [];
  globalThis.fetch = async (path, init = {}) => {
    if (init.method !== 'GET') sent.push([init.method, path]);
    return { status: 200, text: async () => '{}' };
  };
  const { setTimeout: realTimeout } = globalThis;
  globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 1000 ? 0 : realTimeout(fn, ms, ...rest));
  const host = new FakeEl('div');
  IMAGES.mount(host, ctx);
  const [box] = host.byClass('sv-model');
  const [turbo, , klein] = box.byClass('seg')[0].children;
  const switches = () => host.showing((e) => e.classList.contains('switch-title')).map((e) => e.textContent);
  const tool = () => host.byClass('sv-tool-row')[0];
  assert.equal(klein.disabled, true);
  assert.equal(klein.textContent, 'FLUX.3 Klein\n(16 steps)Next update');
  assert.equal(klein.title, FLUX3_NEEDS_ENGINE);
  klein.click();
  await new Promise((resolve) => realTimeout(resolve, 20));
  assert.deepEqual(sent, [], 'no download for a model the engine cannot draw');
  assert.ok(turbo.classList.contains('on') && !klein.classList.contains('on'));
  assert.equal(store.project.props[0].images.settings.qwen_model, undefined);
  assert.equal(tool().byClass('sv-tool-name')[0].textContent, 'Qwen-Image 2.1');

  // the engine draws it, its files are not there: the button offers its download (the hub opens models\flux3-klein)
  hub.flux3 = st();
  IMAGES.update(ctx);
  assert.equal(klein.disabled, false);
  assert.equal(klein.textContent, 'FLUX.3 Klein\n(16 steps)Download 6.8 GB');
  assert.ok(klein.classList.contains('sv-get'));
  klein.click();
  await new Promise((resolve) => realTimeout(resolve, 20));
  assert.deepEqual(sent, [['POST', '/api/flux3/download']]);
  assert.ok(turbo.classList.contains('on'), 'the prop stays on its model meanwhile');
  hub.flux3 = st({ downloading: true, done_bytes: total / 4 });
  IMAGES.update(ctx);
  assert.equal(klein.textContent, 'FLUX.3 Klein\n(16 steps)Downloading… 25%');
  assert.equal(box.byClass('progress')[0].hidden, false);
  hub.flux3 = st({ error: 'The drive is full.' });
  IMAGES.update(ctx);
  assert.equal(box.byClass('sv-model-error')[0].textContent, 'The drive is full.');   // the hub's words name the model

  // there: picked like the others; no Steps, no "Faster drawing", no "Transparent background"; its name and chip
  hub.flux3 = st({ available: true, done_bytes: total });
  IMAGES.update(ctx);
  klein.click();
  assert.equal(store.project.props[0].images.settings.qwen_model, 'flux3');
  assert.ok(klein.classList.contains('on'));
  assert.deepEqual(switches(), ['Faster image generation'], 'no transparent background, no tileable pictures, no First-Block Cache');
  assert.equal(tool().byClass('sv-tool-name')[0].textContent, 'FLUX.3 Image Klein');
  const [kl] = tool().byClass('kl');
  assert.equal(kl.textContent, 'Kommunity licence');
  kl.click();
  assert.equal(host.byClass('sv-terms')[0].textContent, `${FLUX3_TERMS} Licence ↗`);
  // back to the Turbo: Qwen-Image's name and licence, its transparent background
  turbo.click();
  assert.equal(tool().byClass('sv-tool-name')[0].textContent, 'Qwen-Image 2.1');
  assert.equal(tool().byClass('rl')[0].textContent, 'Research licence');
  assert.deepEqual(switches(), ['Faster image generation', 'Transparent background', 'Tileable']);
  IMAGES.unmount();
  await new Promise((resolve) => realTimeout(resolve, 600));     // the settings' own save (0.5 s) goes out
  globalThis.setTimeout = realTimeout;
});

test('the live preview: the newest one of the job drawing this prop, kept between two pictures, gone with the job', () => {
  reset(status('qwen'));
  const job = (progress, status = 'running') => hubJob({
    id: 'j1', stage: 'images', status, prop_ids: ['p_1', 'p_2'], current_prop: 'p_1', progress,
  });
  const first = '/api/jobs/j1/preview?v=1';
  store.jobs = [job({ preview: { url: first, step: 1 } })];
  assert.equal(livePreview(jobFor('images', 'p_1')), first);
  assert.equal(livePreview(jobFor('images', 'p_2'), first), '', 'never on another prop of the job');
  store.jobs = [job({ preview: { url: '/api/jobs/j1/preview?v=2', step: 2 } })];
  assert.equal(livePreview(jobFor('images', 'p_1'), first), '/api/jobs/j1/preview?v=2', 'a new address with every step');
  store.jobs = [job({})];
  assert.equal(livePreview(jobFor('images', 'p_1')), '', 'before the first step');
  assert.equal(livePreview(jobFor('images', 'p_1'), first), first, 'between two pictures of a batch');
  for (const ended of ['done', 'failed', 'cancelled']) {
    store.jobs = [job({ preview: { url: first, step: 1 } }, ended)];
    assert.equal(livePreview(jobFor('images', 'p_1'), first), '', ended);
  }
  store.jobs = [job({ preview: { url: first, step: 1 } }, 'starting_tool')];
  assert.equal(livePreview(jobFor('images', 'p_1'), first), '', 'not drawing yet');
  store.jobs = [];
  assert.equal(livePreview(jobFor('images', 'p_1'), first), '');
  // a prop that still works behind the next one (Omni-Texture): the job's preview is the next prop's, not its own
  const next = '/api/jobs/j1/preview?v=3';
  const behind = { ...job({ preview: { url: next, step: 1, prop: 'p_2' } }), current_prop: 'p_2' };
  behind.props = { p_1: { state: 'working', fraction: 0.9 }, p_2: { state: 'working', fraction: 0.1 } };
  store.jobs = [behind];
  assert.equal(livePreview(jobFor('images', 'p_1'), first, 'p_1'), first, 'its own last one stays');
  assert.equal(livePreview(jobFor('images', 'p_1'), '', 'p_1'), '', 'none shown: none');
  assert.equal(livePreview(jobFor('images', 'p_2'), '', 'p_2'), next, 'the prop on the card');
  assert.equal(livePreview(jobFor('images', 'p_1'), first), next, 'without its prop: as before');
  store.jobs = [job({ preview: { url: next, step: 1 } })];
  assert.equal(livePreview(jobFor('images', 'p_1'), first, 'p_1'), next, 'a preview that names no prop is taken');
});

test('a preview of a window: its box on the picture, cut where it passes a tileable picture\'s edge, the cut part laid at the other side', () => {
  reset(status('qwen'));
  const job = (preview) => hubJob({
    id: 'j1', stage: 'images', status: 'running', prop_ids: ['p_1'], current_prop: 'p_1', progress: { preview },
  });
  const boxOf = (preview) => {
    store.jobs = [job(preview)];
    return previewBox(jobFor('images', 'p_1'));
  };
  assert.deepEqual(boxOf({ url: 'u', step: 1, box: [0.25, 0.5, 0.75, 1] }), [0.25, 0.5, 0.75, 1]);
  for (const box of [undefined, null, [0, 0, 1], [0, 0, Number.NaN, 1], [0.5, 0, 0.5, 1], '0,0,1,1']) {
    assert.equal(boxOf({ url: 'u', step: 1, box }), null, `of all the picture: ${box}`);
  }
  assert.deepEqual(previewParts([0.25, 0.5, 0.75, 1]), [{ part: [0.25, 0.5, 0.75, 1], preview: [0, 0, 1, 1] }]);
  assert.deepEqual(previewParts([0.75, 0, 1.25, 1]), [
    { part: [0.75, 0, 1, 1], preview: [0, 0, 2, 1] },
    { part: [0, 0, 0.25, 1], preview: [-1, 0, 2, 1] },
  ], 'past the right edge: its right half at the left');
  assert.deepEqual(previewParts([0.25, 0, 1.25, 1]), [
    { part: [0.25, 0, 1, 1], preview: [0, 0, 1 / 0.75, 1] },
    { part: [0, 0, 0.25, 1], preview: [-3, 0, 4, 1] },
  ], 'as wide as the picture and moved round its edge: cut too, not laid along all of it');
  assert.deepEqual(previewParts([0, 0.75, 1, 1.25]).map((p) => p.part), [[0, 0.75, 1, 1], [0, 0, 1, 0.25]], 'past the bottom');
  assert.equal(givenFolder('images/0012.png'), 'images/.given/0012');
  assert.equal(givenFolder('images/upload_001.webp'), 'images/.given/upload_001');
});

test('the words go as typed and the live preview always shows: no "3D-ready framing", no switch for the preview', () => {
  const ctx = { selectedId: 'p_1', select() {}, goto() {} };
  const asked = [];
  globalThis.fetch = async (path, init = {}) => {
    asked.push([init.method, path]);
    return { status: 200, text: async () => '{}' };
  };
  store.settings = undefined;
  for (const tool of ['qwen', 'forge']) {
    reset(status(tool));
    store.project.props = [{ id: 'p_1', name: 'Barrel', images: { settings: { prompt: 'a barrel', object_mode: true } } }];
    const host = new FakeEl('div');
    IMAGES.mount(host, ctx);
    const switches = host.showing((e) => e.classList.contains('switch')).map((e) => e.querySelector('.switch-title').textContent);
    assert.deepEqual(switches, tool === 'qwen' ? ['Faster image generation', 'Transparent background', 'Tileable'] : [], tool);
    IMAGES.unmount();
  }
  assert.deepEqual(asked, [['GET', '/api/settings']],
    'the hub\'s settings are asked for once, for Qwen-Image\'s "Faster image generation": the preview is no setting any more');
});

test('Faster image generation sits in the panel while Qwen-Image is the image generator, off until the hub says otherwise', async () => {
  const ctx = { selectedId: 'p_1', select() {}, goto() {} };
  globalThis.fetch = async () => ({ status: 200, text: async () => JSON.stringify({ qwen: { int8_matmul: true } }) });
  try {
    for (const tool of ['qwen', 'forge']) {
      store.settings = undefined;
      reset(status(tool));
      store.project.props = [{ id: 'p_1', name: 'Barrel', images: { settings: { prompt: 'a barrel' } } }];
      const host = new FakeEl('div');
      IMAGES.mount(host, ctx);
      const sw = host.all((e) => e.classList.contains('switch')).find((e) => /Faster image generation/.test(e.textContent));
      assert.equal(host.all((e) => e.tagName === 'DETAILS').some((d) => d.all((e) => e === sw).length), false, 'not under "More"');
      assert.equal(host.showing((e) => e === sw).length === 1, tool === 'qwen', `${tool}: Forge Neo has no such kernels`);
      assert.equal(!!sw.input.checked, false, 'off until the hub\'s settings are read');
      await new Promise((r) => setTimeout(r, 0));
      assert.equal(!!sw.input.checked, tool === 'qwen', 'the hub\'s setting, asked for only with Qwen-Image');
      IMAGES.unmount();
    }
  } finally {
    store.settings = undefined;
  }
});

test('a prop\'s example pictures: several, in order, up to 10; a prop from before keeps its one example', () => {
  assert.equal(MAX_REFS, 10);
  const refs = ['reference/reference_001.png', 'reference/reference_002.webp'];
  assert.deepEqual(refsOf({ references: refs }), refs);
  assert.deepEqual(refsOf({ references: [], reference: 'reference/reference_009.png' }), [], 'the list is what counts');
  assert.deepEqual(refsOf({ reference: 'reference/reference_001.png' }), ['reference/reference_001.png']);
  assert.deepEqual(refsOf({ references: ['reference/reference_001.png', null, '', 3] }), ['reference/reference_001.png']);
  assert.deepEqual(refsOf({}), []);
  assert.deepEqual(refsOf(undefined), []);
  const files = ['a', 'b', 'c', 'd', 'e'];
  assert.deepEqual(refsToAdd(0, files), { take: files, left: 0 });
  assert.deepEqual(refsToAdd(8, files), { take: ['a', 'b'], left: 3 }, 'the first ones that fit');
  assert.deepEqual(refsToAdd(10, files), { take: [], left: 5 });
  assert.deepEqual(refsToAdd(12, ['a']), { take: [], left: 1 });
});

test('Qwen-Image 2.1\'s example pictures: one heading and row, "Add examples" until there is one, "Paste" last', () => {
  const ctx = { selectedId: 'w_1', select() {}, goto() {} };
  const open = (references) => {
    reset(status('qwen'));
    store.project.props = [{ id: 'w_1', name: 'Barrel', images: { settings: { prompt: 'a barrel' }, references } }];
    const host = new FakeEl('div');
    IMAGES.mount(host, ctx);
    return host;
  };
  const ref = (n) => `reference/reference_${String(n).padStart(3, '0')}.png`;

  // None yet: the same heading and row as with pictures (the first one moves nothing under it), a wide tile to add
  // them and "Paste".
  const empty = open([]);
  assert.equal(empty.byClass('sv-refs-count')[0].textContent, `0 of ${MAX_REFS}`);
  const [slot, paste] = empty.byClass('sv-refs-row')[0].children;
  assert.equal(slot.className, 'sv-ref sv-ref-add sv-ref-first');
  assert.equal(slot.textContent, 'Add examplesDrag-drop here or click');
  assert.equal(paste.className, 'sv-ref sv-ref-add sv-ref-paste');
  assert.equal(paste.textContent, 'Paste');
  IMAGES.unmount();

  // With example pictures: a + after them, then "Paste", while there is room.
  const some = open([ref(1), ref(2)]);
  assert.equal(some.byClass('sv-ref-first').length, 0);
  const [row] = some.byClass('sv-refs-row');
  assert.deepEqual(row.children.map((c) => c.className), ['sv-ref', 'sv-ref', 'sv-ref sv-ref-add', 'sv-ref sv-ref-add sv-ref-paste']);
  assert.equal(row.children.at(-1).textContent, 'Paste');
  assert.match(row.children.at(-1).title, /Ctrl\+V/);
  IMAGES.unmount();
  const full = open(Array.from({ length: MAX_REFS }, (_, i) => ref(i + 1)));
  assert.equal(full.byClass('sv-ref-paste').length, 0);
  assert.equal(full.byClass('sv-ref-add').length, 0);
  IMAGES.unmount();
});

test('example pictures go to the hub in one upload, and come off one by one or all at once', async () => {
  reset(status('qwen'));
  const calls = [];
  const prop = { id: 'p_1', name: 'Barrel', images: { chosen: 'images/0001.png', references: ['reference/reference_001.png'] } };
  globalThis.fetch = async (path, init = {}) => {
    calls.push([init.method, path]);
    return { status: 200, text: async () => JSON.stringify(prop) };
  };
  globalThis.FormData = class { constructor() { this.parts = []; } append(...a) { this.parts.push(a); } };
  globalThis.XMLHttpRequest = class {
    constructor() { this.upload = {}; }
    open(method, path) { this.method = method; this.path = path; }
    send(form) {
      calls.push([this.method, this.path, form.parts.map(([field, , name]) => `${field}:${name}`)]);
      this.status = 200;
      this.responseText = JSON.stringify(prop);
      queueMicrotask(() => this.onload());
    }
  };
  const before = store.bust['refs:p_1'];
  await addReferences('p_1', [{ name: 'front.png' }, { name: 'side.png' }]);
  assert.deepEqual(calls.at(-1), ['POST', '/api/projects/Props/props/p_1/images/references', ['files:front.png', 'files:side.png']]);
  assert.notEqual(store.bust['refs:p_1'], before, 'a new picture is never an old one from the cache');
  assert.deepEqual(store.project.props[0].images.references, ['reference/reference_001.png'], 'the hub\'s prop replaces it');
  await removeReferences('p_1', 'reference/reference 002.png');
  assert.deepEqual(calls.at(-1), ['DELETE', '/api/projects/Props/props/p_1/images/references?path=reference%2Freference%20002.png']);
  await removeReferences('p_1');
  assert.deepEqual(calls.at(-1), ['DELETE', '/api/projects/Props/props/p_1/images/references']);
});

test('Qwen-Image 2.1 has a "Research licence", not a non-commercial one: the pictures are the artist\'s', () => {
  reset(status('qwen'));
  assert.equal(nonCommercial('qwen'), false);
  assert.equal(researchLicence('qwen'), true);
  const own = licenseOf('qwen');
  assert.equal(own.url, QWEN_LICENSE, 'the licence at the revision Spellcast3D downloads');
  assert.equal(researchText('qwen'), `Qwen-Image 2.1 is under Qwen’s research licence. ${STATEMENT}`);
  reset(status('qwen', {
    license: { name: 'Qwen Research License (model)', url: QWEN_LICENSE, noncommercial: false, research: true, note: STATEMENT },
  }));
  assert.equal(researchLicence('qwen'), true, 'the hub says the same');
  assert.equal(licenseOf('qwen').note, STATEMENT);
  reset(status('qwen', { license: { name: 'Qwen', url: 'https://example.org/L', research: true, note: 'Theirs.' } }));
  assert.deepEqual([licenseOf('qwen').url, licenseOf('qwen').note], ['https://example.org/L', 'Theirs.'], 'the hub\'s words when it has them');
  reset(status('qwen', { license: { name: 'MIT', noncommercial: false, research: false } }));
  assert.equal(licenseOf('qwen'), null, 'nothing to flag');
  // TriFlow stays non-commercial, and its chip says so.
  reset(status('qwen'));
  assert.equal(researchLicence('triflow'), false);
  assert.equal(licenseChip('triflow').textContent, 'Non-commercial');
  assert.equal(licenseChip('lato2'), null);
});

test('the "Research licence" chip says it on hover, and as a button opens the line with the licence and the way to Forge Neo', () => {
  reset(status('qwen'));
  const chip = licenseChip('qwen');
  assert.equal(chip.tagName, 'SPAN');
  assert.equal(chip.textContent, 'Research licence');
  assert.ok(chip.classList.contains('rl') && !chip.classList.contains('nc'), 'not the warning colour of "Non-commercial"');
  assert.equal(chip.title, `Qwen-Image 2.1 is under Qwen’s research licence. ${STATEMENT}`);
  let clicked = 0;
  const button = licenseChip('qwen', () => { clicked += 1; });
  assert.equal(button.tagName, 'BUTTON');
  assert.match(button.title, /Click for the licence/);
  button.click();
  assert.equal(clicked, 1);
  const line = licenseLine('qwen');
  assert.ok(line.classList.contains('research'));
  assert.equal(line.textContent, `Qwen-Image 2.1 is under Qwen’s research licence. ${STATEMENT} Licence ↗ · Qwen’s statement ↗ · `
    + 'Prefer Forge Neo? Switch the line in spellcast3d-config.ini (Spellcast3D’s folder).');
  const [link, said] = line.all((e) => e.tagName === 'A');
  assert.deepEqual([link.getAttribute('href'), link.getAttribute('target')], [QWEN_LICENSE, '_blank']);
  assert.deepEqual([said.getAttribute('href'), said.getAttribute('target')], [QWEN_STATEMENT, '_blank'],
    'the statement the claim about the pictures rests on');
  // On the Images workbench: the file is one click away (File Explorer, with it selected).
  let shown = 0;
  const withFile = licenseLine('qwen', { showFile: () => { shown += 1; } });
  assert.match(withFile.textContent, /Qwen’s statement ↗ · Prefer Forge Neo\? Show spellcast3d-config\.ini$/);
  withFile.all((e) => e.tagName === 'A').at(-1).click();
  assert.equal(shown, 1);
  assert.equal(licenseLine('forge'), null, 'Forge Neo: nothing to say');
  assert.equal(licenseLine('triflow').textContent, 'TriFlow is free for non-commercial use.License ↗ · MeshLib terms ↗',
    'the links on a line of their own');
});

test('the status line, the install card and Diagnostics know Qwen-Image 2.1', async () => {
  assert.equal(TOOL_NAMES.qwen, 'Qwen-Image 2.1');
  assert.deepEqual(DIAGNOSTICS, ['qwen', 'forge', 'pixal3d', 'imesh', 'lato2', 'triflow']);
  assert.equal(INSTALL_NAMES.qwen, 'Image generator (Qwen-Image 2.1)');
  assert.equal(INSTALL_NAMES.forge, 'Image generator (Forge Neo)');
  assert.ok(INSTALL_ORDER.indexOf('qwen') === INSTALL_ORDER.indexOf('imesh') + 1, 'in the image generator\'s place');
  reset(status('qwen', { states: { qwen: 'starting' } }));
  assert.equal(headline().text, 'Loading Qwen-Image 2.1… (this can take a minute)');
  // The stage's own line says how far a start begun earlier is (the hub loads the next stage's tool ahead).
  const { loadingText } = await load('stages/common.js');
  assert.equal(loadingText('omnitexture', 'qwen'), 'Loading Qwen-Image 2.1… (this can take a minute)');
  store.status.tools.qwen.starting_seconds = 47;
  assert.equal(loadingText('omnitexture', 'qwen'), 'Loading Qwen-Image 2.1… (this can take a minute; 47 s so far)');
  store.status.tools.qwen.state = 'ready';
  assert.equal(loadingText('omnitexture', 'qwen'), '');
  store.status.tools.qwen = { ...store.status.tools.qwen, state: 'starting', starting_seconds: null };
  assert.equal(headline(new Set(), { stage: 'images', prop: 'p_1' }), null, 'the Images workbench says it already');
  reset(status('qwen', { states: { qwen: 'error' } }));
  assert.deepEqual([headline().kind, headline().tool, headline().text], ['error', 'qwen', 'Qwen-Image 2.1 stopped with an error.']);
  reset({ ...status('qwen'), jobs: [] });
  store.jobs = [hubJob({ id: 'j1', stage: 'images', tool: 'qwen', status: 'starting_tool', prop_ids: ['p_1'], progress: {} })];
  store.status.tools.qwen.state = 'starting';
  assert.equal(headline().text, 'Loading Qwen-Image 2.1… (this can take a minute)');
  const drawing = { id: 'j1', stage: 'images', tool: 'qwen', status: 'running', prop_ids: ['p_1'], current_prop: 'p_1',
    progress: { done: 0, total: 1, fraction: 0.4, message: 'Drawing · step 8 of 25 · about 2 min left' } };
  store.jobs = [hubJob(drawing)];
  assert.equal(headline().text, 'Making pictures · Barrel · Drawing · step 8 of 25 · about 2 min left');
  // A new prop's first pictures: it has no name yet, so the one its prompt will give it.
  store.project.props.push({ id: 'p_2', name: '', draft: true, images: { settings: { prompt: 'a toy rocket with fins' } } });
  store.jobs = [hubJob({ ...drawing, prop_ids: ['p_2'], current_prop: 'p_2' })];
  assert.equal(headline().text, 'Making pictures · Toy rocket · Drawing · step 8 of 25 · about 2 min left');
  store.project.props.at(-1).images.settings.prompt = '';
  assert.equal(headline().text, 'Making pictures · the new prop · Drawing · step 8 of 25 · about 2 min left');
});

test('the two ways to a picture, on the cards and on the mode control; Qwen-Image 2.1 takes several example images', () => {
  const words = (tool) => Object.fromEntries(modesFor(tool).map((m) => [m.value, [m.label, m.choice]]));
  assert.deepEqual(modesFor('qwen').map((m) => m.value), ['import', 'example'], 'two ways: Text and Sketch are in Render Images');
  for (const tool of ['qwen', 'forge', null]) {
    assert.deepEqual(words(tool).import, ['Use Existing Images', 'Use Existing Images'], 'the tab and the card');
    assert.deepEqual(words(tool).example, ['Render Images', 'Render Images'], 'the tab and the card');
  }
  const text = (tool) => modesFor(tool).find((m) => m.value === 'example').text;
  assert.equal(text('qwen'), 'Generate images from words, with example images, or your own sketch');
  assert.equal(text(null), text('qwen'), 'no generator: the default one\'s');
  assert.equal(text('forge'), 'Generate images from words, with an example image, or your own sketch');
  assert.equal(IMAGES.peek, undefined, 'the hover peek is gone');
});

test('a prop opens in the mode it was left in; the former Text way opens as Render Images with its examples', () => {
  const refs = ['reference/reference_001.png'];
  assert.equal(modeOf({ id: 'q1', images: { references: refs, settings: { prompt: 'a lamp' } } }), 'example');
  assert.equal(modeOf({ id: 'q2', images: { references: refs, settings: { prompt: 'a lamp', use_references: true } } }), 'example');
  assert.equal(modeOf({ id: 'q3', images: { references: refs, settings: { prompt: 'a lamp', use_references: false } } }),
    'example', 'saved by the former Text way: its example pictures go along again');
  assert.equal(modeOf({ id: 'q4', images: { chosen: 'images/0001.png', settings: { prompt: '' } } }), 'import');
  assert.equal(modeOf({ id: 'q5', images: { settings: { prompt: 'a lamp', use_references: false } } }), 'example');
  // Left on the former Sketch tab: Render Images, where the viewport is painted on.
  assert.equal(modeOf({ id: 'q6', images: { sketch: 'sketch/sketch.png', references: refs, chosen: 'images/0001.png', settings: { prompt: 'a lamp', from_sketch: true } } }), 'example');
});

/** A hub that keeps `prop`'s pictures like projects.py does: images/select, images/remove, images/restore. */
function pictureHub(prop) {
  const calls = [];
  const hub = { calls, hold: null };
  globalThis.fetch = async (path, init = {}) => {
    const [, what] = path.split('/images/');
    const body = init.body ? JSON.parse(init.body) : {};
    const img = prop.images;
    if (what) calls.push([what, body]);
    if (what === 'select') img.chosen = body.path;
    if (what === 'remove') {
      const at = img.candidates.indexOf(body.path);
      img.candidates = img.candidates.filter((c) => c !== body.path);
      img.views = img.views.filter((v) => v.src !== body.path);
      if (img.chosen === body.path) img.chosen = img.candidates[Math.max(at - 1, 0)] ?? null;
    }
    if (what === 'restore') {
      img.candidates.splice(body.index, 0, body.path);
      img.views = [...img.views, ...body.views];
      if (body.chosen) img.chosen = body.path;
    }
    const answer = JSON.stringify(prop);
    if (hub.hold) await hub.hold;
    return { status: 200, text: async () => answer };
  };
  return hub;
}

test('the row of pictures marks the chosen one as the store has it: an import, a generation, a click, a removal, undo', async () => {
  const ctx = { selectedId: 'p_1', select() {}, goto() {} };
  const tick = () => new Promise((r) => setTimeout(r, 0));
  const pic = (n) => `images/000${n}.png`;
  const views = ['front', 'left', 'back', 'right'].map((view, i) => ({ view, src: pic(1), rect: [i * 10, 0, 10, 10] }));
  const hubProp = {
    id: 'p_1', name: 'Barrel', thumbs: {},
    images: { chosen: pic(1), candidates: [pic(1), pic(2)], views, settings: { prompt: 'a barrel' } },
  };
  const { replaceProp, queueSettings, flushSettings } = await load('actions.js');
  reset(status('qwen'));
  store.project.props = [structuredClone(hubProp)];
  const hub = pictureHub(hubProp);
  // The workbench's keys: the listeners it adds to the page, the capturing ones first, as a browser calls them.
  const keys = [];
  const { addEventListener } = document;
  document.addEventListener = (type, fn, capture) => { if (type === 'keydown') keys.push([fn, capture === true]); };
  const ctrlZ = () => {
    const e = {
      type: 'keydown', key: 'z', ctrlKey: true, defaultPrevented: false, stopped: false,
      preventDefault() { e.defaultPrevented = true; }, stopImmediatePropagation() { e.stopped = true; },
    };
    for (const [fn] of [...keys.filter((k) => k[1]), ...keys.filter((k) => !k[1])]) if (!e.stopped) fn(e);
    return e;
  };
  const host = new FakeEl('div');
  IMAGES.mount(host, ctx);
  document.addEventListener = addEventListener;
  const tiles = () => host.byClass('sv-cand');
  const name = (el) => el.all((x) => x.tagName === 'IMG')[0].attrs.src.match(/images\/\d+\.png/)[0];
  const row = () => tiles().map(name);
  const marked = () => tiles().filter((t) => t.classList.contains('on')).map(name);
  const shown = () => host.byClass('sv-picture-img').filter((e) => e.tagName === 'IMG').map(name);   // (none: the black canvas)
  const state = () => [row(), marked(), shown()];
  const [strip] = host.byClass('sv-cands');
  assert.deepEqual(state(), [[pic(1), pic(2)], [pic(1)], [pic(1)]], 'the oldest first, the chosen one marked and shown');
  assert.ok(tiles().every((t) => t.byClass('sv-cand-x').length === 1), 'each has its ×');

  // An import (the hub chooses the new picture) and a finished generation (it chooses the first new one).
  hubProp.images.candidates.push(pic(3));
  hubProp.images.chosen = pic(3);
  replaceProp(structuredClone(hubProp));
  IMAGES.update(ctx);
  assert.deepEqual(state(), [[pic(1), pic(2), pic(3)], [pic(3)], [pic(3)]], 'after an import: on the right');
  hubProp.images.candidates.push(pic(4), pic(5));
  hubProp.images.chosen = pic(4);
  store.project.props = [structuredClone(hubProp)];   // the project read again after the job
  IMAGES.update(ctx);
  assert.deepEqual(state(), [[pic(1), pic(2), pic(3), pic(4), pic(5)], [pic(4)], [pic(4)]], 'after a generation');

  // A click marks and shows the picture at once; an answer to something asked before it cannot put the old one back.
  let release;
  hub.hold = new Promise((r) => { release = r; });
  tiles()[0].click();
  IMAGES.update(ctx);
  assert.deepEqual([marked(), shown()], [[pic(1)], [pic(1)]], 'before the hub has answered');
  replaceProp({ ...structuredClone(hubProp), images: { ...structuredClone(hubProp.images), chosen: pic(4) } });   // a late answer
  IMAGES.update(ctx);
  assert.deepEqual([marked(), shown()], [[pic(1)], [pic(1)]], 'a late answer does not move the mark back');
  hub.hold = null;
  release();
  await tick();
  IMAGES.update(ctx);
  assert.deepEqual([marked(), shown(), store.project.props[0].images.chosen], [[pic(1)], [pic(1)], pic(1)]);

  // The hub chose another picture meanwhile and only a save's answer says so: the row follows that too.
  hubProp.images.chosen = pic(2);
  let told = 0;
  const { on } = await load('store.js');
  on('prop', () => { told += 1; });
  queueSettings('p_1', 'images', { seed: 5 });
  await flushSettings();
  assert.equal(told, 1, 'the views are told');
  IMAGES.update(ctx);
  assert.deepEqual([marked(), shown()], [[pic(2)], [pic(2)]]);

  // × takes a picture out of the prop: the chosen one (with its views) hands over to its neighbour.
  hubProp.images.chosen = pic(1);
  replaceProp(structuredClone(hubProp));
  IMAGES.update(ctx);
  hub.calls.length = 0;
  tiles()[0].byClass('sv-cand-x')[0].click();
  await tick();
  await tick();
  IMAGES.update(ctx);
  assert.deepEqual(hub.calls, [['remove', { path: pic(1) }]]);
  assert.deepEqual(state(), [[pic(2), pic(3), pic(4), pic(5)], [pic(2)], [pic(2)]]);
  assert.deepEqual(store.project.props[0].images.views, []);
  // Ctrl+Z: back at its place, with its views, chosen again; the slicer does not get these keys.
  const e = ctrlZ();
  assert.ok(e.defaultPrevented && e.stopped);
  await tick();
  await tick();
  IMAGES.update(ctx);
  assert.deepEqual(hub.calls.at(-1), ['restore', { path: pic(1), index: 0, views, chosen: true }]);
  assert.deepEqual(state(), [[pic(1), pic(2), pic(3), pic(4), pic(5)], [pic(1)], [pic(1)]]);
  assert.deepEqual(store.project.props[0].images.views, views);
  assert.equal(ctrlZ().defaultPrevented, false, 'nothing more to bring back');

  // One picture: the row still shows, so it can be clicked and removed; the last one gone: no row, no picture.
  for (const n of [5, 4, 3, 2]) {
    tiles().find((t) => name(t) === pic(n)).byClass('sv-cand-x')[0].click();
    await tick();
    await tick();
    IMAGES.update(ctx);
  }
  assert.deepEqual([state(), strip.hidden], [[[pic(1)], [pic(1)], [pic(1)]], false]);
  tiles()[0].byClass('sv-cand-x')[0].click();
  await tick();
  await tick();
  IMAGES.update(ctx);
  assert.deepEqual([state(), strip.hidden, store.project.props[0].images.chosen], [[[], [], []], true, null]);
  ctrlZ();
  await tick();
  await tick();
  IMAGES.update(ctx);
  assert.deepEqual(state(), [[pic(1)], [pic(1)], [pic(1)]], 'the last one comes back too');
  IMAGES.unmount();
});

test('a tile of the row takes the click as a whole - the picture and the margin around it - and only its × removes', async () => {
  const ctx = { selectedId: 'p_1', select() {}, goto() {} };
  const tick = () => new Promise((r) => setTimeout(r, 0));
  const pic = (n) => `images/000${n}.png`;
  const hubProp = {
    id: 'p_1', name: 'Barrel', thumbs: {},
    images: { chosen: pic(1), candidates: [pic(1), pic(2)], views: [], settings: { prompt: 'a barrel' } },
  };
  reset(status('qwen'));
  store.project.props = [structuredClone(hubProp)];
  const hub = pictureHub(hubProp);
  const host = new FakeEl('div');
  IMAGES.mount(host, ctx);
  // The header: the picture's tools and the mode control, nothing else (the prop's name is in the props list).
  const [head] = host.byClass('sv-imghead');
  assert.deepEqual(head.children.map((c) => c.className), ['sv-tools', 'seg sv-modes']);
  assert.equal(head.all((e) => e.tagName === 'INPUT').length, 0);

  // The newest picture's tile (on the right): the click is the tile's own (a click on the picture inside it comes
  // up to it, one on the margin around the picture starts on it), so the picture's button has none.
  const [, tile] = host.byClass('sv-cand');
  const [pick] = tile.byClass('sv-cand-pick');
  const [x] = tile.byClass('sv-cand-x');
  assert.deepEqual(tile.children, [pick, x]);
  assert.equal(pick.listeners.click, undefined);
  assert.equal(tile.title, 'Use this picture. Alt + click: what Qwen-Image was given for it.');
  assert.equal(x.title, 'Remove this picture (Ctrl+Z brings it back)');
  const { on } = await load('store.js');
  const rang = [];
  on('pulse-prop', ({ id }) => rang.push(id));
  tile.click();
  await tick();
  await tick();
  assert.deepEqual(hub.calls, [['select', { path: pic(2) }]], 'chosen, not removed');
  assert.deepEqual(rang, ['p_1'], 'the prop\'s tile in the column rings: it is this picture now');

  // The ×: its click stops there (it would choose the picture on its way through the tile) and removes.
  hub.calls.length = 0;
  IMAGES.update(ctx);
  let stopped = 0;
  for (const fn of host.byClass('sv-cand-x')[0].listeners.click) fn({ stopPropagation() { stopped += 1; }, preventDefault() {} });
  await tick();
  await tick();
  assert.equal(stopped, 1);
  assert.deepEqual(hub.calls, [['remove', { path: pic(1) }]], 'removed, not chosen');
  IMAGES.update(ctx);
  assert.equal(host.byClass('sv-cand').length, 1);
  IMAGES.unmount();
});

/**
 * The page's windows and toasts for run(): <body> holds the windows (a <dialog> opens as showModal() says), the
 * toasts have their place, and their timers that wait a second or more never come. Then all is back.
 */
async function withWindows(run) {
  const before = { body: document.body, timeout: globalThis.setTimeout };
  document.body = new FakeEl('body');
  const toasts = new FakeEl('div');
  document.getElementById = (id) => (id === 'toasts' ? toasts : null);
  globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 1000 ? 0 : before.timeout(fn, ms, ...rest));
  FakeEl.prototype.showModal = function showModal() { this.open = true; };
  try {
    await run({ toasts });
  } finally {
    document.body = before.body;
    delete document.getElementById;
    globalThis.setTimeout = before.timeout;
    delete FakeEl.prototype.showModal;
  }
}

test('Alt + click on a tile shows what Qwen-Image was given for its picture and chooses nothing; a picture without a record only says so; a plain click still chooses', async () => {
  const ctx = { selectedId: 'p_1', select() {}, goto() {} };
  const tick = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };
  const pic = (n) => `images/000${n}.png`;
  const hubProp = {
    id: 'p_1', name: 'Barrel', thumbs: {},
    images: { chosen: pic(1), candidates: [pic(1), pic(2), pic(3)], views: [], settings: { prompt: 'a barrel' } },
  };
  reset(status('qwen'));
  store.project.props = [structuredClone(hubProp)];
  const hub = pictureHub(hubProp);
  // The records the hub keeps beside the pictures: a masked change drawn in a window of a 2K picture (0002), one
  // drawn by FLUX.3 from the words alone (0003); the first picture has none.
  const records = {
    '0002': {
      words: 'Change that part: a red door.', model: 'turbo', steps: 8, seed: 12, noise_seed: 34,
      size: [1024, 1024], picture: [2048, 2048], way: 'window', window: [1024, 983, 2048, 2007],
      pictures: [{ file: 'picture-1.webp', seen: [1024, 1024], is: 'window' }],
      start: 'start.webp', strength: 0.8, mask: 'mask.png', answers: ['answer-1.webp', 'answer-2.webp'], answer: 1,
    },
    '0003': {
      words: 'a barrel', model: 'flux3', steps: 16, seed: 5, size: [1024, 1024], picture: [1024, 1024], way: 'words',
      pictures: [], start: null, mask: null, answers: [], answer: 0,
    },
  };
  const read = [];
  const keeps = globalThis.fetch;
  globalThis.fetch = async (path, init) => {
    const record = /\/\.given\/(\w+)\/given\.json/.exec(path);
    if (!record) return keeps(path, init);
    read.push(path.split('?')[0]);
    const kept = records[record[1]];
    return kept ? { ok: true, status: 200, json: async () => kept } : { ok: false, status: 404 };
  };
  await withWindows(async ({ toasts }) => {
    const host = new FakeEl('div');
    IMAGES.mount(host, ctx);
    try {
      const tiles = () => host.byClass('sv-cand');
      const altClick = (tile) => { for (const fn of tile.listeners.click) fn({ altKey: true, preventDefault() {}, stopPropagation() {} }); };
      const windows = () => document.body.byClass('ot-inputs');
      const file = (el) => el.all((e) => e.tagName === 'IMG')[0].attrs.src.split('?')[0].split('/').pop();
      assert.ok(tiles().every((t) => t.title.endsWith('. Alt + click: what Qwen-Image was given for it.')), 'the tooltip says the key');

      altClick(tiles()[1]);
      await tick();
      assert.deepEqual(read, ['/files/Props/props/p_1/images/.given/0002/given.json'], 'the record beside that picture');
      assert.deepEqual(hub.calls, [], 'the Alt + click chose nothing');
      assert.equal(windows().length, 1);
      const [shown] = windows();
      assert.equal(shown.open, true);
      assert.equal(shown.all((e) => e.tagName === 'H2')[0].textContent, 'What Qwen-Image was given for this picture');
      assert.deepEqual(shown.byClass('ot-words').map((e) => e.textContent), ['Change that part: a red door.']);
      const fields = shown.byClass('field').map((f) => [f.byClass('field-label')[0].textContent,
        f.all((e) => e.tagName === 'FIGCAPTION').map((e) => e.textContent), f.byClass('ot-input').map(file)]);
      assert.deepEqual(fields.slice(1), [
        ['Pictures given, in this order', [
          'Picture 1: the part of the picture it works on, what you brushed in magenta (Qwen-Image sees it at 1024 × 1024 px)',
        ], ['picture-1.webp']],
        ['Not given to it as pictures', [
          'Not a picture it sees: what its drawing is held to, outside where it may draw',
          'Not a picture it sees: where it may draw (white)',
        ], ['start.webp', 'mask.png']],
        ['What came back', ['Qwen-Image’s own picture, before it was laid into the picture', 'The picture as it was saved'],
          ['answer-2.webp', '0002.png']],
      ], 'in order, the start and the mask apart, the answer this picture came from');
      assert.deepEqual(shown.byClass('hint').map((e) => e.textContent), [
        'Qwen-Image 2.1 Turbo · seed 12 · noise seed 34 · 8 steps · a 1024 × 1024 px window of a 2048 × 2048 px picture · '
          + 'a masked change, drawn in the window and laid back into the picture',
        'The files: projects/Props/props/p_1/images/.given/0002',
      ]);

      // Drawn from the words alone, and the picture saved is the model's answer itself.
      altClick(tiles()[2]);
      await tick();
      const [, words] = windows();
      assert.equal(words.all((e) => e.tagName === 'H2')[0].textContent, 'What FLUX.3 Image Klein was given for this picture');
      assert.deepEqual(words.byClass('field').map((f) => f.byClass('field-label')[0].textContent),
        ['Words', 'Pictures given, in this order', 'What came back'], 'nothing apart from the pictures');
      assert.ok(words.byClass('hint').some((e) => e.textContent === 'None: it drew from the words alone.'));
      assert.deepEqual(words.all((e) => e.tagName === 'FIGCAPTION').map((e) => e.textContent), ['FLUX.3 Image Klein’s picture, as it was saved']);
      assert.ok(words.byClass('hint')[0].textContent.startsWith('FLUX.3 Image Klein · seed 5 · 16 steps · 1024 × 1024 px · drawn from the words alone'));

      // No record: a toast says so, no window opens and nothing is chosen.
      altClick(tiles()[0]);
      await tick();
      assert.equal(read.at(-1), '/files/Props/props/p_1/images/.given/0001/given.json');
      assert.equal(windows().length, 2);
      assert.match(toasts.textContent, /^There is no record of what Qwen-Image was given for this picture: it was made before/);
      assert.deepEqual(hub.calls, []);

      // A plain click: the picture is chosen, as ever, and no record is read.
      const asked = read.length;
      tiles()[1].click();
      await tick();
      assert.deepEqual([hub.calls, read.length, windows().length], [[['select', { path: pic(2) }]], asked, 2]);

      // A masked change of a large picture, drawn whole: the hint says what part of the redraw was laid into it.
      records['0003'] = {
        ...records['0003'], model: 'turbo', way: 'whole', size: [2048, 2048], picture: [2048, 2048], patch: [488, 488, 1512, 1512],
      };
      altClick(tiles()[2]);
      await tick();
      assert.ok(windows().at(-1).byClass('hint')[0].textContent.endsWith('16 steps · 2048 × 2048 px · a masked change: all of the '
        + 'picture drawn anew from the brushed one, then brought to its tone and its place; a 1024 × 1024 px part of that, '
        + 'round what you brushed, laid into the picture'));
    } finally {
      IMAGES.unmount();
    }
  });
});

test('the rough preview of a window lies over that part of the picture - in parts round a tileable picture\'s edge -, the picture round it; one of all of it in the picture\'s place, as before', async () => {
  const ctx = { selectedId: 'p_1', select() {}, goto() {} };
  const tick = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };
  const hubProp = {
    id: 'p_1', name: 'Wall', thumbs: {},
    images: { chosen: 'images/0001.png', candidates: ['images/0001.png'], views: [], settings: { prompt: 'a wall' } },
  };
  reset(status('qwen'));
  store.project.props = [structuredClone(hubProp)];
  pictureHub(hubProp);
  const host = new FakeEl('div');
  // A picture loads as soon as it has its address; the layer swaps one preview for the next in its place.
  const page = {
    prepend(...nodes) { this.children.unshift(...nodes); },
    replaceWith(node) {
      for (const el of host.all((e) => e.children.includes(this))) el.children.splice(el.children.indexOf(this), 1, node);
    },
  };
  Object.assign(FakeEl.prototype, page);
  Object.defineProperty(FakeEl.prototype, 'src', {
    configurable: true, get() { return this.address; },
    set(v) {
      this.address = v;
      queueMicrotask(() => this.onload?.());
    },
  });
  try {
    IMAGES.mount(host, ctx);
    const [img] = host.byClass('sv-picture-img');
    Object.assign(img, { naturalWidth: 2048, naturalHeight: 1024 });
    const [live] = host.byClass('sv-live');
    const show = async (preview) => {
      store.jobs = [hubJob({
        id: 'j1', stage: 'images', status: 'running', prop_ids: ['p_1'], current_prop: 'p_1', progress: { preview },
      })];
      IMAGES.update(ctx);
      await tick();
    };
    const parts = () => host.byClass('sv-live-part').map((p) => [p.style.cssText, p.children[0].style.cssText]);
    const url = (n) => `/api/jobs/j1/preview?v=${n}`;

    await show({ url: url(1), step: 1, prop: 'p_1', box: [0.25, 0.5, 0.75, 1] });
    assert.equal(live.hidden, false);
    assert.ok(live.classList.contains('part'), 'the picture shows round it');
    const [fit] = host.byClass('sv-live-fit');
    assert.equal(fit.style['--ar'], '2', 'on the picture\'s own fitted box, not on the viewport');
    assert.deepEqual(parts(), [['left: 25%; top: 50%; width: 50%; height: 50%', 'left: 0%; top: 0%; width: 100%; height: 100%']]);
    assert.equal(host.byClass('sv-live-part')[0].children[0].attrs.src, url(1));
    assert.equal(live.children.at(-1).className, 'sv-live-tag', '"Rough preview" over it');

    // On a tileable picture, past its right and bottom edge: what passes them comes back at the left and the top.
    await show({ url: url(2), step: 2, prop: 'p_1', box: [0.75, 0.5, 1.25, 1.5] });
    assert.deepEqual(parts(), [
      ['left: 75%; top: 50%; width: 25%; height: 50%', 'left: 0%; top: 0%; width: 200%; height: 200%'],
      ['left: 75%; top: 0%; width: 25%; height: 50%', 'left: 0%; top: -100%; width: 200%; height: 200%'],
      ['left: 0%; top: 50%; width: 25%; height: 50%', 'left: -100%; top: 0%; width: 200%; height: 200%'],
      ['left: 0%; top: 0%; width: 25%; height: 50%', 'left: -100%; top: -100%; width: 200%; height: 200%'],
    ]);
    assert.equal(host.byClass('sv-live-fit').length, 1, 'in place of the one before');

    // A preview of all of the picture: in its place, over all of it, as before.
    await show({ url: url(3), step: 3, prop: 'p_1' });
    assert.equal(live.classList.contains('part'), false);
    assert.deepEqual(live.children.map((c) => c.className), ['sv-live-img', 'sv-live-tag']);
    assert.equal(live.children[0].src, url(3));
    assert.equal(host.byClass('sv-live-part').length, 0);
  } finally {
    IMAGES.unmount();
    for (const k of Object.keys(page)) delete FakeEl.prototype[k];
    delete FakeEl.prototype.src;
    store.jobs = [];
  }
});

test('a picture dragged out of the row and let go on the props column becomes a prop of its own; inside the row a drag still scrolls', async () => {
  const ctx = { selectedId: 'p_1', select() {}, goto() {} };
  const tick = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };
  const pic = (n) => `images/000${n}.png`;
  const hubProp = {
    id: 'p_1', name: 'Cabin', thumbs: {},
    images: { chosen: pic(1), candidates: [pic(1), pic(2)], views: [], settings: { prompt: 'a cabin' } },
  };
  reset(status('qwen'));
  store.project.props = [structuredClone(hubProp)];
  const calls = [];
  globalThis.fetch = async (path, init = {}) => {
    calls.push([init.method || 'GET', path, init.body ? JSON.parse(init.body) : null]);
    if (init.method === 'POST' && path.endsWith('/props')) {
      return { ok: true, status: 200, text: async () => JSON.stringify({ id: 'p_new', name: JSON.parse(init.body).name, images: {} }) };
    }
    return { ok: true, status: 200, blob: async () => ({ type: 'image/png' }), text: async () => JSON.stringify(hubProp) };
  };
  const uploads = [];
  globalThis.XMLHttpRequest = class {
    constructor() { this.upload = {}; }
    open(method, path) { this.path = path; }
    send() {
      uploads.push(this.path);
      this.status = 200;
      this.responseText = JSON.stringify({ id: 'p_new', name: 'Cabin 2', images: { chosen: 'images/upload_001.png' } });
      queueMicrotask(() => this.onload());
    }
  };
  globalThis.File = class { constructor(parts, name, opts) { Object.assign(this, { parts, name, type: opts?.type }); } };
  const column = new FakeEl('div');
  column.className = 'props';
  const inColumn = new FakeEl('div');
  column.append(inColumn);
  const parts = {
    contains(node) { return node === this || this.all((e) => e === node).length > 0; },
    closest(sel) { return sel === '.props' && (this === column || this === inColumn) ? column : null; },
    remove() { this.removed = true; },
    setPointerCapture() {},
    getBoundingClientRect: () => ({ left: 0, right: 800, top: 580, bottom: 640, width: 800, height: 60 }),   // the row
  };
  Object.assign(FakeEl.prototype, parts);
  const before = {
    query: document.querySelector, at: document.elementFromPoint, body: document.body,
    add: globalThis.addEventListener, remove: globalThis.removeEventListener,
  };
  const windowPointer = {};
  globalThis.addEventListener = (type, fn) => { if (type.startsWith('pointer')) (windowPointer[type] ||= []).push(fn); };
  globalThis.removeEventListener = (type, fn) => {
    if (windowPointer[type]) windowPointer[type] = windowPointer[type].filter((f) => f !== fn);
  };
  document.body = new FakeEl('body');   // where the lifted picture goes
  const toasts = new FakeEl('div');
  document.getElementById = (id) => (id === 'toasts' ? toasts : null);
  const { setTimeout: realTimeout } = globalThis;   // the toast's own end never comes after the test
  globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 1000 ? 0 : realTimeout(fn, ms, ...rest));
  document.querySelector = (sel) => (sel === '.props' ? column : null);
  document.elementFromPoint = (x) => (x < 100 ? inColumn : null);
  const { on } = await load('store.js');
  const rang = [];
  on('pulse-prop', ({ id }) => rang.push(id));
  try {
    const host = new FakeEl('div');
    IMAGES.mount(host, ctx);
    const [strip] = host.byClass('sv-cands');
    const [, tile] = host.byClass('sv-cand');
    assert.equal(tile.attrs['data-path'], pic(2));
    const [img] = tile.all((e) => e.tagName === 'IMG');
    // A press on the row; what follows it, on the window (a quick drag leaves the row at once).
    const send = (type, x, y) => {
      const e = { type, pointerId: 3, pointerType: 'mouse', button: 0, clientX: x, clientY: y, target: img };
      for (const fn of (type === 'pointerdown' ? strip.listeners[type] : windowPointer[type]) || []) fn(e);
    };
    // Inside the row: it scrolls, nothing is lifted.
    send('pointerdown', 500, 600);
    send('pointermove', 470, 602);
    send('pointerup', 470, 602);
    assert.equal(document.body.children.filter((e) => e.className === 'tr-ghost').length, 0);

    // Out of the row (leftwards and up, the way to the column): the row goes back, the picture follows, the column
    // shows it takes it, and the release makes the prop.
    strip.scrollLeft = 0;
    send('pointerdown', 500, 600);
    send('pointermove', 420, 610);
    assert.equal(strip.scrollLeft, 80, 'inside the row it scrolls first');
    send('pointermove', 300, 560);
    assert.equal(strip.scrollLeft, 0, 'out of it: the row as it was');
    const ghost = document.body.children.find((e) => e.className === 'tr-ghost' && !e.removed);
    assert.ok(ghost, 'lifted');
    assert.ok(column.classList.contains('drop-ready') && !column.classList.contains('drop-here'));
    send('pointermove', 60, 300);
    assert.ok(column.classList.contains('drop-here') && ghost.classList.contains('over'));
    send('pointerup', 60, 300);
    assert.ok(ghost.removed && !column.classList.contains('drop-ready'), 'the column as it was');
    assert.deepEqual(Object.values(windowPointer).flat(), [], 'the window let go of the press');
    await tick();
    const made = calls.find(([m, p]) => m === 'POST' && p.endsWith('/props'));
    assert.deepEqual(made?.[2], { name: 'Cabin 2' }, 'named after this prop, with the next free number');
    assert.ok(calls.some(([m, p]) => m === 'GET' && p.includes('0002.png')), 'the picture read');
    assert.deepEqual(uploads.map((u) => u.split('/props/')[1]), ['p_new/images/upload']);
    assert.ok(rang.includes('p_new'), 'its tile rings');
    assert.match(toasts.textContent, /“Cabin 2” is a prop of its own now\./);

    // Let go anywhere else: nothing happens.
    calls.length = 0;
    send('pointerdown', 500, 600);
    send('pointermove', 500, 560);
    send('pointerup', 400, 300);
    await tick();
    assert.deepEqual(calls.filter(([m]) => m === 'POST'), []);
    IMAGES.unmount();
  } finally {
    document.querySelector = before.query;
    document.elementFromPoint = before.at;
    document.body = before.body;
    globalThis.addEventListener = before.add;
    globalThis.removeEventListener = before.remove;
    delete document.getElementById;
    globalThis.setTimeout = realTimeout;
    for (const k of Object.keys(parts)) delete FakeEl.prototype[k];
    delete globalThis.XMLHttpRequest;
    delete globalThis.File;
  }
});

test('Render Images without a picture: a black canvas of the prop\'s Shape to paint on, and Re-Render Viewport sends it', async () => {
  const ctx = { selectedId: 'b_1', select() {}, goto() {} };
  const tick = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };
  const hubProp = {
    id: 'b_1', name: 'Lamp', thumbs: {},
    images: { chosen: null, candidates: [], references: [], views: [], settings: { prompt: 'a lamp', width: 832, height: 1216 } },
  };
  reset(status('qwen'));
  store.project.props = [structuredClone(hubProp)];
  const calls = [];
  const job = hubJob({ id: 'j_1', status: 'queued', stage: 'images', prop_ids: ['b_1'], progress: {} });
  const what = (path) => (path.includes('/api/jobs') ? 'jobs' : path.endsWith('/images/viewport') ? 'viewport' : 'prop');
  globalThis.FormData = class { constructor() { this.parts = []; } append(...a) { this.parts.push(a); } };
  globalThis.fetch = async (path, init = {}) => {
    const body = typeof init.body === 'string' ? JSON.parse(init.body) : init.body;
    if (init.method && init.method !== 'GET') calls.push([init.method, what(path), body]);
    if (what(path) === 'jobs') return { status: 200, text: async () => JSON.stringify(init.method === 'GET' ? [job] : job) };
    if (what(path) === 'viewport') return { status: 200, text: async () => JSON.stringify({ path: 'viewport/viewport.png' }) };
    return { status: 200, text: async () => JSON.stringify(hubProp) };
  };
  const canvasParts = {
    getContext() {
      this.ctx ||= new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => {}), set: (t, k, v) => { t[k] = v; return true; } });
      return this.ctx;
    },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 512, height: 512 }),
    toBlob: (done) => done({ png: true }),
  };
  Object.assign(FakeEl.prototype, canvasParts);
  const { setTimeout: realTimeout } = globalThis;
  globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 1000 ? 0 : realTimeout(fn, ms, ...rest));
  try {
    const host = new FakeEl('div');
    IMAGES.mount(host, ctx);
    assert.deepEqual(host.byClass('sv-modes')[0].all((e) => e.tagName === 'BUTTON').map((b) => b.textContent),
      ['Use Existing Images', 'Render Images'], 'no Render Sketch: the viewport is painted on instead');
    const [blank] = host.byClass('sv-blank');
    assert.deepEqual([blank.tagName, blank.width, blank.height], ['CANVAS', 832, 1216], 'black, the prop\'s Shape');
    assert.equal(host.byClass('sv-empty').length, 0, 'no "Your picture appears here"');
    assert.equal(host.byClass('mk-bar')[0].hidden, false, 'the markers, to paint on it');
    const [viewport, viaExamples] = host.byClass('sv-rerender');
    assert.deepEqual([viewport.disabled, viaExamples.disabled], [false, true], 'Re-Render Viewport is always there');
    viewport.click();
    await tick();
    assert.deepEqual(calls.map((c) => `${c[0]} ${c[1]}`), ['POST viewport', 'POST jobs'], 'the canvas goes up, painted or not');
    assert.deepEqual(calls.at(-1)[2].params, { from_viewport: 'viewport/viewport.png', use_references: false });
    IMAGES.unmount();
  } finally {
    globalThis.setTimeout = realTimeout;
    for (const key of Object.keys(canvasParts)) delete FakeEl.prototype[key];
  }
});

test('the card decides "Faster drawing" also when the status arrives after the bench was built', () => {
  // The project can arrive before the hub's first /api/status (which runs nvidia-smi): the tick is read again at every
  // sync, so a GTX 10xx shows it on as the launcher draws it; the artist's own tick, stored, wins.
  const ctx = { selectedId: 'w_1', select() {}, goto() {} };
  const total = 4_199_565_024;
  const withCard = (cap) => {
    const st = status('qwen');
    st.gpu = { compute_cap: cap };
    st.qwen_full = { available: true, downloading: false, waiting: false, done_bytes: total, total_bytes: total, error: null };
    return st;
  };
  const tick = (host) => host.all((e) => e.classList.contains('switch')).find((e) => e.textContent.includes('Faster drawing'));
  reset(null);
  store.project.props = [{ id: 'w_1', name: 'Barrel', images: { settings: { prompt: 'a barrel', qwen_model: 'full', first_block_cache: null } } }];
  const host = new FakeEl('div');
  IMAGES.mount(host, ctx);
  assert.equal(tick(host).input.checked, false, 'no card known yet: off, as on a new card');
  store.status = withCard(6.1);
  IMAGES.update(ctx);
  assert.equal(tick(host).input.checked, true, 'a GTX 10xx: on, as the launcher draws it');
  store.status = withCard(8.6);
  IMAGES.update(ctx);
  assert.equal(tick(host).input.checked, false, 'an RTX 3060: off');
  store.project.props[0].images.settings.first_block_cache = true;
  IMAGES.update(ctx);
  assert.equal(tick(host).input.checked, true, 'the artist\'s stored tick wins over the card');
  IMAGES.unmount();
});

test('a download that still waits for the installer says so instead of "Downloading… 0%"', () => {
  const total = 4_199_565_024;
  const waiting = optionalModelState('full', { available: false, downloading: true, waiting: true, done_bytes: 0, total_bytes: total, error: null });
  assert.deepEqual([waiting.state, waiting.note], ['downloading', 'Waiting for the installer…']);
  assert.match(waiting.title, /The full model \(4\.2 GB\) downloads once Spellcast3D’s installer has finished with models\\qwen-image/);
  const going = optionalModelState('full', { available: false, downloading: true, waiting: false, done_bytes: total / 4, total_bytes: total, error: null });
  assert.equal(going.note, 'Downloading… 25%');
  const older = optionalModelState('flux3', { available: false, downloading: true, done_bytes: 0, total_bytes: total, error: null, engine: true });
  assert.equal(older.note, 'Downloading… 0%', 'a hub without the field: as before');
});

test('Render Images\' three buttons: Render New (no or with examples, never the viewport), and Re-Render Viewport alone or with examples', async () => {
  const ctx = { selectedId: 'r_1', select() {}, goto() {} };
  const tick = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };
  const pic = (n) => `images/000${n}.png`;
  const hubProp = {
    id: 'r_1', name: 'Table', thumbs: {},
    images: { chosen: null, candidates: [], references: [], views: [], settings: { prompt: 'make it brown', seed: 5, batch_count: 2 } },
  };
  const { replaceProp } = await load('actions.js');
  reset(status('qwen'));
  store.project.props = [structuredClone(hubProp)];
  const calls = [];
  const job = hubJob({ id: 'j_1', status: 'queued', stage: 'images', prop_ids: ['r_1'], progress: {} });
  const what = (path) => (path.includes('/api/jobs') ? 'jobs' : path.endsWith('/images/viewport') ? 'viewport'
    : path.includes('/images/references') ? 'references' : 'prop');
  globalThis.FormData = class { constructor() { this.parts = []; } append(...a) { this.parts.push(a); } };
  globalThis.fetch = async (path, init = {}) => {
    const body = typeof init.body === 'string' ? JSON.parse(init.body) : init.body;
    if (init.method && init.method !== 'GET') calls.push([init.method, what(path), body]);
    if (what(path) === 'jobs') return { status: 200, text: async () => JSON.stringify(init.method === 'GET' ? [job] : job) };
    if (what(path) === 'viewport') return { status: 200, text: async () => JSON.stringify({ path: 'viewport/viewport.png' }) };
    if (init.method === 'PATCH') Object.assign(hubProp.images.settings, body.images?.settings);
    return { status: 200, text: async () => JSON.stringify(hubProp) };
  };
  globalThis.XMLHttpRequest = class {
    constructor() { this.upload = {}; }
    open(method, path) { this.method = method; this.path = path; }
    send() {
      calls.push([this.method, what(this.path)]);
      hubProp.images.references = [...hubProp.images.references, 'reference/reference_002.png'];
      this.status = 200;
      this.responseText = JSON.stringify(hubProp);
      queueMicrotask(() => this.onload());
    }
  };
  // A canvas that takes paint, the keys (the window's and the page's: Ctrl+Z is the page's) and the pastes as the
  // workbench adds them, and the animations it starts.
  const canvasParts = {
    getContext() {
      this.ctx ||= new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => {}), set: (t, k, v) => { t[k] = v; return true; } });
      return this.ctx;
    },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 512, height: 512 }),
    toBlob: (done) => done({ png: true }),
    scrollIntoView() {},
  };
  Object.assign(FakeEl.prototype, canvasParts);
  const windowKeys = [];
  const pastes = [];
  const added = { window: globalThis.addEventListener, document: document.addEventListener };
  globalThis.addEventListener = (type, fn) => { if (type === 'keydown') windowKeys.push(fn); };
  document.addEventListener = (type, fn) => {
    if (type === 'paste') pastes.push(fn);
    if (type === 'keydown') windowKeys.push(fn);
  };
  const animated = [];
  const animate = FakeEl.prototype.animate;
  // no job polling left behind when the test ends
  const { setTimeout: realTimeout } = globalThis;
  globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 1000 ? 0 : realTimeout(fn, ms, ...rest));
  FakeEl.prototype.animate = function (frames) { animated.push([this.className, frames]); return animate.call(this); };
  const press = (key, more = {}) => {
    const e = {
      key, ctrlKey: false, ...more, stopped: false, prevented: false, preventDefault() { e.prevented = true; },
      stopImmediatePropagation() { e.stopped = true; },
    };
    for (const fn of windowKeys) if (!e.stopped) fn(e);
    return e;
  };
  try {
    const host = new FakeEl('div');
    IMAGES.mount(host, ctx);
    globalThis.addEventListener = added.window;
    document.addEventListener = added.document;
    const [render] = host.byClass('sv-render');
    const [viewport, viaExamples] = host.byClass('sv-rerender');
    assert.deepEqual(host.byClass('sv-makes')[0].children, [render, viewport, viaExamples], 'one row');
    assert.equal(host.byClass('sv-makes')[0].all((e) => e.classList.contains('icon')).length, 0, 'no icons');
    const jobs = () => calls.filter((c) => c[1] === 'jobs').map((c) => c[2].params);

    // No picture yet: Render, and Re-Render Viewport from the black canvas (no examples to go with it); the
    // variations are said after Render.
    assert.deepEqual([render.textContent, viewport.textContent, viaExamples.textContent],
      ['Render\nFrom Scratch ×2no examples', 'Re-Render Viewportno examples', 'Re-Render Viewportwith examples']);
    assert.deepEqual([render.disabled, viewport.disabled, viaExamples.disabled], [false, false, true]);
    assert.ok(![render, viewport, viaExamples].some((b) => b.classList.contains('primary')), 'never purple, also without a picture');
    const [transparent] = host.all((e) => e.classList.contains('switch') && e.textContent.startsWith('Transparent'));
    assert.equal(!!transparent.input.checked, false, 'opaque pictures unless the artist ticks it');

    // A picture and an example picture: "Render New / with examples", and both Re-Renders.
    Object.assign(hubProp.images, { candidates: [pic(1)], chosen: pic(1), references: ['reference/reference_001.png'] });
    replaceProp(structuredClone(hubProp));
    IMAGES.update(ctx);
    assert.equal(render.textContent, 'Render\nFrom Scratch ×2with examples');
    assert.deepEqual([render.disabled, viewport.disabled, viaExamples.disabled], [false, false, false]);
    assert.deepEqual([render, viewport, viaExamples].map((b) => [b.classList.contains('ghost'), b.classList.contains('primary')]),
      [[false, false], [false, false], [false, false]], 'the three alike, with or without a picture');

    // Re-Render Viewport: the picture shown, alone; the seed stays as the artist set it.
    viewport.click();
    await tick();
    assert.deepEqual(jobs().at(-1), { from_viewport: pic(1), use_references: false });
    assert.equal(hubProp.images.settings.seed, 5);
    assert.ok(!host.byClass('fold')[0].open, 'quietly: "More" stays as it was');
    assert.deepEqual([viewport.textContent, render.disabled, viaExamples.disabled], ['Waiting…no examples', true, true]);
    store.jobs = [];
    IMAGES.update(ctx);
    viaExamples.click();
    await tick();
    assert.deepEqual(jobs().at(-1), { from_viewport: pic(1), use_references: true });
    store.jobs = [];
    IMAGES.update(ctx);
    render.click();
    await tick();
    assert.equal(jobs().at(-1), undefined, 'Render: the words and the example pictures, as before');
    store.jobs = [];
    IMAGES.update(ctx);

    // The markers: in the viewport's corner once the picture has loaded; a press and drag paints, white by default.
    const [img] = host.byClass('sv-picture-img');
    Object.assign(img, { complete: true, naturalWidth: 64, naturalHeight: 64 });
    for (const fn of img.listeners.load || []) fn();
    const [bar] = host.byClass('mk-bar');
    const [white, black, rainbow] = host.byClass('mk-pen');
    const [wipe] = host.byClass('mk-wipe');
    assert.deepEqual([bar.hidden, wipe.hidden, rainbow.attrs.type], [false, true, 'color']);
    const [canvas] = host.byClass('mk-canvas');
    const stroke = () => {
      for (const [type, x, y] of [['pointerdown', 10, 10], ['pointermove', 60, 40], ['pointerup', 60, 40]]) {
        for (const fn of canvas.listeners[type]) fn({ pointerId: 1, pointerType: 'mouse', button: 0, clientX: x, clientY: y, preventDefault() {} });
      }
    };
    const [hint] = host.byClass('mk-hint');
    const [size] = host.byClass('mk-size');
    assert.ok(white.classList.contains('on') && !black.classList.contains('on'), 'white, picked');
    assert.equal(hint.hidden, true, 'no tip until the slider is dragged');
    stroke();
    assert.equal(wipe.hidden, false, 'marked');
    assert.ok(press('z', { ctrlKey: true }).stopped);
    assert.equal(wipe.hidden, true, 'Ctrl+Z took the stroke back');
    black.click();
    assert.ok(black.classList.contains('on') && !white.classList.contains('on'));
    black.click();
    assert.ok(black.classList.contains('on'), 'one marker is always picked');
    stroke();
    // Shift + moving sideways sizes the marker, no button pressed; the slider dragged shows the tip.
    const move = (x, shiftKey) => {
      for (const fn of canvas.listeners.pointermove) fn({ pointerId: 1, pointerType: 'mouse', buttons: 0, clientX: x, clientY: 30, shiftKey });
    };
    const before = +size.value;
    move(100, true);
    move(225, true);
    assert.equal(+size.value, before + 31, 'half the sweep: half the range');
    move(300, false);
    assert.equal(+size.value, before + 31, 'without Shift the pointer only moves');
    size.value = 20;
    for (const fn of size.listeners.input) fn();
    assert.equal(hint.hidden, false, 'the tip about Shift shows for a moment');
    white.click();
    // Re-Render Viewport with marks: the marked picture goes up first, and the job names it.
    calls.length = 0;
    viewport.click();
    await tick();
    assert.deepEqual(calls.map((c) => `${c[0]} ${c[1]}`), ['POST viewport', 'POST jobs']);
    assert.deepEqual(jobs().at(-1), { from_viewport: 'viewport/viewport.png', use_references: false });
    store.jobs = [];
    IMAGES.update(ctx);
    // Render New with marks on the picture: the words and the example pictures only.
    calls.length = 0;
    render.click();
    await tick();
    assert.deepEqual(calls.map((c) => `${c[0]} ${c[1]}`), ['POST jobs']);
    assert.equal(jobs().at(-1), undefined);
    store.jobs = [];
    IMAGES.update(ctx);

    // Ctrl+V: the field's own while the text cursor is in the words; elsewhere the picture joins the examples and
    // pops in.
    const paste = () => {
      const e = {
        clipboardData: { files: [{ type: 'image/png', name: 'copied.png' }], getData: () => '' },
        prevented: false, preventDefault() { e.prevented = true; },
      };
      for (const fn of pastes) fn(e);
      return e;
    };
    const [words] = host.byClass('sv-prompt');
    document.activeElement = words;
    calls.length = 0;
    assert.equal(paste().prevented, false);
    assert.deepEqual(calls, []);
    document.activeElement = null;
    animated.length = 0;
    assert.equal(paste().prevented, true);
    await tick();
    assert.deepEqual(calls.map((c) => `${c[0]} ${c[1]}`), ['POST references']);
    assert.equal(animated.filter(([cls]) => cls === 'sv-ref').length, 1, 'the new one pops in, once');
    assert.equal(host.byClass('sv-ref').filter((e) => e.className === 'sv-ref').length, 2);
    // Ctrl+Z takes the pasted picture out again.
    assert.ok(press('z', { ctrlKey: true }).stopped);
    await tick();
    assert.deepEqual(calls.at(-1).slice(0, 2), ['DELETE', 'references']);
    assert.ok(press('z', { ctrlKey: true }).stopped);
    assert.equal(wipe.hidden, true, 'then the stroke made before it');
    IMAGES.unmount();
  } finally {
    globalThis.addEventListener = added.window;
    document.addEventListener = added.document;
    document.activeElement = null;
    globalThis.setTimeout = realTimeout;
    FakeEl.prototype.animate = animate;
    for (const key of Object.keys(canvasParts)) delete FakeEl.prototype[key];
  }
});

test('New prop: the cards say Use Existing Images and Render Images; Ctrl+V takes a copied picture as the prop\'s, a field\'s paste stays the field\'s', async () => {
  const ctx = { selectedId: null, select() {}, goto() {} };
  const uploads = [];
  globalThis.XMLHttpRequest = class {
    constructor() { this.upload = {}; }
    open(method, path) { this.path = path; }
    send() { uploads.push(this.path); }
  };
  globalThis.fetch = async () => ({ status: 200, text: async () => '{}' });
  const pastes = [];
  const added = document.addEventListener;
  const removed = document.removeEventListener;
  document.addEventListener = (type, fn) => { if (type === 'paste') pastes.push(fn); };
  document.removeEventListener = (type, fn) => { if (pastes.includes(fn)) pastes.splice(pastes.indexOf(fn), 1); };
  // What the cards' flight to the workbench (morph) asks of the page.
  const flight = {
    prepend(...nodes) { this.children.unshift(...nodes); },
    remove() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }),
  };
  Object.assign(FakeEl.prototype, flight);
  const paste = (files) => {
    const e = { clipboardData: { files, getData: () => '' }, prevented: false, preventDefault() { e.prevented = true; } };
    for (const fn of [...pastes]) fn(e);
    return e;
  };
  try {
    reset(status('qwen'));
    store.project.props = [{ id: 'd_1', name: '', draft: true, images: { candidates: [], views: [], settings: {} } }];
    const host = new FakeEl('div');
    IMAGES.mount(host, ctx);
    const cards = host.byClass('sv-choice');
    assert.deepEqual(cards.map((c) => c.all((e) => e.tagName === 'B')[0].textContent),
      ['Use Existing Images', 'Render Images', 'Import folder (multiple prop images)']);
    assert.equal(cards[1].byClass('sv-choice-text')[0].textContent, 'Generate images from words, with example images, or your own sketch');
    assert.match(cards[2].byClass('sv-choice-text')[0].textContent, /from a folder\.\nA name that ends or starts with front, back, left or right .* any other image is a prop of its own\./,
      'on a line of its own, and no name is "needed"');
    // Where a prop begins: the lessons over the community in the corner; Donate comes with the work, further on.
    const corner = host.byClass('sv-chooser')[0].children.filter((node) => node.tagName === 'A');
    assert.deepEqual(corner.map((link) => [link.textContent, link.className, link.getAttribute('target')]),
      [['Video Lessons', 'btn lessons', '_blank'], ['Join Discord', 'btn discord', '_blank']]);
    assert.deepEqual(host.byClass('donate'), []);
    assert.equal(paste([{ type: 'text/plain', name: 'words.txt' }]).prevented, false, 'no picture: nothing happens');
    document.activeElement = { tagName: 'INPUT', type: 'text' };
    assert.equal(paste([{ type: 'image/png', name: 'image.png' }]).prevented, false, 'the field\'s own paste');
    document.activeElement = null;
    assert.equal(paste([{ type: 'image/png', name: 'image.png' }]).prevented, true);
    assert.ok(cards[0].classList.contains('on'), 'as a drop: Use Existing Images, with the picture');
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(uploads.some((u) => u.endsWith('/props/d_1/images/upload')), `uploaded: ${JSON.stringify(uploads)}`);
    IMAGES.unmount();
  } finally {
    document.addEventListener = added;
    document.removeEventListener = removed;
    document.activeElement = null;
    delete globalThis.XMLHttpRequest;
    for (const k of Object.keys(flight)) delete FakeEl.prototype[k];
  }
});

/**
 * A page for the tools over the picture (Select Object, Smart Mask): a hub that answers and notes what it is sent
 * (calls), canvases that keep what is painted on them (painted), the frames marks.js asks for (frames, runFrames),
 * the page's and the window's own listeners (fire), timers that never wait long. run(kit) with it, then all is back.
 */
async function withMarkedPicture(run) {
  const ctx = { selectedId: 'w_1', select() {}, goto() {} };
  const tick = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };
  const hubProp = {
    id: 'w_1', name: 'Hut', thumbs: {},
    images: { chosen: 'images/0001.png', candidates: ['images/0001.png'], references: [], views: [], settings: { prompt: 'a hut' } },
  };
  const calls = [];
  const job = hubJob({ id: 'j_9', status: 'queued', stage: 'images', prop_ids: ['w_1'], progress: {} });
  const what = (path) => (path.includes('/api/jobs') ? 'jobs' : path.endsWith('/images/viewport') ? 'viewport' : 'prop');
  globalThis.FormData = class { constructor() { this.parts = []; } append(...a) { this.parts.push(a); } };
  globalThis.fetch = async (path, init = {}) => {
    const body = typeof init.body === 'string' ? JSON.parse(init.body) : init.body;
    if (init.method && init.method !== 'GET') calls.push([init.method, what(path), body]);
    if (what(path) === 'jobs') return { status: 200, text: async () => JSON.stringify(init.method === 'GET' ? [job] : job) };
    if (what(path) === 'viewport') return { status: 200, text: async () => JSON.stringify({ path: 'viewport/viewport.png' }) };
    return { status: 200, text: async () => JSON.stringify(hubProp) };
  };
  // Canvases that keep what is painted on them: {on (the canvas), k (the call), args, style, width, op (how it is laid
  // on), alpha}; a pattern is {stripes: its tile}. The picture shows at its own size.
  const painted = [];
  const PAINTS = ['stroke', 'fill', 'arc', 'fillRect', 'drawImage', 'translate', 'quadraticCurveTo'];
  const canvasParts = {
    getContext() {
      const on = this;
      this.ctx ||= new Proxy({}, {
        get: (t, k) => {
          if (k in t) return t[k];
          if (k === 'createPattern') return (tile) => ({ stripes: tile });
          return (...args) => {
            if (!PAINTS.includes(k)) return;
            painted.push({
              on, k, args, style: k === 'stroke' ? t.strokeStyle : t.fillStyle, width: t.lineWidth, op: t.globalCompositeOperation,
              alpha: t.globalAlpha ?? 1,
            });
          };
        },
        set: (t, k, v) => { t[k] = v; return true; },
      });
      return this.ctx;
    },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 1024, height: 768 }),
    naturalWidth: 1024, naturalHeight: 768, complete: true,      // every picture is there, at this size
    toBlob: (done) => done({ png: true }),
    focus() {},
    select() { this.selected = true; },
    remove() { this.removed = true; },
    contains(node) { return node === this || this.all((e) => e === node).length > 0; },
  };
  Object.assign(FakeEl.prototype, canvasParts);
  // The frames marks.js asks for (requestAnimationFrame), until one is run or cancelled; the others' (the row's) never come.
  const frames = new Map();
  let frameNo = 0;
  globalThis.requestAnimationFrame = (fn) => {
    frameNo += 1;
    if (/marks\.js/.test(new Error().stack)) frames.set(frameNo, fn);
    return frameNo;
  };
  globalThis.cancelAnimationFrame = (n) => { frames.delete(n); };
  const runFrames = (now) => { for (const [n, fn] of [...frames]) { frames.delete(n); fn(now); } };
  const { setMotionPref } = await load('shell/anim.js');
  const { ringCursor } = await load('stages/marks.js');
  const { setTimeout: realTimeout } = globalThis;
  globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 1000 ? 0 : realTimeout(fn, ms, ...rest));
  // The page's own listeners and the window's, to press keys and the pointer on them.
  const added = document.addEventListener;
  const removed = document.removeEventListener;
  const listening = [];
  const listen = (type, fn) => { listening.push([type, fn]); };
  const unlisten = (type, fn) => {
    const at = listening.findIndex(([t, f]) => t === type && f === fn);
    if (at >= 0) listening.splice(at, 1);
  };
  document.addEventListener = globalThis.addEventListener = listen;
  document.removeEventListener = globalThis.removeEventListener = unlisten;
  const fire = (type, e) => {
    for (const [t, fn] of [...listening]) if (t === type) fn(e);
    return e;
  };
  const toasts = new FakeEl('div');
  document.getElementById = (elId) => (elId === 'toasts' ? toasts : null);
  const toasted = () => toasts.children.map((t) => t.textContent);
  document.body = new FakeEl('body');   // where a ripple goes (anim.js)
  const ripples = () => document.body.children.filter((c) => c.className === 'ripple').length;
  try {
    await run({ ctx, hubProp, calls, painted, frames, runFrames, fire, tick, ringCursor, setMotionPref, toasted, ripples });
  } finally {
    delete document.body;
    delete document.getElementById;
    document.addEventListener = added;
    document.removeEventListener = removed;
    delete globalThis.addEventListener;
    delete globalThis.removeEventListener;
    delete globalThis.requestAnimationFrame;
    delete globalThis.cancelAnimationFrame;
    setMotionPref('full');
    globalThis.setTimeout = realTimeout;
    for (const k of Object.keys(canvasParts)) delete FakeEl.prototype[k];
    store.jobs = [];
  }
}

/** The kit's prop on the workbench, its picture there -> the page's parts and the hands to press and read them. */
function mountMarked({ ctx, hubProp, painted, frames, fire }) {
  store.project.props = [structuredClone(hubProp)];
  const host = new FakeEl('div');
  IMAGES.mount(host, ctx);
  const [img] = host.byClass('sv-picture-img');
  Object.assign(img, { complete: true, naturalWidth: 1024, naturalHeight: 768 });
  for (const fn of img.listeners.load || []) fn();
  const [canvas] = host.byClass('mk-canvas');
  const [layer] = host.byClass('mk-circle');
  const [size] = host.byClass('mk-size');
  const point = (type, x, y) => {
    for (const fn of canvas.listeners[type]) fn({ pointerId: 1, pointerType: 'mouse', button: 0, clientX: x, clientY: y, preventDefault() {} });
  };
  const on = (el, k) => painted.filter((r) => r.on === el && (!k || r.k === k));
  return {
    host, canvas, layer, point, on,
    wand: host.byClass('mk-select')[0], brush: host.byClass('mk-smart')[0], glass: host.byClass('mk-wand')[0],
    wipe: host.byClass('mk-wipe')[0], rainbow: host.byClass('mk-rainbow')[0],
    sized(v) {
      size.value = v;
      for (const fn of size.listeners.input) fn();
    },
    /** A stroke through the places given: pressed at the first, let go at the last. */
    stroke(...places) {
      places.forEach(([x, y], i) => point(i === 0 ? 'pointerdown' : i === places.length - 1 ? 'pointerup' : 'pointermove', x, y));
    },
    asked: () => host.byClass('mk-ask').filter((b) => !b.removed),
    key(input, k, more = {}) {
      for (const fn of input.listeners.keydown) fn({ key: k, preventDefault() {}, stopPropagation() {}, ...more });
    },
    typed(input, words) {
      input.value = words;
      for (const fn of input.listeners.input) fn();
    },
    layerNow: () => [layer.hidden, layer.width, frames.size],   // hidden, its width, the frames asked for
    keyOnPage: (k, more = {}) => fire('keydown', {
      type: 'keydown', key: k, defaultPrevented: false,
      preventDefault() { this.defaultPrevented = true; }, stopImmediatePropagation() {}, stopPropagation() {}, ...more,
    }),
  };
}

test('Select Object: the wand, a circle on the picture shown as moving stripes and closed when it is let go, its name asked beside it; the wand stays taken up until its picture is rendered; the job sends the circle alone, magenta, and the circled words', async () => {
  await withMarkedPicture(async (kit) => {
    const { ctx, calls, painted, frames, runFrames, fire, tick, ringCursor, setMotionPref } = kit;
    for (const tool of ['forge', 'qwen']) {
      reset(status(tool));
      const { host, canvas, layer, wand, brush, glass, wipe, rainbow, sized, point, stroke, asked, key, on, layerNow, keyOnPage } = mountMarked(kit);
      if (tool === 'forge') {
        assert.equal(glass.hidden, true, 'Forge Neo starts from a picture: it does not take "I circled ..."');
        IMAGES.unmount();
        continue;
      }
      const [dock] = host.byClass('mk-tools');
      assert.deepEqual(dock.children, [glass, host.byClass('mk-bar')[0]], 'their own glass, left of the markers');
      assert.deepEqual([glass.children, glass.hidden], [[wand, brush], false], 'the wand, then Smart Mask\'s brush');
      assert.ok(/drawn again/.test(brush.title) && !/stays as it is/.test(brush.title), 'the picture is redrawn: no promise that the rest stays');
      const circle = () => stroke([100, 100], [300, 120], [200, 300], [100, 110]);
      const pressed = () => wand.getAttribute('aria-pressed');
      const ctrlZ = () => keyOnPage('z', { ctrlKey: true });
      /** The picture sent, as painted: what comes first, then the circle's parts as "colour width" (its dot's too). */
      const sent = () => {
        const out = on(painted.find((r) => r.k === 'drawImage')?.on);
        const parts = out.filter((r) => r.k === 'arc' || r.k === 'stroke');
        return [out[0]?.k, ...new Set(parts.map((r) => (r.k === 'arc' ? `dot ${r.style} ${2 * r.args[2]}` : `${r.style} ${r.width}`)))];
      };
      /** Where the last line painted on `el` ends. */
      const ends = (el) => on(el, 'quadraticCurveTo').at(-1).args.slice(2);

      // Taken up, whatever marker is picked, the cursor's ring is as wide as the circle goes up: the marker's size, but
      // 0.8 % of the picture's longer side at the least (a 2 px marker: 8 px on this 1024 px picture).
      sized(2);
      host.byClass('mk-black')[0].click();
      assert.equal(canvas.style.cursor, ringCursor(6), 'the marker\'s ring');
      const [markers] = host.byClass('mk-bar');
      wand.click();
      assert.equal(pressed(), 'true');
      assert.equal(canvas.style.cursor, ringCursor(8), 'the circle\'s ring');
      // One thing is lit in that corner: the markers' ring waits while the wand is up, and a marker picked puts
      // the wand down.
      assert.equal(markers.classList.contains('idle'), true);
      host.byClass('mk-white')[0].click();
      assert.deepEqual([pressed(), markers.classList.contains('idle'), canvas.style.cursor], ['false', false, ringCursor(6)]);
      wand.click();
      for (const fn of rainbow.listeners.click) fn();
      assert.equal(pressed(), 'false', 'the marker of any colour too');
      host.byClass('mk-black')[0].click();
      wand.click();

      // Drawn, the circle is a window onto stripes on a layer of its own, moving on frame by frame (half a pair of
      // 12 px in half of 600 ms); the marks are not touched. Let go, it is closed: a line back to where it began.
      painted.length = 0;
      point('pointerdown', 100, 100);
      assert.deepEqual(layerNow(), [false, 1024, 1], 'shown, as large as the picture, its stripes moving');
      runFrames(300);
      const [moved, filled] = on(layer).slice(-2);
      assert.deepEqual([moved.k, moved.args], ['translate', [6, 0]], 'half a pair in half the time');
      assert.deepEqual([filled.k, filled.op, !!filled.style?.stripes], ['fillRect', 'source-atop', true], 'through the circle alone');
      assert.equal(frames.size, 1, 'and on');
      point('pointermove', 300, 120);
      point('pointermove', 200, 300);
      assert.deepEqual(ends(layer), [250, 210], 'open while it is drawn: up to the middle of its last two places');
      point('pointerup', 200, 300);
      assert.deepEqual(ends(layer), [100, 100], 'closed');
      assert.ok(on(layer, 'stroke').length > 0, 'painted on its layer');
      assert.deepEqual(on(canvas), [], 'nothing on the marks');
      assert.equal(pressed(), 'true', 'the wand stays taken up');
      assert.equal(canvas.style.cursor, ringCursor(8));

      // It asks what it holds, "wooden crate" offered, the circle showing meanwhile; Escape lets it go: nothing sent,
      // no mark kept, nothing for Ctrl+Z to take back - and the wand is still taken up.
      let [box] = asked();
      let [input] = box.byClass('mk-ask-input');
      assert.deepEqual([input.value ?? '', input.attrs.placeholder], ['', 'wooden crate'], 'a grey example, nothing typed');
      input.value = '';                             // as a page's empty field has it
      key(input, 'Enter');
      assert.equal(asked().length, 1, 'nothing typed: the question stays');
      assert.match(box.textContent, /What did you circle\?/);
      assert.deepEqual(layerNow(), [false, 1024, 1], 'the circle shows while it is asked about');
      keyOnPage('Escape');
      assert.deepEqual([asked().length, pressed()], [1, 'true'], 'Escape is the box\'s while it asks');
      key(input, 'Escape');
      assert.deepEqual(asked(), []);
      assert.deepEqual(layerNow(), [true, 1, 0], 'Escape: the circle goes, its frames with it');
      assert.equal(pressed(), 'true', 'not the wand');
      await tick();
      assert.deepEqual(calls, []);
      assert.equal(wipe.hidden, true, 'the circle is no mark of the picture');
      assert.equal(ctrlZ().defaultPrevented, false, 'nor a step Ctrl+Z takes back');

      // A press elsewhere lets the question and the circle go; one on the picture begins the next circle at once.
      circle();
      await tick();                                 // the question listens once the circle's own press is over
      fire('pointerdown', { target: host, stopPropagation() {} });
      assert.deepEqual([asked().length, pressed(), ...layerNow()], [0, 'true', true, 1, 0], 'a press elsewhere');
      circle();
      await tick();
      fire('pointerdown', { target: canvas, stopPropagation() {} });
      point('pointerdown', 400, 400);
      assert.deepEqual([asked().length, ...layerNow()], [0, false, 1024, 1], 'a press on the picture: the next circle');
      point('pointermove', 500, 420);
      point('pointerup', 500, 420);
      assert.equal(asked().length, 1, 'asked about in its turn');
      point('pointerdown', 400, 400);               // at once, before the question listens for a press
      assert.deepEqual([asked().length, ...layerNow()], [0, false, 1024, 1], 'the same');
      point('pointerup', 400, 400);
      assert.equal(asked().length, 1);
      // Put down by the artist - its button, or Escape once no box asks - the wand lets go of what it drew.
      wand.click();
      assert.deepEqual([pressed(), asked().length, ...layerNow()], ['false', 0, true, 1, 0], 'put down: the circle and its question go');
      wand.click();
      point('pointerdown', 100, 100);
      wand.click();
      assert.deepEqual(layerNow(), [true, 1, 0], 'put down while circling');
      point('pointerup', 100, 110);
      assert.deepEqual(asked(), [], 'nothing asked');
      wand.click();
      assert.deepEqual([keyOnPage('Escape').defaultPrevented, pressed()], [true, 'false'], 'Escape puts it down, and is spent on that');
      assert.equal(keyOnPage('Escape').defaultPrevented, false, 'the next one is the page’s');

      // Named: the picture goes up with that circle alone, closed, magenta and 8 px wide from the black marker, and
      // the job says what was circled, kept as it faces, in place of the prop's words. The wand is put down.
      wand.click();
      circle();
      [box] = asked();
      [input] = box.byClass('mk-ask-input');
      input.value = '  wooden barrel ';
      painted.length = 0;
      key(input, 'Enter');
      assert.deepEqual(layerNow(), [true, 1, 0], 'answered: the circle goes');
      assert.equal(pressed(), 'false', 'rendered: the wand is put down');
      await tick();
      assert.deepEqual(asked(), []);
      assert.deepEqual(sent(), ['drawImage', 'dot #ff00ff 8', '#ff00ff 8'], 'the picture, then the circle over it');
      assert.deepEqual(ends(painted.find((r) => r.k === 'drawImage').on), [100, 100], 'closed there too');
      assert.deepEqual(calls.map((c) => `${c[0]} ${c[1]}`), ['POST viewport', 'POST jobs']);
      assert.deepEqual(calls.at(-1)[2].params, {
        from_viewport: 'viewport/viewport.png', use_references: false,
        prompt: 'I circled the wooden barrel with the magenta marker - put it in the center, large and prominently '
          + 'visible, close zoom, keeping its style and its orientation. Remove everything else around it and apply a '
          + 'gray background.',
      });
      assert.equal(store.project.props[0].images.settings.prompt, 'a hut', 'the prop keeps its own words');
      assert.equal(wipe.hidden, true);

      // Reduced animations: the same stripes, standing still - no frame asked for. A 20 px marker of any colour: the
      // circle goes up 20 px wide, magenta still.
      store.jobs = [];
      IMAGES.update(ctx);
      calls.length = 0;
      setMotionPref('reduced');
      sized(20);
      for (const fn of rainbow.listeners.click) fn();
      wand.click();
      painted.length = 0;
      circle();
      assert.deepEqual(layerNow(), [false, 1024, 0], 'shown, no frame asked for');
      assert.ok(on(layer, 'fillRect').length > 0, 'its stripes');
      assert.deepEqual([...new Set(on(layer, 'translate').map((r) => r.args[0]))], [0], 'where they stand');
      [input] = asked()[0].byClass('mk-ask-input');
      input.value = 'oak chair';
      painted.length = 0;
      key(input, 'Enter');
      await tick();
      assert.deepEqual(sent(), ['drawImage', 'dot #ff00ff 20', '#ff00ff 20']);
      assert.match(calls.at(-1)[2].params.prompt, /^I circled the oak chair with the magenta marker - /);
      setMotionPref('full');

      // A circle under way when the workbench goes: it goes with it, and its frames.
      store.jobs = [];
      IMAGES.update(ctx);
      wand.click();
      point('pointerdown', 100, 100);
      assert.equal(frames.size, 1);
      IMAGES.unmount();
      assert.deepEqual(layerNow(), [true, 1, 0], 'nothing left behind');
    }
  });
});

test('maskedPrompt: the words go as they are when they speak of the mask, misspelt too, else after "for the masked area do: "', async () => {
  const { maskedPrompt } = await load('stages/images.js');
  for (const words of ['make the mask brass', 'Masked part: brass', 'the msak brass', 'MSAKED brass', 'masck brass',
    'mascked brass', 'the maskid bit', 'amsk brass', 'maks brass', 'masking it brass']) {
    assert.equal(maskedPrompt(words), words, words);
  }
  for (const words of ['make it brass', 'damask cloth', 'makes it brass', 'a mas k']) {
    assert.equal(maskedPrompt(words), `for the masked area do: ${words}`, words);
  }
});

test('Smart Mask: the brush beside the wand; strokes add up to a mask the picture shows through on the page, each one let go rings the words on the right; while it is on, From Scratch waits and Re-Render Viewport (or Enter in the words) sends it as flat magenta over the picture and alone, the picture it lies on and the words for the mask', async () => {
  await withMarkedPicture(async (kit) => {
    const { ctx, hubProp, calls, painted, tick, ringCursor, ripples } = kit;
    reset(status('qwen'));
    const { host, canvas, layer, wand, brush, wipe, sized, stroke, point, asked, key, typed, on, layerNow, keyOnPage } = mountMarked(kit);
    const pressed = () => [wand, brush].map((b) => b.getAttribute('aria-pressed'));
    const [scratch] = host.byClass('sv-render');
    const [viewport] = host.byClass('sv-rerender');
    const [words] = host.byClass('sv-prompt');
    const posts = () => calls.filter((c) => c[0] === 'POST').map((c) => c[1]);

    // One tool at a time; the brush's ring is twice the circle's (a 10 px marker: 20 px).
    sized(10);
    wand.click();
    brush.click();
    assert.deepEqual(pressed(), ['false', 'true']);
    assert.equal(canvas.style.cursor, ringCursor(20));
    assert.deepEqual([scratch.disabled, wipe.hidden], [false, true]);

    // A stroke: on the selection's layer, which the picture shows through, never on the marks. No box asks: the
    // words on the right ring, From Scratch waits, the wipe takes the mask off.
    painted.length = 0;
    stroke([100, 100], [300, 120], [300, 120]);
    assert.deepEqual([layer.hidden, layer.classList.contains('veil'), on(canvas).length], [false, true, 0]);
    assert.deepEqual([...new Set(on(layer, 'stroke').map((r) => r.width))], [20]);
    assert.deepEqual([asked().length, ripples(), scratch.disabled, viewport.disabled], [0, 1, true, false]);
    assert.match(scratch.title, /mask is on the picture/);
    assert.deepEqual([wipe.hidden, wipe.getAttribute('aria-label')], [false, 'Take the mask off']);

    // No words: nothing is sent, the words ring again.
    typed(words, '  ');
    calls.length = 0;
    viewport.click();
    await tick();
    assert.deepEqual([posts(), ripples(), layer.hidden], [[], 2, false]);
    assert.match(kit.toasted().at(-1), /what should be in the masked area/);

    // A stroke more: the mask grows (the first stroke stays on its layer), and the words ring.
    typed(words, 'make it brass');
    const before = on(layer, 'stroke').length;
    point('pointerdown', 200, 300);
    point('pointermove', 260, 340);
    point('pointerup', 260, 340);
    assert.ok(on(layer, 'stroke').length > before);
    assert.equal(ripples(), 3);

    // Re-Render Viewport: the picture goes up with both strokes over it in flat magenta - nothing of what they cover
    // shows -, and the job's words are the prop's after "for the masked area do: ". The brush stays taken up; the
    // mask is gone with its picture.
    painted.length = 0;
    calls.length = 0;
    viewport.click();
    await tick();
    const out = painted.find((r) => r.k === 'drawImage').on;
    assert.deepEqual(on(out, 'drawImage').map((r) => r.alpha), [1], 'the picture, once');
    assert.deepEqual([on(out, 'arc').length, [...new Set(on(out, 'stroke').map((r) => `${r.style} ${r.width} ${r.alpha}`))]],
      [2, ['#ff00ff 20 1']], 'both strokes straight onto it, magenta, covering');
    // The mask alone goes up beside it - the same strokes, white on black, no picture under them -, and the job
    // names the picture they lie on: the hub keeps that picture everywhere else.
    const alone = painted.find((r) => r.k === 'fillRect' && r.on !== out).on;
    assert.deepEqual(on(alone, 'fillRect').map((r) => [r.style, ...r.args]), [['#000000', 0, 0, 1024, 768]]);
    assert.deepEqual([on(alone, 'drawImage').length, on(alone, 'arc').length, [...new Set(on(alone, 'stroke').map((r) => `${r.style} ${r.width}`))]],
      [0, 2, ['#ffffff 20']]);
    const sent = calls.filter((c) => c[0] === 'POST');
    assert.deepEqual(posts(), ['viewport', 'jobs']);
    assert.deepEqual(sent[0][2].parts.map(([field, , name]) => `${field}:${name}`), ['file:viewport.png', 'mask:mask.png']);
    assert.deepEqual(sent[1][2].params, {
      from_viewport: 'viewport/viewport.png', use_references: false,
      prompt: 'for the masked area do: make it brass',
      viewport_mask_of: 'images/0001.png',
    });
    assert.deepEqual([pressed(), asked().length, ...layerNow()], [['false', 'true'], 0, true, 1, 0]);
    assert.equal(store.jobs.length, 1);
    IMAGES.update(ctx);                               // its job shows: the markers hide while it runs
    assert.deepEqual(pressed(), ['false', 'true'], 'taken up through its job too');

    // The next mask, words that speak of it, Enter in them: they go as they are, and the picture is re-rendered -
    // From Scratch waits while a mask is on.
    store.jobs = [];
    IMAGES.update(ctx);
    assert.equal(scratch.disabled, false, 'no mask: From Scratch again');
    stroke([100, 100], [300, 120], [300, 120]);
    typed(words, 'paint the masked part copper');
    calls.length = 0;
    key(words, 'Enter');
    await tick();
    assert.deepEqual(posts(), ['viewport', 'jobs']);
    assert.equal(calls.filter((c) => c[0] === 'POST')[1][2].params.prompt, 'paint the masked part copper');
    store.jobs = [];
    IMAGES.update(ctx);

    // Escape takes the mask off and leaves the brush taken up; Escape then puts it down. The wipe takes it off too.
    stroke([100, 100], [300, 120], [300, 120]);
    keyOnPage('Escape');
    assert.deepEqual([pressed(), ...layerNow(), scratch.disabled, wipe.hidden], [['false', 'true'], true, 1, 0, false, true]);
    keyOnPage('Escape');
    assert.deepEqual(pressed(), ['false', 'false']);
    brush.click();
    stroke([100, 100], [300, 120], [300, 120]);
    wipe.click();
    assert.deepEqual([pressed(), layer.hidden, scratch.disabled], [['false', 'true'], true, false]);

    // Another picture of the prop: the mask on the one before goes.
    stroke([100, 100], [300, 120], [300, 120]);
    store.project.props[0].images = { ...store.project.props[0].images, chosen: 'images/0002.png', candidates: ['images/0001.png', 'images/0002.png'] };
    IMAGES.update(ctx);
    const [img] = host.byClass('sv-picture-img');
    Object.assign(img, { complete: true, naturalWidth: 1024, naturalHeight: 768 });
    for (const fn of img.listeners.load || []) fn();
    assert.deepEqual([pressed(), layer.hidden, scratch.disabled], [['false', 'true'], true, false]);
    // On the black canvas of a prop without a picture there is none to keep: the canvas with its mask alone goes up.
    store.project.props.push({ ...structuredClone(hubProp), id: 'w_3', name: 'Shed', images: { ...hubProp.images, chosen: null, candidates: [] } });
    IMAGES.update({ ...ctx, selectedId: 'w_3' });
    host.byClass('mk-smart')[0].click();
    for (const [type, x, y] of [['pointerdown', 100, 100], ['pointermove', 300, 120], ['pointerup', 300, 120]]) {
      for (const fn of host.byClass('mk-canvas')[0].listeners[type]) fn({ pointerId: 1, pointerType: 'mouse', button: 0, clientX: x, clientY: y, preventDefault() {} });
    }
    typed(host.byClass('sv-prompt')[0], 'a shed');
    calls.length = 0;
    host.byClass('sv-rerender')[0].click();
    await tick();
    assert.deepEqual(posts(), ['viewport', 'jobs']);
    const shed = calls.filter((c) => c[0] === 'POST');
    assert.deepEqual(shed[0][2].parts.map(([field, , name]) => `${field}:${name}`), ['file:viewport.png']);
    assert.deepEqual(Object.keys(shed[1][2].params), ['from_viewport', 'use_references', 'prompt']);
    IMAGES.unmount();
  });
});

test('the pointer on a Render button "with examples" rings the example pictures that go along; a marker or a tool taken up ends the slicing', async () => {
  await withMarkedPicture(async (kit) => {
    reset(status('qwen'));
    const { host, wand, brush, rainbow } = mountMarked(kit);
    const rung = [];
    const { animate } = FakeEl.prototype;
    FakeEl.prototype.animate = function (frames, timing) { rung.push(this); return animate.call(this); };
    const hover = (btn) => {
      rung.length = 0;
      for (const fn of btn.listeners.pointerenter || []) fn({});
      return rung.filter((el) => el.classList.contains('sv-ref')).length;
    };
    try {
      const [scratch] = host.byClass('sv-render');
      const [viewport, viewportEx] = host.byClass('sv-rerender');
      assert.deepEqual([scratch, viewport, viewportEx].map(hover), [0, 0, 0], 'no examples: none to ring');
      store.project.props[0].images = { ...store.project.props[0].images, references: ['refs/a.png', 'refs/b.png'] };
      IMAGES.update(kit.ctx);
      assert.deepEqual([scratch, viewport, viewportEx].map(hover), [2, 0, 2], 'the buttons that say "with examples"');
    } finally {
      FakeEl.prototype.animate = animate;
    }

    // Slicing, then a tool, the brush or a marker taken up: slicing ends.
    const [toggle] = host.byClass('sl-toggle');
    const slicing = () => toggle.getAttribute('aria-pressed');
    for (const take of [() => wand.click(), () => brush.click(), () => host.byClass('mk-black')[0].click(), () => rainbow.click()]) {
      toggle.click();
      assert.equal(slicing(), 'true');
      take();
      assert.equal(slicing(), 'false');
    }
    IMAGES.unmount();
  });
});

test('a tileable picture, slid by its offset or not: what is rendered from it says how far, so the hub lays the result where that picture lies', async () => {
  await withMarkedPicture(async (kit) => {
    const { ctx, hubProp, calls, tick } = kit;
    reset(status('qwen'));
    hubProp.images.settings.tileable = true;
    const { host, wand, brush, stroke, asked, key, typed } = mountMarked(kit);
    const [slider] = host.byClass('sv-shift');
    const [viewport] = host.byClass('sv-rerender');
    const sent = () => calls.filter((c) => c[1] === 'jobs').at(-1)[2].params;
    const over = () => {
      store.jobs = [];
      IMAGES.update(ctx);
    };
    // As it is: the picture itself, slid by nothing - said all the same: a tileable picture is laid onto the one it
    // was made from (the model lays it some pixels off).
    viewport.click();
    await tick();
    assert.deepEqual(sent(), { from_viewport: 'images/0001.png', use_references: false, viewport_shift: 0 });
    over();
    // A tenth right and down: it goes up as it shows, and the job says by how much (the result is slid back as far).
    slider.value = '10';
    for (const fn of slider.listeners.input) fn();
    await tick();
    viewport.click();
    await tick();
    assert.deepEqual(sent(), { from_viewport: 'viewport/viewport.png', use_references: false, viewport_shift: 0.1 });
    over();
    // Smart Mask's picture is that same picture, changed in a part: slid back too.
    brush.click();
    stroke([100, 100], [300, 120], [300, 120]);
    typed(host.byClass('sv-prompt')[0], 'more moss');
    viewport.click();
    await tick();
    assert.deepEqual([sent().viewport_shift, sent().viewport_mask_of], [0.1, 'images/0001.png']);
    over();
    brush.click();
    // A circled object's picture is the object alone, placed anew: it lies as it is drawn.
    wand.click();
    stroke([100, 100], [300, 120], [200, 300], [100, 110]);
    asked()[0].byClass('mk-ask-input')[0].value = 'pebble';
    key(asked()[0].byClass('mk-ask-input')[0], 'Enter');
    await tick();
    assert.deepEqual(Object.keys(sent()), ['from_viewport', 'use_references', 'prompt']);
    over();
    // Left and up, and back to the middle.
    slider.value = '-30';
    for (const fn of slider.listeners.input) fn();
    await tick();
    viewport.click();
    await tick();
    assert.equal(sent().viewport_shift, -0.3);
    over();
    slider.value = '0';
    for (const fn of slider.listeners.input) fn();
    await tick();
    viewport.click();
    await tick();
    assert.deepEqual(sent(), { from_viewport: 'images/0001.png', use_references: false, viewport_shift: 0 });
    over();
    // Unticked, the picture is one like any other: nothing is said, nothing is moved.
    const [tick_] = host.byClass('sv-tileable')[0].all((e) => e.classList.contains('switch'));
    tick_.input.checked = false;
    for (const fn of tick_.input.listeners.change) fn();
    await tick();
    viewport.click();
    await tick();
    assert.deepEqual(sent(), { from_viewport: 'images/0001.png', use_references: false });
    IMAGES.unmount();
  });
});

test('"R" beside Tileable: the chosen picture, turned a quarter clockwise pixel for pixel, becomes a new picture of the prop; a Shape that fits the picture turns with it', async () => {
  await withMarkedPicture(async (kit) => {
    const { flushSettings } = await load('actions.js');
    const { hubProp, calls, tick } = kit;
    const before = { fetch: globalThis.fetch, create: document.createElement };
    try {
      reset(status('qwen'));
      hubProp.images.settings = { prompt: 'moss', tileable: true, width: 1216, height: 832 };
      const { host } = mountMarked(kit);
      const [turn] = host.byClass('sv-turn');
      const [row] = host.byClass('sv-tileable-row');
      const [tileTick, , slider] = row.children;
      assert.deepEqual([row.children.indexOf(turn), slider.classList.contains('sv-shift')], [1, true], 'between the tick and the slider');
      assert.deepEqual([turn.textContent, turn.hidden, turn.disabled], ['R', false, false]);
      const landscape = () => host.byClass('sv-shapes')[0].buttons.map((b) => b.classList.contains('on'));
      assert.deepEqual(landscape(), [false, false, true]);

      // The hub's pictures as files, a picture of 1216 x 832 in each; the canvas a turn draws on; the upload.
      const read = [];
      globalThis.fetch = async (path, init = {}) => {
        if (init.method || !/\.png/.test(path)) return before.fetch(path, init);
        read.push(String(path).match(/images\/[\w.]+/)[0]);
        return { ok: true, status: 200, blob: async () => ({ type: 'image/png' }) };
      };
      let bitmap = { width: 1216, height: 832, close() {} };
      globalThis.createImageBitmap = async () => bitmap;
      const drawn = [];
      document.createElement = (tag) => {
        const el = before.create(tag);
        if (tag !== 'canvas') return el;
        const did = [];
        drawn.push({ el, did });
        el.getContext = () => new Proxy({}, {
          get: (t, k) => (...args) => { did.push([k, ...args]); },
          set: (t, k, v) => { did.push([k, v]); return true; },
        });
        return el;
      };
      const uploads = [];
      globalThis.XMLHttpRequest = class {
        constructor() { this.upload = {}; }
        open(method, path) { this.path = path; }
        send(form) {
          uploads.push([this.path.split('/props/')[1], form.parts]);
          const made = `images/upload_00${uploads.length}.png`;
          Object.assign(hubProp.images, { chosen: made, candidates: [...hubProp.images.candidates, made] });
          Object.assign(this, { status: 200, responseText: JSON.stringify(hubProp) });
          queueMicrotask(() => this.onload());
        }
      };
      globalThis.File = class { constructor(parts, name, opts) { Object.assign(this, { parts, name, type: opts?.type }); } };

      // Pressed: the chosen picture is read and drawn turned on a canvas of its sides swapped - what was its left edge
      // along the top, no smoothing -, and that goes up as a picture of the prop's own.
      turn.click();
      assert.equal(turn.disabled, true, 'one turn at a time');
      await tick();
      assert.deepEqual(read, ['images/0001.png']);
      assert.deepEqual([drawn.length, drawn[0].el.width, drawn[0].el.height], [1, 832, 1216]);
      assert.deepEqual(drawn[0].did, [['imageSmoothingEnabled', false], ['setTransform', 0, 1, -1, 0, 832, 0], ['drawImage', bitmap, 0, 0]]);
      assert.deepEqual(uploads.map(([path, parts]) => [path, parts.map(([field, file, name]) => [field, file.name, file.type, file.parts[0], name])]),
        [['w_1/images/upload', [['files', 'rotated.png', 'image/png', { png: true }, 'rotated.png']]]]);
      assert.equal(turn.disabled, false);
      // The Shape fitted the picture: it is the turned picture's now, shown and saved.
      assert.deepEqual(landscape(), [false, true, false]);
      await flushSettings();
      assert.deepEqual(calls.filter((c) => c[0] === 'PATCH').at(-1)[2], { images: { settings: { width: 832, height: 1216 } } });

      // Pressed again, it is the turned picture that turns. A square one leaves the Shape alone.
      bitmap = { width: 1024, height: 1024, close() {} };
      const patches = calls.filter((c) => c[0] === 'PATCH').length;
      turn.click();
      await tick();
      assert.deepEqual([read.at(-1), drawn.at(-1).el.width, uploads.length], ['images/upload_001.png', 1024, 2]);
      await flushSettings();
      assert.deepEqual([landscape(), calls.filter((c) => c[0] === 'PATCH').length], [[false, true, false], patches]);

      // Not tileable: no "R", as no slider.
      tileTick.input.checked = false;
      for (const fn of tileTick.input.listeners.change) fn();
      assert.deepEqual([turn.hidden, slider.hidden], [true, true]);
    } finally {
      await flushSettings();
      IMAGES.unmount();
      Object.assign(globalThis, { fetch: before.fetch });
      document.createElement = before.create;
      delete globalThis.createImageBitmap;
    }
  });
});

test('Smart Mask over a large picture or a tileable one: the part its change takes shows round the mask while it is brushed - the patch of a whole redraw, a tileable picture\'s window -, and fades when let go; a mask too large for a window is not sent until it fits again', async () => {
  await withMarkedPicture(async (kit) => {
    const { flushSettings } = await load('actions.js');
    try {
      const { ctx, hubProp, calls, painted, tick } = kit;
      const { MASK_TOO_LARGE } = await load('stages/marks.js');
      reset(status('qwen'));
      hubProp.images.settings = { prompt: 'a hut', width: 1024, height: 768 };   // the shape of the picture shown
      const { host, layer, brush, wipe, sized, stroke, point, on, keyOnPage } = mountMarked(kit);
      const [win] = host.byClass('mk-window');
      const [k1, , k2] = host.byClass('seg-btn').filter((b) => /^[\d.]+K$/.test(b.textContent));
      const [viewport] = host.byClass('sv-rerender');
      const refused = () => viewport.title === MASK_TOO_LARGE;   // the button says why it sends nothing
      const pct = (v, of) => `${(v / of) * 100}%`;
      const part = (open, [x0, y0, x1, y1], [w, hh]) => [['mk-window-part', ...open].join(' '),
        `left: ${pct(x0, w)}; top: ${pct(y0, hh)}; width: ${pct(x1 - x0, w)}; height: ${pct(y1 - y0, hh)}`];
      const shows = () => [win.hidden, win.classList.contains('fade'), win.children.map((p) => [p.className, p.style.cssText])];
      sized(10);                                  // the mask's brush: 20 px
      brush.click();

      // At 1K the picture is drawn whole: no window, never refused.
      stroke([100, 100], [300, 120], [300, 120]);
      assert.deepEqual(shows(), [true, false, []]);
      assert.equal(refused(), false);
      keyOnPage('Escape');

      // At 2K (2048 x 1536) all of the picture is drawn anew and a patch of that laid onto it. From the press on the
      // patch shows round what is brushed - a stroke 90 .. 310 across and 90 .. 130 down, twice that in the request's
      // pixels -: 1024 px a side, off the picture's left side and its top, which the stroke lies near, by a third of
      // the way to the stroke; at full strength, in no colour of its own (dashes of white and black) ...
      const S = [2048, 1536];
      k2.click();
      point('pointerdown', 100, 100);
      assert.deepEqual(shows().slice(0, 2), [false, false]);
      point('pointermove', 300, 120);
      assert.deepEqual(shows(), [false, false, [part([], [59, 54, 1083, 1078], S)]]);
      assert.ok(!win.style.color);
      // ... and fades out when the stroke is let go; the next stroke brings it back at full strength, round both:
      // wider now.
      point('pointerup', 300, 120);
      assert.deepEqual(shows().slice(0, 2), [false, true]);
      assert.equal(refused(), false);
      point('pointerdown', 350, 200);
      assert.deepEqual(shows().slice(0, 2), [false, false]);
      point('pointermove', 400, 220);
      point('pointerup', 400, 220);
      assert.deepEqual(shows(), [false, true, [part([], [59, 54, 1191, 1078], S)]]);

      // A stroke across most of the picture: the patch is most of it, off its sides still, and nothing is refused.
      stroke([50, 50], [1000, 700], [1000, 700]);
      assert.deepEqual(shows(), [false, true, [part([], [21, 24, 2045, 1498], S)]]);
      assert.equal(refused(), false);

      // Ctrl+Z takes the last stroke back: the patch shows a moment, and the rest is painted again on the selection's
      // layer. Once more: one stroke left.
      painted.length = 0;
      assert.equal(keyOnPage('z', { ctrlKey: true }).defaultPrevented, true);
      assert.deepEqual(shows(), [false, true, [part([], [59, 54, 1191, 1078], S)]]);
      assert.equal(on(layer, 'arc').length, 2, 'two strokes left, painted again');
      keyOnPage('z', { ctrlKey: true });
      assert.deepEqual(shows(), [false, true, [part([], [59, 54, 1083, 1078], S)]]);

      // Rendered: the mask goes up with its one stroke left - the frame is nothing of the pictures sent - and goes.
      painted.length = 0;
      viewport.click();
      await tick();
      assert.deepEqual(calls.filter((c) => c[0] === 'POST').map((c) => c[1]), ['viewport', 'jobs'], 'the Resolution saved meanwhile');
      const out = painted.find((r) => r.k === 'drawImage').on;
      assert.deepEqual([on(out, 'drawImage').length, on(out, 'arc').length], [1, 1], 'the picture, one stroke over it');
      assert.equal(painted.some((r) => r.on === win || win.children.includes(r.on)), false);
      assert.deepEqual(shows().slice(0, 1), [true]);
      store.jobs = [];
      IMAGES.update(ctx);

      // A TILEABLE picture's change is drawn in a window of it, and a window holds only so much. A stroke across most of
      // a 2K one: too large. The frame, round the least it would take (what was brushed and 96 px each way, as the image
      // server moves the picture: round its edge here); Re-Render Viewport says why and sends nothing.
      k2.click();                                 // (the stand-in hub answered with the prop as it had it)
      const [tileTick] = host.byClass('sv-tileable')[0].all((e) => e.classList.contains('switch'));
      tileTick.input.checked = true;
      for (const fn of tileTick.input.listeners.change) fn();
      stroke([50, 50], [1000, 700], [1000, 700]);
      assert.deepEqual(shows(), [false, true, [part(['open-b'], [0, 1520, 2048, 1536], S), part(['open-t'], [0, 0, 2048, 1516], S)]]);
      assert.equal(refused(), true);
      viewport.click();
      await tick();
      assert.equal(calls.filter((c) => c[0] === 'POST' && c[1] === 'jobs').length, 1, 'no job for it');
      assert.equal(win.hidden, false, 'the mask stays');
      assert.equal(kit.toasted().at(-1), MASK_TOO_LARGE);
      // At 1K a window holds it: allowed; back at 2K, refused again.
      k1.click();
      assert.equal(refused(), false);
      k2.click();
      assert.deepEqual([win.hidden, refused()], [false, true]);
      // The wipe empties a mask too large, and the next one fits.
      wipe.click();
      assert.deepEqual([win.hidden, refused()], [true, false]);
      stroke([100, 100], [300, 120], [300, 120]);
      assert.equal(refused(), false);
      keyOnPage('Escape');

      // Tileable at 1K: a stroke at the right edge lies in the middle of the picture as the image server moves it, its
      // window round it there - all of the height, 768 px across - and laid back: it leaves the right edge and comes
      // back at the left, without an outline where the edge cuts it.
      k1.click();
      stroke([1000, 300], [1010, 300], [1010, 300]);
      assert.deepEqual(shows(), [false, true, [part(['open-r'], [621, 0, 1024, 768], [1024, 768]),
        part(['open-l'], [0, 0, 365, 768], [1024, 768])]]);
      assert.equal(refused(), false);
    } finally {
      await flushSettings();   // the settings set here go to this test's hub, not the next one's
      IMAGES.unmount();
    }
  });
});

test('the row lets a removed picture go at once and tells the hub one removal after another: Ctrl+Z puts each back', async () => {
  const ctx = { selectedId: 'p_1', select() {}, goto() {} };
  const tick = async (n = 4) => { for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0)); };
  const pic = (n) => `images/000${n}.png`;
  const hubProp = {
    id: 'p_1', name: 'Barrel', thumbs: {},
    images: { chosen: pic(3), candidates: [pic(1), pic(2), pic(3)], views: [], settings: { prompt: 'a barrel' } },
  };
  reset(status('qwen'));
  store.project.props = [structuredClone(hubProp)];
  const hub = pictureHub(hubProp);
  const keys = [];
  const { addEventListener } = document;
  document.addEventListener = (type, fn, capture) => { if (type === 'keydown') keys.push([fn, capture === true]); };
  const ctrlZ = () => {
    const e = {
      type: 'keydown', key: 'z', ctrlKey: true, defaultPrevented: false, stopped: false,
      preventDefault() { e.defaultPrevented = true; }, stopImmediatePropagation() { e.stopped = true; },
    };
    for (const [fn] of [...keys.filter((k) => k[1]), ...keys.filter((k) => !k[1])]) if (!e.stopped) fn(e);
    return e;
  };
  const host = new FakeEl('div');
  IMAGES.mount(host, ctx);
  document.addEventListener = addEventListener;
  const tiles = () => host.byClass('sv-cand');
  const row = () => tiles().map((t) => t.all((x) => x.tagName === 'IMG')[0].attrs.src.match(/images\/\d+\.png/)[0]);
  assert.deepEqual(row(), [pic(1), pic(2), pic(3)]);

  // Two quick clicks on the × under the pointer: each tile goes at once, the next one moving up under it; the hub
  // hears of the second after the first.
  let release;
  hub.hold = new Promise((r) => { release = r; });
  tiles()[1].byClass('sv-cand-x')[0].click();
  assert.deepEqual(row(), [pic(1), pic(3)], 'gone from the row before the hub answers');
  tiles()[1].byClass('sv-cand-x')[0].click();
  assert.deepEqual(row(), [pic(1)]);
  await tick();
  assert.deepEqual(hub.calls, [['remove', { path: pic(2) }]], 'the second waits for the first');
  hub.hold = null;
  release();
  await tick();
  IMAGES.update(ctx);
  assert.deepEqual(hub.calls, [['remove', { path: pic(2) }], ['remove', { path: pic(3) }]]);
  assert.deepEqual(row(), [pic(1)]);

  // Ctrl+Z, twice: each comes back at its place, chosen as it was.
  ctrlZ();
  await tick();
  IMAGES.update(ctx);
  assert.deepEqual(hub.calls.at(-1), ['restore', { path: pic(3), index: 1, views: [], chosen: true }]);
  ctrlZ();
  await tick();
  IMAGES.update(ctx);
  assert.deepEqual(hub.calls.at(-1), ['restore', { path: pic(2), index: 1, views: [], chosen: false }]);
  assert.deepEqual(row(), [pic(1), pic(2), pic(3)]);
  IMAGES.unmount();
});

test('each picture of the row keeps the words and settings last set while it was chosen: a click brings them back', async () => {
  const ctx = { selectedId: 'p_1', select() {}, goto() {} };
  const tick = () => new Promise((r) => setTimeout(r, 0));
  const pic = (n) => `images/000${n}.png`;
  const hubProp = {
    id: 'p_1', name: 'Barrel', thumbs: {},
    images: { chosen: pic(1), candidates: [pic(1), pic(2)], views: [], settings: { prompt: 'a barrel', seed: 5 } },
  };
  reset(status('qwen'));
  store.project.ui = {};
  store.project.props = [structuredClone(hubProp)];
  const hub = pictureHub(hubProp);
  const host = new FakeEl('div');
  IMAGES.mount(host, ctx);
  const [words] = host.byClass('sv-prompt');
  const type = (text) => {
    words.value = text;
    for (const fn of words.listeners.input) fn();
  };
  const click = async (n) => {
    host.byClass('sv-cand')[n - 1].click();
    await tick();
    await tick();
    IMAGES.update(ctx);
  };
  type('a barrel with iron bands');
  await click(2);
  assert.equal(words.value, 'a barrel with iron bands', 'a picture that kept nothing changes nothing');
  type('a mossy barrel');
  await click(1);
  assert.equal(words.value, 'a barrel with iron bands');
  assert.equal(store.project.props[0].images.settings.prompt, 'a barrel with iron bands', 'the prop takes them');
  await click(2);
  assert.equal(words.value, 'a mossy barrel');
  assert.deepEqual(Object.keys(store.project.ui.picture_settings.p_1), [pic(1), pic(2)]);
  assert.equal(store.project.ui.picture_settings.p_1[pic(1)].seed, 5);
  assert.deepEqual(hub.calls.map((c) => c[0]), ['select', 'select', 'select']);
  IMAGES.unmount();
});

test('the prompt library: "+prompt" and the shown prompts under the words; a click puts the words at the text cursor; the panel adds, edits and saves', async () => {
  const { insertAt, promptBar, libraryPanel } = await load('stages/promptlib.js');
  const { emit } = await load('store.js');
  const sent = [];
  globalThis.fetch = async (path, init = {}) => {
    if (init.method === 'PUT') sent.push([path, JSON.parse(init.body)]);
    return { status: 200, text: async () => JSON.stringify({ prompts: store.promptLibrary }) };
  };
  const field = {
    value: 'a crate, wooden', selectionStart: 7, selectionEnd: 7, events: [],
    focus() { this.focused = true; }, dispatchEvent(e) { this.events.push(e.type); },
  };
  insertAt(field, 'hand-painted style');
  assert.deepEqual([field.value, field.selectionStart, field.events, field.focused],
    ['a crate hand-painted style, wooden', 26, ['input'], true], 'a space before it, the cursor after it');

  store.promptLibrary = [{ id: 'a', name: 'painted', text: 'soft light', shown: true }, { id: 'b', name: 'hidden', text: 'x', shown: false }];
  let opened = 0;
  const bar = promptBar(field, () => { opened += 1; });
  emit('prompts');
  assert.deepEqual(bar.children.map((c) => c.textContent), ['painted', '+prompt'], 'only the shown ones, "+prompt" last');
  bar.children[0].click();
  assert.equal(field.value, 'a crate hand-painted style soft light, wooden');
  bar.children[1].click();
  assert.equal(opened, 1);

  const panel = libraryPanel();
  panel.open();
  const rows = () => panel.el.byClass('pl-row');
  assert.equal(rows().length, 2);
  panel.el.byClass('pl-add')[0].click();
  assert.equal(rows().length, 3);
  const [name] = rows()[2].byClass('pl-name');
  name.value = 'a long name';
  for (const fn of name.listeners.input) fn();
  const [text] = rows()[2].byClass('pl-text');
  text.value = 'game-ready, clean silhouette';
  for (const fn of text.listeners.input) fn();
  assert.equal(store.promptLibrary[2].name, 'a long nam', 'at the end of the list, 10 characters at most');
  assert.deepEqual(bar.children.map((c) => c.textContent), ['painted', 'a long nam', '+prompt'], 'a new prompt shows at once');
  rows()[1].byClass('pl-bin')[0].click();
  const closers = panel.el.byClass('pl-close');
  assert.equal(closers.length, 2, 'one in each top corner');
  closers[1].click();
  assert.equal(panel.el.hidden, true);
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(sent.at(-1)[0], '/api/prompt-library');
  assert.deepEqual(sent.at(-1)[1].prompts.map((p) => [p.name, p.text, p.shown]),
    [['painted', 'soft light', true], ['a long nam', 'game-ready, clean silhouette', true]], 'saved when the panel closes');
  bar.destroy();
  panel.destroy();
});

// ---- "Tileable" (More): the tick, the offset slider beside it, the look from afar ---------------------------------

test('Tileable sits under More; ticked, an offset slider stands to its right and a hint under it, and the wheel turned out past the whole picture shows it from afar among copies of itself', async () => {
  reset(status('qwen'));
  store.project.props = [{ id: 'p_1', name: 'Bricks', images: { chosen: 'images/0001.png', candidates: ['images/0001.png'], settings: { prompt: 'bricks' } } }];
  const host = new FakeEl('div');
  IMAGES.mount(host, { selectedId: 'p_1', select() {}, goto() {} });
  const [box] = host.byClass('sv-tileable');
  const [tick] = box.all((e) => e.classList.contains('switch'));
  const [slider] = box.byClass('sv-shift');
  const [hint] = box.byClass('hint');
  const [stage] = host.byClass('sv-imgstage');
  const [tiling] = host.byClass('sv-tiling');
  const [area] = host.byClass('sv-picture');      // laid out at the whole picture's zoom (picview.js measures it)
  Object.assign(area, { offsetWidth: 800, offsetHeight: 600 });
  const wheel = (deltaY) => {
    const e = { deltaY, prevented: false, preventDefault() { e.prevented = true; } };
    for (const fn of stage.listeners.wheel) fn(e);
    return e.prevented;
  };
  const afar = () => stage.classList.contains('afar');
  assert.equal(host.all((e) => e.tagName === 'DETAILS').some((d) => d.all((e) => e === box).length), true, 'under "More"');
  assert.deepEqual([tick.textContent, hint.textContent], ['Tileable', 'scrollwheel on canvas to zoom out']);
  assert.deepEqual([slider.getAttribute('min'), slider.getAttribute('max'), slider.getAttribute('step'), slider.value], ['-50', '50', '10', 0],
    'tenths of the picture, nought in the middle');
  // Off: no slider, no hint, and the wheel out at the whole picture goes no further (the view's: picview.js).
  assert.deepEqual([!!tick.input.checked, slider.hidden, hint.hidden, tiling.hidden], [false, true, true, true]);
  assert.deepEqual([wheel(120), afar()], [true, false]);
  // The tick is the last thing in the side list: ticked, the list scrolls on until the hint under it stands whole
  // above the list's fading bottom edge.
  const [list] = host.byClass('sv-side-scroll');
  const scrolled = [];
  list.getBoundingClientRect = () => ({ top: 122, bottom: 878 });
  list.scrollBy = (how) => scrolled.push(how.top);
  hint.getBoundingClientRect = () => ({ top: 884, bottom: 902 });
  tick.input.checked = true;
  for (const fn of tick.input.listeners.change) fn();
  assert.equal(store.project.props[0].images.settings.tileable, true, 'the prop\'s own setting');
  assert.deepEqual([slider.hidden, hint.hidden, tiling.hidden], [false, false, false]);
  assert.deepEqual(scrolled, [52], 'the hint ends at 902, plus 28 of fade and air, against the list ending at 878');
  // The wheel towards the artist, out at the whole picture: from afar; away: back. The page does not scroll meanwhile.
  assert.deepEqual([wheel(120), afar()], [true, true]);
  assert.deepEqual([wheel(120), afar()], [true, true]);
  assert.deepEqual([wheel(-120), afar()], [true, false]);
  // The slider slides the picture - a canvas of the page's stands in for it - and the copies with it; unticked,
  // everything is back as it was.
  const shown = () => host.byClass('sv-picture-img').map((e) => e.tagName);
  assert.deepEqual(shown(), ['IMG']);
  slider.value = '-30';
  for (const fn of slider.listeners.input) fn();
  assert.equal(tiling.style['--shift'], '0.7', 'three tenths up and left is seven tenths down and right');
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(shown(), ['CANVAS'], 'once the picture is there to draw');
  wheel(120);
  tick.input.checked = false;
  for (const fn of tick.input.listeners.change) fn();
  assert.deepEqual([slider.hidden, hint.hidden, tiling.hidden, afar(), slider.value, tiling.style['--shift'], shown()],
    [true, true, true, false, 0, '0', ['IMG']]);
  // Unticking scrolls nothing, nor does a tick whose hint shows already.
  hint.getBoundingClientRect = () => ({ top: 700, bottom: 718 });
  tick.input.checked = true;
  for (const fn of tick.input.listeners.change) fn();
  assert.deepEqual(scrolled, [52]);
  IMAGES.unmount();
});

test('a picture shifted for tiling is drawn four times, so that what leaves one side comes back at the other', () => {
  const drawn = [];
  const { createElement } = document;
  document.createElement = (tag) => Object.assign(createElement(tag), { getContext: () => ({ drawImage: (image, x, y) => drawn.push([x, y]) }) });
  try {
    const image = { naturalWidth: 1000, naturalHeight: 500 };
    const right = shiftedCanvas(image, 0.2);
    assert.deepEqual([right.tagName, right.width, right.height, right.className], ['CANVAS', 1000, 500, 'sv-picture-img']);
    assert.deepEqual(drawn, [[-800, -400], [200, -400], [-800, 100], [200, 100]], 'a fifth to the right and down');
    drawn.length = 0;
    shiftedCanvas(image, -0.5);
    assert.deepEqual(drawn, [[-500, -250], [500, -250], [-500, 250], [500, 250]], 'half: the edges cross in the middle, either way');
  } finally {
    document.createElement = createElement;
  }
});
