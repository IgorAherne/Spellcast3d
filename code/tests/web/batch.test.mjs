// A job's props in the page (no browser), read from where the hub says each one stands (job.props): the 3D
// generator's jobs, whose props take turns step by step - a prop that waits for its next step (the tile's
// still bar, the workbench, the button), the status line with the step, the prop the workbench shows (it does
// not hop after the work, and a finished model loads only for the prop on it), results and failures told the
// moment they happen, jobs gone with a restarted hub - and the jobs of the other tools, shown as before.
// Also Settings without the 3D generator's low-video-memory switch.
//   node --test code/tests/web/batch.test.mjs
// (code/tests/test_web.py runs every code/tests/web/*.test.mjs when Node.js is installed)
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
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

const { store, bustFor, jobFor, jobLoading, jobStopping, FIRST_START, TOOL_ETA } = await load('store.js');
const { openProject, startJob } = await load('actions.js');
const { busyLayer, focusBox, jobState } = await load('stages/common.js');
const { badgeOf, barOf, createPropList } = await load('shell/proplist.js');
const { headline } = await load('shell/statusline.js');
const { openSettings } = await load('shell/settings.js');
const { fasterSwitch } = await load('stages/faster.js');
const { default: MESH } = await load('stages/mesh.js');
const { default: RETOPO } = await load('stages/retopo.js');
const { PARTS, batchJob, hubJob } = await import('./hub_job.mjs');

const IDS = ['qwen', 'forge', 'pixal3d', 'imesh', 'lato2', 'triflow'];
const tools = (states = {}) => Object.fromEntries(IDS.map((t) => [t, {
  state: states[t] || 'stopped', message: '', installed: true, supported: true, uses_gpu: t !== 'imesh',
}]));
const NAMES = ['Barrel', 'Lantern', 'Chest', 'Anvil', 'Well', 'Cart'];
/** The prop p_<n> with a picture; `made`: it has its 3D model, made from that picture. */
const prop = (n, made = false, extra = {}) => ({
  id: `p_${n}`, name: NAMES[n - 1], images: { chosen: 'images/0001.png', candidates: ['images/0001.png'], views: [] },
  mesh: made ? { glb: 'mesh/mesh.glb', source: 'images/0001.png', stats: { seconds: 190, seed: 7 } } : { glb: null },
  retopo: { obj: null, settings: {} }, ...extra,
});

function reset({ jobs = [], props = [prop(1), prop(2), prop(3), prop(4), prop(5)], states = {} } = {}) {
  store.connected = true;
  store.jobs = jobs;
  store.jobErrors = {};
  store.status = { tools: tools(states), install: { running: false, components: {} }, multiview: { available: true } };
  store.project = { id: 'Props', name: 'Props', props, ui: {} };
  store.bust = {};
  store.openStamp = 7;
  toasts.replaceChildren();
}

/** A stage view's context, as the shell gives it; the workbench is not moving unless `moving`. */
const ctxOf = (selectedId = null, extra = {}) => ({
  selectedId, stageId: 'mesh', project: store.project, boxes: store.project.props, select() {}, goto() {}, ...extra,
});

// ---- where a prop stands in a job ---------------------------------------------------------------------

test('a prop\'s job state is what the hub says of it: working, resting, waiting, and none once the job is through with it', () => {
  const job = batchJob('j1', {
    p_1: ['working', 0.2, 'Sculpting the shape · step 3 of 12'], p_2: 'resting', p_3: 'waiting', p_4: 'done', p_5: 'failed', p_6: 'cancelled',
  }, { part: 'lr', position: 1, of: 3 });
  reset({ jobs: [job] });
  assert.deepEqual(jobFor('mesh', 'p_1'), { job, phase: 'running', fraction: 0.2, message: 'Sculpting the shape · step 3 of 12' });
  assert.deepEqual(jobFor('mesh', 'p_2'), { job, phase: 'resting', fraction: 0.29, message: 'Waiting for the next step' });
  assert.deepEqual(jobFor('mesh', 'p_3'), { job, phase: 'waiting', fraction: 0, message: '' });
  for (const over of ['p_4', 'p_5', 'p_6']) assert.equal(jobFor('mesh', over), null, over);
  assert.equal(jobFor('retopo', 'p_1'), null, 'another stage');
  assert.equal(jobFor('mesh', 'p_9'), null, 'not a prop of the job');

  // Nothing else is read: a job that does not say where its props stand has no prop in work, whatever its
  // prop_ids, current_prop and progress.done would have meant.
  const { props: _none, ...bare } = job;
  reset({ jobs: [bare] });
  assert.equal(jobFor('mesh', 'p_1'), null);
  reset({ jobs: [{ ...job, props: { p_4: job.props.p_4 }, progress: { ...job.progress, done: 0 } }] });
  assert.equal(jobFor('mesh', 'p_4'), null, 'done is done, whatever progress.done counts');
  assert.equal(jobFor('mesh', 'p_1'), null);

  // Before the job runs, every prop of it waits with the job: in the queue, or for its tool - with the job's words.
  const waits = (status, message) => ({ ...batchJob('j1', { p_1: 'waiting', p_2: 'waiting' }), status, current_prop: null, progress: { message } });
  reset({ jobs: [waits('queued', 'Waiting for the jobs before it…')] });
  assert.deepEqual([jobFor('mesh', 'p_2').phase, jobFor('mesh', 'p_2').message], ['queued', 'Waiting for the jobs before it…']);
  reset({ jobs: [waits('starting_tool', 'Starting Pixal3D…')] });
  assert.deepEqual([jobFor('mesh', 'p_1').phase, jobFor('mesh', 'p_1').fraction], ['starting', 0]);
  // A job that has ended has none of its props in work.
  reset({ jobs: [{ ...job, status: 'cancelled' }] });
  assert.equal(jobFor('mesh', 'p_1'), null);
  // The newest job that still has work on a prop is its job (the hub lists the newest first): made again.
  const again = { ...batchJob('j2', { p_5: 'waiting' }), status: 'queued', current_prop: null };
  reset({ jobs: [again, job] });
  assert.equal(jobFor('mesh', 'p_5').job, again);
  assert.equal(jobFor('mesh', 'p_1').job, job);
});

test('the fixtures\' jobs say where their props stand as the hub derives it for a job that runs them one after another', () => {
  const states = (job, opts) => Object.values(hubJob(job, opts).props).map((p) => p.state);
  const job = (status, done = 0, extra = {}) => ({ id: 'j1', stage: 'retopo', status, prop_ids: ['a', 'b', 'c'], progress: { done, total: 3 }, ...extra });
  assert.deepEqual(states(job('queued')), ['waiting', 'waiting', 'waiting']);
  assert.deepEqual(states(job('starting_tool')), ['waiting', 'waiting', 'waiting']);
  assert.deepEqual(states(job('running')), ['working', 'waiting', 'waiting']);
  assert.deepEqual(states(job('running', 2, { errors: { a: 'No.' } })), ['failed', 'done', 'working']);
  assert.deepEqual(states(job('done', 3)), ['done', 'done', 'done']);
  assert.deepEqual(states(job('error', 3, { errors: { c: 'No.' } })), ['done', 'done', 'failed']);
  assert.deepEqual(states(job('cancelled', 2), { cutShort: 1 }), ['done', 'cancelled', 'cancelled'], 'the prop cut short and all after it');
  assert.deepEqual(states(job('running', 2), { cutShort: 1 }), ['done', 'cancelled', 'working'], 'cut short, the job still winding down');
  assert.deepEqual(states(job('cancelled', 1)), ['done', 'cancelled', 'cancelled'], 'cancelled between two props');
  assert.deepEqual(states(job('cancelled')), ['cancelled', 'cancelled', 'cancelled'], 'cancelled in the queue');
  const noStart = { a: 'It could not start.', b: 'It could not start.', c: 'It could not start.' };
  assert.deepEqual(states(job('error', 0, { errors: noStart })), ['failed', 'failed', 'failed'], 'its tool could not start');
  assert.deepEqual(states(job('error', 1)), ['done', 'cancelled', 'cancelled'], 'a job that broke off');
  // The prop in work has its own share of the job's progress and the job's words; `part` is the 3D generator's.
  const running = hubJob({ ...job('running', 1), progress: { done: 1, total: 3, fraction: 0.5, message: 'Solving 40%' } });
  assert.deepEqual(running.props.b, { state: 'working', fraction: 0.5, message: 'Solving 40%' });
  assert.deepEqual([running.props.a.fraction, running.props.c.fraction, running.props.c.message], [1, 0, '']);
  assert.equal(running.part, null);
  assert.equal(hubJob(job('running')).props.a.fraction, 0);
});

// ---- the tile ---------------------------------------------------------------------------------------

test('a prop that waits for its next step: a still bar where it got to - no pulse, no clock, not dimmed', async () => {
  reset({ jobs: [batchJob('j1', { p_1: ['working', 0.2, 'Sculpting the shape · step 3 of 12'], p_2: 'resting', p_3: 'waiting' }, { part: 'lr' })] });
  const [working, resting, waiting] = store.project.props.slice(0, 3).map((p) => jobState('mesh', p));
  assert.deepEqual(working, { state: 'processing', message: 'Sculpting the shape · step 3 of 12', fraction: 0.2 });
  assert.deepEqual(resting, { state: 'resting', fraction: 0.29,
    message: 'Waiting for the next step · Barrel:\nsculpting the shape · step 3 of 12' }, 'it says who is at work meanwhile, and how far');
  assert.deepEqual(waiting, { state: 'waiting', message: 'Waiting for its turn' });
  assert.deepEqual(MESH.status(store.project.props[1]), resting, 'the 3D stage says the same of it');

  assert.equal(badgeOf('mesh', 'resting'), '', 'no mark: the bar says it');
  assert.equal(badgeOf('mesh', 'waiting'), 'waiting', 'not begun: the clock, as before');
  assert.equal(barOf(resting), 0.29);
  assert.equal(barOf({ state: 'resting' }), 0, 'never a sliding bar: nothing happens to it meanwhile');
  assert.equal(barOf({ state: 'resting', fraction: 7 }), 1);
  assert.equal(barOf(working), 0.2);
  assert.equal(barOf({ state: 'processing', fraction: null }), 'unknown', 'being made, how far is not known: it slides');
  for (const state of ['waiting', 'idle', 'done', 'error', 'stale', 'review', 'unreachable']) assert.equal(barOf({ state, fraction: 0.5 }), '', state);
  assert.equal(barOf(null), '');

  // The style sheet: the bar shows for a resting tile, and nothing else is said of one (the pulse is the
  // working tile's, the dimmed picture the waiting one's).
  const css = await readFile(new URL('shell.css', STATIC), 'utf8');
  const rules = css.replace(/\/\*[\s\S]*?\*\//g, '').split('}').map((r) => r.trim()).filter((r) => r.includes('.st-resting'));
  assert.deepEqual(rules, ['.st-processing .tile-bar, .st-resting .tile-bar { display: block;']);
});

test('the props list draws each prop of a job as it stands: bar and pulse, still bar, clock, check, "!"', () => {
  const job = batchJob('j1', {
    p_1: ['working', 0.2, 'Sculpting the shape · step 3 of 12'], p_2: 'resting', p_3: 'waiting', p_4: 'done', p_5: 'failed',
  }, { part: 'lr', errors: { p_5: 'The picture has no object in it.' } });
  reset({ jobs: [job], props: [prop(1), prop(2), prop(3), prop(4, true), prop(5)] });
  store.jobErrors['mesh:p_5'] = job.errors.p_5;
  const list = createPropList({ select() {} });
  list.render(ctxOf(), MESH, 'p_1');
  const tile = (n) => list.tile(`p_${n}`);
  const states = (n) => tile(n).className.split(' ').filter((c) => c.startsWith('st-'));
  const bar = (n) => tile(n).byClass('tile-bar')[0];
  const fill = (n) => tile(n).byClass('tile-bar-fill')[0].style.transform;
  assert.deepEqual([1, 2, 3, 4, 5].map(states), [['st-processing'], ['st-resting'], ['st-waiting'], ['st-done'], ['st-error']]);
  assert.deepEqual([1, 2, 3, 4, 5].map((n) => tile(n).dataset.badge), [undefined, undefined, 'waiting', 'done', 'error']);
  assert.deepEqual([fill(1), fill(2)], ['scaleX(0.2)', 'scaleX(0.29)']);
  assert.equal(bar(2).classList.contains('indeterminate'), false);
  assert.deepEqual([1, 2, 3, 5].map((n) => tile(n).getAttribute('aria-label')), [
    'Barrel · Sculpting the shape · step 3 of 12', 'Lantern · Waiting for the next step · Barrel:\nsculpting the shape · step 3 of 12', 'Chest · Waiting for its turn',
    'Well · The picture has no object in it.',
  ]);

  // The next step: every bar goes on by one step, the working prop is another one, and nothing else moves.
  store.jobs = [batchJob('j1', { p_1: 'resting', p_2: ['working', 0.4], p_3: 'resting', p_4: 'done', p_5: 'failed' }, { part: 'hr', position: 2 })];
  list.render(ctxOf(), MESH, 'p_1');
  assert.deepEqual([1, 2, 3].map(states), [['st-resting'], ['st-processing'], ['st-resting']]);
  assert.deepEqual([fill(1), fill(2), fill(3)], ['scaleX(0.55)', 'scaleX(0.4)', 'scaleX(0.55)']);
  assert.deepEqual([1, 2, 3].map((n) => tile(n).dataset.badge), [undefined, undefined, undefined], 'the clock is gone once a prop is begun');
});

test('a stage may put a box to tick beside each tile: a click on it goes to the stage alone, a click on the tile selects the prop', () => {
  reset({ props: [prop(1), prop(2), prop(3)] });
  const heard = [];
  const list = createPropList({ select: (id) => heard.push(`select ${id}`), check: (id) => heard.push(`check ${id}`) });
  const chosen = new Set(['p_2']);
  // The stage offers a box for the first two props only.
  const stage = { ...MESH, tileCheck: (p) => (p.id === 'p_3' ? null : { on: chosen.has(p.id), title: chosen.has(p.id) ? 'Chosen' : 'Choose it' }) };
  list.render(ctxOf(), stage, 'p_1');
  const box = (n) => list.tile(`p_${n}`).byClass('tile-check')[0];
  assert.deepEqual([1, 2, 3].map((n) => box(n).hidden), [false, false, true]);
  assert.deepEqual([1, 2].map((n) => [box(n).getAttribute('role'), box(n).getAttribute('aria-checked'), box(n).title]),
    [['checkbox', 'false', 'Choose it'], ['checkbox', 'true', 'Chosen']]);
  assert.equal(box(1).getAttribute('aria-label'), 'Choose it: Barrel');
  assert.equal(list.el.classList.contains('checks'), true, 'the list makes room for the boxes');
  let stopped = 0;
  box(1).fire('click', { stopPropagation() { stopped += 1; } });
  assert.deepEqual([heard, stopped], [['check p_1'], 1], 'the tile under it hears nothing: the prop is not selected');
  list.tile('p_1').fire('click', {});
  assert.deepEqual(heard, ['check p_1', 'select p_1']);
  chosen.add('p_1');
  list.render(ctxOf(), stage, 'p_1');
  assert.equal(box(1).getAttribute('aria-checked'), 'true');
  // A stage without boxes: none shows, and the list is as narrow as ever.
  list.render(ctxOf(), MESH, 'p_1');
  assert.deepEqual([1, 2, 3].map((n) => box(n).hidden), [true, true, true]);
  assert.equal(list.el.classList.contains('checks'), false);
});

test('a tileable texture says so on its tile, top left, in every stage', () => {
  const texture = (settings) => ({ ...prop(1), images: { ...prop(1).images, settings } });
  reset({ props: [texture({ tileable: true }), prop(2)] });
  const list = createPropList({ select() {} });
  const tag = (n) => list.tile(`p_${n}`).byClass('tile-tag')[0];
  for (const stage of [MESH, RETOPO]) {
    list.render(ctxOf(), stage, 'p_1');
    assert.deepEqual([tag(1).textContent, tag(1).hidden, tag(2).hidden], ['tileable', false, true]);
    assert.equal(list.tile('p_1').byClass('tile-pic')[0].children.includes(tag(1)), true, 'on the picture');
  }
  // Unticked, or with FLUX.3 (which draws no tileable picture), it is an object again.
  for (const settings of [{ tileable: false }, { tileable: true, qwen_model: 'flux3' }]) {
    reset({ props: [texture(settings), prop(2)] });
    list.render(ctxOf(), MESH, 'p_1');
    assert.equal(tag(1).hidden, true, JSON.stringify(settings));
  }
});

test('a stage may put number boxes on each tile: what is typed there keeps to a number and goes to the stage alone', () => {
  reset({ props: [prop(1), prop(2)] });
  const heard = [];
  const list = createPropList({
    select: (id) => heard.push(`select ${id}`), sizes: (id, numbers) => heard.push(`sizes ${id} ${numbers.join(',')}`),
  });
  const sizes = { p_1: [4, 22] };
  const offer = { count: 3, min: 0.1, max: 99, decimals: 2, unit: 'k verts', title: 'Sizes' };
  const stage = { ...MESH, tileSizes: (p) => (p.id === 'p_2' ? null : { values: sizes[p.id], ...offer }) };
  list.render(ctxOf(), stage, 'p_1');
  const row = (n) => list.tile(`p_${n}`).byClass('tile-sizes')[0];
  const boxes = (n) => row(n).byClass('tile-size');
  assert.deepEqual([row(1).hidden, row(2).hidden], [false, true]);
  assert.deepEqual(boxes(1).map((b) => [b.value, b.maxLength, b.getAttribute('inputmode')]),
    [['4', 5, 'decimal'], ['22', 5, 'decimal'], ['', 5, 'decimal']], 'room for "12.25"');
  assert.equal(row(1).textContent, 'k verts');
  assert.equal(boxes(1)[0].getAttribute('aria-label'), 'Sizes (box 1): Barrel');
  // Typed: digits and one dot or comma, two digits before it and two after; a box that holds no size yet says nothing.
  const [first, second, third] = boxes(1);
  third.value = '6x75';
  third.fire('input');
  first.value = ',3';
  first.fire('input');
  second.value = '0.';
  second.fire('input');
  assert.deepEqual([first.value, second.value, third.value], [',3', '0.', '67']);
  assert.deepEqual(heard, ['sizes p_1 4,22,67', 'sizes p_1 0.3,22,67', 'sizes p_1 0.3,67']);
  // A click, a double click or a key in them stays theirs: the prop is not selected, the page hears no key.
  let stopped = 0;
  for (const type of ['click', 'dblclick']) row(1).fire(type, { stopPropagation() { stopped += 1; } });
  third.fire('keydown', { key: 'ArrowDown', stopPropagation() { stopped += 1; } });
  assert.deepEqual([stopped, heard.filter((h) => h.startsWith('select'))], [3, []]);
  // Drawn again with the numbers the stage kept, in order, while one box is typed in: no box is rewritten. The second
  // one holds no size for the moment ("0."), so the third one's 67 is the stage's second number - moved into the
  // second box's place it would be gone from its own, and the next key would lose it.
  sizes.p_1 = [0.3, 67];
  document.activeElement = second;
  list.render(ctxOf(), stage, 'p_1');
  assert.deepEqual(boxes(1).map((b) => b.value), [',3', '0.', '67']);
  second.value = '0.5';
  second.fire('input');
  assert.equal(heard.at(-1), 'sizes p_1 0.3,0.5,67', 'nothing lost on the way from "22" to "0.5"');
  // Left, the boxes show the numbers kept, without gaps; a box that says its number already keeps the artist's way of
  // writing it (",3" is not made "0.3").
  second.value = '0.';
  second.fire('input');
  // (Left for another box of the same tile: the others show them, the one now typed in keeps its text.)
  document.activeElement = third;
  second.fire('blur');
  assert.deepEqual(boxes(1).map((b) => b.value), [',3', '67', '67']);
  document.activeElement = null;
  third.fire('blur');
  assert.deepEqual(boxes(1).map((b) => b.value), [',3', '67', '']);
  third.value = '0.05';
  third.fire('input');
  third.fire('blur');
  assert.deepEqual(boxes(1).map((b) => b.value), [',3', '67', ''], 'under the smallest size: as if empty, once left');
  // Numbers the stage has from elsewhere are written with a dot.
  sizes.p_1 = [0.25, 12.5];
  list.render(ctxOf(), stage, 'p_1');
  assert.deepEqual(boxes(1).map((b) => b.value), ['0.25', '12.5', '']);
  list.render(ctxOf(), MESH, 'p_1');
  assert.equal(row(1).hidden, true, 'a stage without them: none');
});

test('the call to action leads down even with nothing to make, unless its stage holds the props back', async () => {
  const { createColumn } = await load('shell/column.js');
  reset({ props: [prop(1)] });
  const column = createColumn({ list: { el: new FakeEl('div'), tail: new FakeEl('div') }, statusLine: { el: new FakeEl('div') } },
    { go() {}, cta() {}, plus() {}, row() {} });
  const cta = column.el.byClass('cta')[0];
  const draw = (gate, hover = 'What it does.') => {
    column.render({ ...RETOPO, ctaEnabled: () => gate, ctaHover: hover }, { ...ctxOf(), stageId: 'retopo' });
    return [cta.classList.contains('disabled'), cta.getAttribute('aria-disabled'), column.el.byClass('sr-only')[0].textContent];
  };
  assert.deepEqual(draw({ enabled: true }), [false, 'false', 'What it does.']);
  assert.deepEqual(draw({ enabled: false, hint: 'Nothing yet.' }), [false, 'false', 'What it does.'], 'it still leads down');
  assert.deepEqual(draw({ enabled: false, blocked: true, hint: 'Pick them first.' }, 'Pick them first.'),
    [true, 'true', 'Pick them first.'], 'held back: disabled, and its hover says why');
});

// ---- the workbench ------------------------------------------------------------------------------------

test('the layer over the workbench: a begun prop shows how far it is, also while it waits; what the tool loads is said', () => {
  const layer = busyLayer('mesh');
  const [text] = layer.byClass('sv-busy-text');
  const [bar] = layer.byClass('progress');
  const width = () => bar.byClass('progress-fill')[0].style.width;
  const sliding = () => bar.classList.contains('indeterminate');
  const show = (job, n = 1) => {
    reset({ jobs: [job] });
    layer.sync(store.project.props[n - 1]);
    return [layer.hidden, text.textContent, sliding() ? 'slides' : width()];
  };
  assert.deepEqual(show(batchJob('j1', { p_1: ['working', 0.4, 'Refining shape details · step 7 of 12'], p_2: 'resting' }, { part: 'hr' })),
    [false, 'Refining shape details · step 7 of 12', '40%']);
  assert.deepEqual(show(batchJob('j1', { p_1: 'resting', p_2: ['working', 0.4] }, { part: 'hr', position: 2 })),
    [false, 'Waiting for the next step · Lantern:\nrefining shape details (3/12)', '40%'], 'the words and the bar of the prop at work');
  assert.deepEqual(show(batchJob('j1', { p_1: 'waiting', p_2: ['working', 0.01] }, { part: 'cutout', position: 2 })),
    [false, 'Waiting for its turn · Lantern:\nremoving the background (3/12)', '1%'], 'a prop not begun yet names the one at work too');
  // Between two steps the 3D generator reads the next model from disk: no prop is on it, the job says what loads.
  const loading = batchJob('j1', { p_1: ['resting', 0.06], p_2: 'waiting' },
    { part: 'ss', loading: 'shape model', progress: { fraction: 0.03, message: 'Loading the shape model…' } });
  assert.equal(jobLoading(loading), 'Loading the shape model…');
  assert.deepEqual(show(loading), [false, 'Loading the shape model…', '6%']);
  assert.deepEqual(show(loading, 2), [false, 'Loading the shape model…', 'slides']);
  // Only what the hub calls loading: the words of a job are not read, and a prop on the tool says its own.
  assert.equal(jobLoading({ ...loading, loading: '' }), '', 'the job’s message says the same, and is not read');
  assert.equal(jobLoading(batchJob('j1', { p_1: ['working', 0.1, 'Loading…'], p_2: 'resting' }, { part: 'ss' })), '');
  // Stop was pressed: every prop the job has begun says the job is being stopped, where it stands; one not
  // begun still waits (as in a job of any other tool).
  const cancelling = 'Cancelling… (the current prop may take a moment to stop)';
  const stopping = batchJob('j1', { p_1: ['working', 0.4, 'Refining shape details · step 7 of 12'], p_2: 'resting', p_3: 'waiting' },
    { part: 'hr', stopping: true, progress: { fraction: 0.3, message: cancelling } });
  assert.equal(jobStopping(stopping), cancelling);
  assert.deepEqual(show(stopping), [false, cancelling, '40%']);
  assert.deepEqual(show(stopping, 2), [false, cancelling, '55%']);
  assert.deepEqual(show(stopping, 3), [false, 'Waiting for its turn', 'slides']);
  assert.equal(jobStopping(loading), '');
  // The job is through with the prop: nothing lies over its model (or its error) any more.
  assert.equal(show(batchJob('j1', { p_1: 'done', p_2: ['working', 0.8] }, { part: 'finish', position: 2 }))[0], true);
  assert.equal(show(batchJob('j1', { p_1: 'failed', p_2: 'resting' }, { part: 'hr' }))[0], true);
  // Before the job runs: its tool loads, as before.
  const starting = { ...batchJob('j1', { p_1: 'waiting' }), status: 'starting_tool', current_prop: null, progress: { message: 'Starting Pixal3D…' } };
  reset({ jobs: [starting], states: { pixal3d: 'starting' } });
  layer.sync(store.project.props[0]);
  assert.deepEqual([text.textContent, sliding()], ['Loading the 3D generator… (about half a minute)', true]);
});

test('the 3D workbench: "Making 3D…" for every prop the job has begun, and the prop on it stays while the work goes from prop to prop', () => {
  const never = new Promise(() => {});   // the workbench "moves": a model waits for it, so none is ever read here
  const ctx = (selectedId = null) => ctxOf(selectedId, { moving: true, settled: never });
  const host = new FakeEl('div');
  const label = () => host.byClass('sv-wide')[0].textContent;
  const busy = () => host.byClass('sv-busy')[0];
  const busyText = () => (busy().hidden ? null : busy().byClass('sv-busy-text')[0].textContent);
  const loadsModel = () => !host.byClass('sv-viewer-loading')[0].hidden;
  const note = () => host.byClass('sv-subtitle')[0].textContent;
  const at = (states, opts, props = store.project.props) => {
    store.jobs = [batchJob('j1', states, opts)];
    store.project.props = props;
    MESH.update(ctx(selected));
    return [MESH.shownId(), label(), busyText(), loadsModel()];
  };
  let selected = null;

  // The job starts: nothing is selected, the workbench shows the first prop the job is not through with.
  reset({ jobs: [batchJob('j1', { p_1: 'waiting', p_2: 'waiting', p_3: 'waiting' }, { part: 'cutout', position: 0 })] });
  MESH.mount(host, ctx());
  assert.deepEqual([MESH.shownId(), label(), busyText()], ['p_1', 'Waiting…', 'Waiting for its turn']);
  assert.deepEqual(at({ p_1: ['working', 0.1, 'Blocking out the shape · step 4 of 12'], p_2: 'resting', p_3: 'resting' }, { part: 'ss' }),
    ['p_1', 'Making 3D…', 'Blocking out the shape · step 4 of 12', false]);
  assert.equal(note(), '', 'the workbench follows nothing: the layer over it says where this prop stands');
  // The work hops to the next prop, and the next: the workbench stays, the button still says it is being made.
  assert.deepEqual(at({ p_1: 'resting', p_2: ['working', 0.1], p_3: 'resting' }, { part: 'ss', position: 2 }),
    ['p_1', 'Making 3D…', 'Waiting for the next step · Lantern:\nblocking out the shape (3/12)', false],
    'the layer says what the generator is at meanwhile, so the wait does not look like a load');
  assert.deepEqual(at({ p_1: 'resting', p_2: 'resting', p_3: ['working', 0.1] }, { part: 'ss', position: 3 }),
    ['p_1', 'Making 3D…', 'Waiting for the next step · Chest:\nblocking out the shape (3/12)', false]);
  // Another prop is finished: the workbench does not go there, and its model is not read.
  const made2 = [prop(1), prop(2, true), prop(3), prop(4), prop(5)];
  assert.deepEqual(at({ p_1: 'resting', p_2: 'done', p_3: ['working', 0.8] }, { part: 'finish', position: 2 }, made2),
    ['p_1', 'Making 3D…', 'Waiting for the next step · Chest:\nbuilding the 3D mesh (3/12)', false]);
  // The prop on the workbench is finished: it stays, and now its model loads there.
  const made12 = [prop(1, true), prop(2, true), prop(3), prop(4), prop(5)];
  assert.deepEqual(at({ p_1: 'done', p_2: 'done', p_3: ['working', 0.9] }, { part: 'finish', position: 3 }, made12),
    ['p_1', 'Make 3D again', null, true]);
  assert.equal(note(), 'Made in 3 min 10 s · Seed 7');
  // The artist opens another prop: a finished one shows its model, one in work its progress.
  selected = 'p_2';
  assert.deepEqual(at({ p_1: 'done', p_2: 'done', p_3: ['working', 0.9] }, { part: 'finish', position: 3 }, made12),
    ['p_2', 'Make 3D again', null, true]);
  selected = 'p_3';
  assert.deepEqual(at({ p_1: 'done', p_2: 'done', p_3: ['working', 0.9, 'Saving the GLB file'] }, { part: 'finish', position: 3 }, made12),
    ['p_3', 'Making 3D…', 'Saving the GLB file', false]);
  MESH.unmount();
});

test('which prop a stage shows: the selected one, else the one shown while a running job has it, else the first it is not through with', () => {
  const pick = (stage, selectedId, previousId) => {
    const { box, auto } = focusBox({ selectedId }, stage, { needsWork: (p) => p.id === 'p_4', previousId });
    return [box?.id ?? null, auto];
  };
  reset();
  assert.deepEqual(pick('mesh', null, null), ['p_4', true], 'no job: the first prop that needs work');
  assert.deepEqual(pick('mesh', null, 'p_2'), ['p_2', true], 'no job: the one shown before');
  assert.deepEqual(pick('mesh', 'p_3', 'p_2'), ['p_3', false]);

  reset({ jobs: [batchJob('j1', { p_2: 'done', p_3: 'resting', p_4: ['working', 0.2], p_5: 'failed' }, { part: 'lr' })] });
  assert.deepEqual(pick('mesh', null, null), ['p_3', true], 'the first the job is not through with, in the artist\'s order');
  assert.deepEqual(pick('mesh', null, 'p_3'), ['p_3', true], 'it stays while another prop is on the tool');
  assert.deepEqual(pick('mesh', null, 'p_2'), ['p_2', true], 'a finished prop of the job stays too');
  assert.deepEqual(pick('mesh', null, 'p_5'), ['p_5', true], 'and one that failed');
  assert.deepEqual(pick('mesh', null, 'p_1'), ['p_3', true], 'shown before, but not a prop of the job: the work is shown');
  assert.deepEqual(pick('mesh', 'p_1', 'p_3'), ['p_1', false], 'what the artist picks always wins');
  assert.deepEqual(pick('retopo', null, 'p_1'), ['p_1', true], 'a job of another stage moves nothing here');

  // A job that only waits (in the queue, for its tool) moves nothing; a job that has ended neither.
  reset({ jobs: [{ ...batchJob('j1', { p_3: 'waiting' }), status: 'starting_tool', current_prop: null }] });
  assert.deepEqual(pick('mesh', null, 'p_1'), ['p_1', true]);
  reset({ jobs: [{ ...batchJob('j1', { p_3: 'done' }), status: 'done', current_prop: null }] });
  assert.deepEqual(pick('mesh', null, null), ['p_4', true]);

  // A job of another tool, whose props run one after another: the same rule from the same states.
  const retopo = (done) => hubJob({ id: 'j2', stage: 'retopo', tool: 'lato2', status: 'running', prop_ids: ['p_1', 'p_2', 'p_3'],
    current_prop: `p_${done + 1}`, progress: { done, total: 3, fraction: done / 3 } });
  reset({ jobs: [retopo(0)] });
  assert.deepEqual(pick('retopo', null, null), ['p_1', true], 'the prop being retopologized');
  reset({ jobs: [retopo(1)] });
  assert.deepEqual(pick('retopo', null, 'p_1'), ['p_2', true], 'it went on to the next prop: the workbench follows');
  assert.deepEqual(pick('retopo', null, 'p_5'), ['p_2', true]);
});

// ---- the status line ----------------------------------------------------------------------------------

test('the status line of the 3D generator: the step, the props in it, the prop - one clock for the job', () => {
  const at = (states, opts) => { store.jobs = [batchJob('j1', states, opts)]; return headline(); };
  reset();
  const line = at({ p_1: 'resting', p_2: ['working', 0.2, 'Sculpting the shape · step 3 of 12'], p_3: 'resting', p_4: 'resting', p_5: 'resting' },
    { part: 'lr', position: 2 });
  assert.deepEqual([line.kind, line.text, line.stage], ['busy', 'Making 3D models · Sculpting the shape · 2 of 5 · Lantern', 'mesh']);
  assert.ok(Math.abs(line.fraction - (0.2 + 4 * 0.29) / 5) < 1e-9, 'the bar: how far the whole job is');
  // The work goes to the next prop, then to the next step: other words, the same clock.
  const next = at({ p_1: 'resting', p_2: 'resting', p_3: ['working', 0.2], p_4: 'resting', p_5: 'resting' }, { part: 'lr', position: 3 });
  assert.equal(next.text, 'Making 3D models · Sculpting the shape · 3 of 5 · Chest');
  const later = at({ p_1: ['working', 0.4], p_2: 'resting', p_3: 'resting', p_4: 'failed', p_5: 'resting' }, { part: 'hr', position: 1, of: 4 });
  assert.equal(later.text, 'Making 3D models · Refining shape details · 1 of 4 · Barrel', 'a failed prop is not in the next step');
  assert.deepEqual([next.clock, later.clock], [line.clock, line.clock]);
  // Between two props, and while the next model is read from disk.
  assert.equal(at({ p_1: 'resting', p_2: 'resting' }, { part: 'lr', position: 1 }).text, 'Making 3D models · Sculpting the shape · 1 of 2');
  assert.equal(at({ p_1: 'resting', p_2: 'resting' }, { part: 'lr', position: 1, loading: 'shape model' }).text,
    'Making 3D models · Sculpting the shape · 1 of 2 · Loading the shape model…', 'what loads is said as well');
  assert.equal(at({ p_1: 'waiting', p_2: 'waiting' }, { loading: 'background remover' }).text,
    'Making 3D models · 1 of 2 · Loading the background remover…', 'before the first step');
  // One prop: nothing to count.
  assert.equal(at({ p_3: ['working', 0.6] }, { part: 'tex' }).text, 'Making 3D models · Painting the texture · Chest');
  // Stop was pressed: short, so the words the artist waits for are not the ones the narrow column cuts off.
  const cancelling = 'Cancelling… (the current prop may take a moment to stop)';
  assert.equal(at({ p_1: 'resting', p_2: ['working', 0.4] }, { part: 'hr', position: 2, stopping: true, progress: { message: cancelling } }).text,
    `Making 3D models · ${cancelling}`);
  // Before the first step, and when the tool had to start again: the job's own words.
  assert.equal(at({ p_1: 'waiting', p_2: 'waiting' }, { progress: { message: 'Starting Pixal3D again…' } }).text,
    'Making 3D models · 1 of 2 · Starting Pixal3D again…');
});

test('the status line leaves a job whose props take turns to the workbench, for every prop of it, and keeps its clock', () => {
  reset({ jobs: [batchJob('j1', { p_1: 'resting', p_2: ['working', 0.2], p_3: 'waiting', p_4: 'done', p_5: 'failed' }, { part: 'lr', position: 2 })] });
  const shows = (stage, id) => headline(new Set(), { stage, prop: id })?.kind ?? null;
  // The workbench says where each prop of such a job stands - on the tool, waiting for its next step, waiting
  // for its turn, finished, failed - so the line says nothing of it there.
  for (const id of ['p_1', 'p_2', 'p_3', 'p_4', 'p_5']) assert.equal(shows('mesh', id), null, id);
  assert.equal(shows('retopo', 'p_2'), 'busy', 'another stage on screen');
  assert.equal(shows('mesh', null), 'busy');
  // The job's clock goes on while its line is away: back, it does not start at 0:00 again.
  const asked = new Set();
  assert.equal(headline(asked, { stage: 'mesh', prop: 'p_2' }), null);
  const clock = headline().clock;
  assert.deepEqual([...asked], [clock]);
  // Its tool has to start again: the hub has no step for a moment, and the job is still the same job - one
  // clock, and still nothing to say for its props.
  const running = store.jobs[0];
  store.jobs = [{ ...running, part: null, current_prop: null, progress: { ...running.progress, message: 'Starting Pixal3D again…' } }];
  assert.equal(headline().clock, clock, 'the clock does not start at 0:00 again');
  assert.equal(shows('mesh', 'p_1'), null);
  // One status poll that failed: the clock is kept too.
  store.connected = false;
  const offline = new Set();
  assert.equal(headline(offline).kind, 'offline');
  assert.ok(offline.has(clock));
  store.connected = true;
  assert.equal(headline().clock, clock);
  // Before the job runs, for any prop of it: the workbench says its tool loads.
  store.jobs = [{ ...batchJob('j2', { p_3: 'waiting', p_4: 'waiting' }), status: 'starting_tool', current_prop: null }];
  assert.equal(shows('mesh', 'p_3'), null);
  assert.equal(shows('mesh', 'p_4'), null);
  assert.equal(shows('mesh', 'p_1'), 'loading', 'not a prop of the job');
  assert.equal(shows('images', 'p_3'), 'loading');
});

test('a job of another tool reads as before from where its props stand: Retopo\'s tiles, workbench words and status line', () => {
  const job = (done, extra = {}) => hubJob({ id: 'r1', stage: 'retopo', tool: 'lato2', status: 'running', prop_ids: ['p_1', 'p_2', 'p_3'],
    current_prop: `p_${done + 1}`, progress: { done, total: 3, fraction: (done + 0.5) / 3, message: 'Generating the mesh… 50%' }, ...extra });
  const made = [1, 2, 3].map((n) => prop(n, true));
  reset({ jobs: [job(1)], props: made });
  assert.equal(jobFor('retopo', 'p_1'), null);
  assert.deepEqual([jobFor('retopo', 'p_2').phase, jobFor('retopo', 'p_3').phase], ['running', 'waiting']);
  assert.ok(Math.abs(jobFor('retopo', 'p_2').fraction - 0.5) < 1e-9, 'the prop\'s own share of the job');
  assert.deepEqual(RETOPO.status(made[1]), { state: 'processing', message: 'Generating the mesh…', fraction: jobFor('retopo', 'p_2').fraction });
  assert.deepEqual(RETOPO.status(made[2]), { state: 'waiting', message: 'Waiting for its turn' });
  assert.equal(RETOPO.status(made[0]).state, 'idle', 'until the project is read again with its clean mesh');
  const line = headline();
  assert.deepEqual([line.kind, line.text], ['busy', 'Retopologizing with LATO.2 · 2 of 3 · Lantern · Generating the mesh…']);
  // Its clock is the prop's, as before; no prop of such a job ever rests.
  store.jobs = [job(2)];
  const next = headline();
  assert.equal(next.text, 'Retopologizing with LATO.2 · 3 of 3 · Chest · Generating the mesh…');
  assert.notEqual(next.clock, line.clock);
  assert.equal(Object.values(store.jobs[0].props).some((p) => p.state === 'resting'), false);
  // The workbench hides it only for the prop being made, as before.
  assert.equal(headline(new Set(), { stage: 'retopo', prop: 'p_3' }), null);
  assert.equal(headline(new Set(), { stage: 'retopo', prop: 'p_1' }).kind, 'busy');
  store.jobs = [job(1)];
  assert.equal(headline(new Set(), { stage: 'retopo', prop: 'p_3' }).kind, 'busy', 'a prop that waits for its turn');
  // A prop that failed is counted as gone through.
  store.jobs = [job(2, { errors: { p_1: 'No.' } })];
  assert.equal(headline().text, 'Retopologizing with LATO.2 · 3 of 3 · Chest · Generating the mesh…');
});

// ---- results and failures, the moment they happen -------------------------------------------------------

/**
 * Keeps the page's slow timers (a second and more: the next job poll, a toast going away) from running, so a
 * test decides when the next poll happens and leaves nothing behind. Returns {later: the functions kept back,
 * tick(): lets everything that is due now happen, release()}.
 */
function holdSlowTimers() {
  const real = globalThis.setTimeout;
  const later = [];
  globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 1000 ? (later.push(fn), 0) : real(fn, ms, ...rest));
  const tick = async () => { for (let i = 0; i < 8; i++) await new Promise((r) => setImmediate(r)); };
  return { later, tick, release: () => { globalThis.setTimeout = real; } };
}

/**
 * A hub for the job poll: `hub.jobs` and `hub.props` are what it answers with, `hub.next(body)` the job a POST
 * starts. poll() runs the job poll that waits (the page's slow timers are held back: holdSlowTimers). The
 * clock goes on by a second every time it is read, so two stamps of a file's address are never the same.
 */
function fakeHub(project) {
  const hub = { jobs: [], props: [], reads: 0, posted: [] };
  const answer = (data) => ({ status: 200, text: async () => JSON.stringify(data) });
  globalThis.fetch = async (path, init = {}) => {
    if (path.startsWith('/api/jobs?')) return answer(hub.jobs);
    if (path === '/api/jobs' && init.method === 'POST') {
      const body = JSON.parse(init.body);
      hub.posted.push(body);
      hub.jobs = [hub.next(body), ...hub.jobs];
      return answer(hub.jobs[0]);
    }
    if (path === `/api/projects/${project}`) {
      hub.reads += 1;
      return answer({ id: project, name: project, props: structuredClone(hub.props), ui: {} });
    }
    return answer([]);
  };
  const timers = holdSlowTimers();
  const realNow = Date.now;
  let now = realNow();
  Date.now = () => (now += 1000);
  hub.tick = timers.tick;
  hub.poll = async () => {
    const i = timers.later.findLastIndex((fn) => fn.name === 'pollJobs');
    assert.ok(i >= 0, 'a job poll waits');
    await timers.later.splice(i, 1)[0]();
    await timers.tick();
  };
  hub.close = () => {
    timers.release();
    Date.now = realNow;
  };
  return hub;
}

const said = () => toasts.children.map((t) => t.byClass('toast-text')[0].textContent);

test('a prop that finishes shows its model at once, one that fails says why at once; the job\'s end adds only its toast', async () => {
  const hub = fakeHub('Batch');
  try {
    toasts.replaceChildren();
    hub.props = [prop(1), prop(2), prop(3)];
    hub.next = () => ({ ...batchJob('b1', { p_1: 'waiting', p_2: 'waiting', p_3: 'waiting' }), project: 'Batch', status: 'queued', current_prop: null });
    await openProject('Batch');
    await hub.tick();
    await startJob('mesh', ['p_1', 'p_2', 'p_3']);
    await hub.tick();
    const reads = hub.reads;
    const stamp = store.openStamp;
    const step = async (states, opts, extra = {}) => {
      hub.jobs = [{ ...batchJob('b1', states, opts), project: 'Batch', ...extra }];
      await hub.poll();
    };

    await step({ p_1: ['working', 0.1], p_2: 'resting', p_3: 'resting' }, { part: 'ss' });
    await step({ p_1: 'resting', p_2: 'resting', p_3: ['working', 0.7] }, { part: 'tex', position: 3 });
    assert.equal(hub.reads, reads, 'steps alone read nothing again');
    assert.deepEqual(store.jobErrors, {});

    // p_2 fails in its last step: its tile says why at once, while the job goes on. No toast yet.
    const errors = { p_2: 'Not enough graphics memory for this prop.' };
    await step({ p_1: ['working', 0.8], p_2: 'failed', p_3: 'resting' }, { part: 'finish', errors });
    assert.deepEqual(store.jobErrors, { 'mesh:p_2': errors.p_2 });
    assert.deepEqual(jobState('mesh', store.project.props[1]), { state: 'error', message: errors.p_2 });
    assert.equal(jobFor('mesh', 'p_1').phase, 'running');
    assert.deepEqual(said(), []);
    assert.equal(hub.reads, reads, 'a failure changes nothing in the project');

    // p_1 is finished: the project is read again and only its model gets a new address.
    hub.props = [prop(1, true), prop(2), prop(3)];
    await step({ p_1: 'done', p_2: 'failed', p_3: ['working', 0.8] }, { part: 'finish', position: 2, errors });
    assert.equal(hub.reads, reads + 1);
    assert.equal(store.project.props[0].mesh.glb, 'mesh/mesh.glb');
    const shown = bustFor('mesh', 'p_1');
    assert.notEqual(shown, stamp);
    assert.deepEqual([bustFor('mesh', 'p_2'), bustFor('mesh', 'p_3')], [stamp, stamp]);
    assert.equal(MESH.status(store.project.props[0]).state, 'done');
    assert.deepEqual(said(), []);
    await step({ p_1: 'done', p_2: 'failed', p_3: ['working', 0.9] }, { part: 'finish', position: 2, errors });
    assert.equal(hub.reads, reads + 1, 'told once');

    // The job ends: the last model shows, the toast says what failed - and the model that showed before is
    // not loaded a second time (its address stays).
    hub.props = [prop(1, true), prop(2), prop(3, true)];
    await step({ p_1: 'done', p_2: 'failed', p_3: 'done' }, { part: null, errors },
      { status: 'error', current_prop: null, part: null, progress: { done: 3, total: 3, fraction: 1, message: '1 of 3 props failed. Each prop shows what went wrong.' } });
    assert.equal(hub.reads, reads + 2);
    assert.equal(bustFor('mesh', 'p_1'), shown);
    assert.notEqual(bustFor('mesh', 'p_3'), stamp);
    assert.deepEqual(said(), [`Lantern: ${errors.p_2}`]);
    assert.deepEqual(store.jobErrors, { 'mesh:p_2': errors.p_2 });
  } finally {
    hub.close();
  }
});

test('a prop that failed and is made again while its first job goes on: that job\'s end does not bring the old failure back', async () => {
  const hub = fakeHub('Again');
  try {
    toasts.replaceChildren();
    hub.props = [prop(1), prop(2)];
    const first = (states, extra = {}) => ({ ...batchJob('a1', states, { part: 'hr', errors: { p_1: 'It failed.' } }), project: 'Again', ...extra });
    hub.next = () => first({ p_1: 'waiting', p_2: 'waiting' }, { status: 'queued', current_prop: null, errors: {} });
    await openProject('Again');
    await hub.tick();
    await startJob('mesh', ['p_1', 'p_2']);
    await hub.tick();
    hub.jobs = [first({ p_1: 'failed', p_2: ['working', 0.4] })];
    await hub.poll();
    assert.deepEqual(store.jobErrors, { 'mesh:p_1': 'It failed.' });

    // The artist presses "Make 3D model" on it again: a second job waits behind the first.
    const second = (states, extra = {}) => ({ ...batchJob('a2', states), project: 'Again', ...extra });
    hub.next = () => second({ p_1: 'waiting' }, { status: 'queued', current_prop: null });
    await startJob('mesh', ['p_1']);
    await hub.tick();
    assert.deepEqual(store.jobErrors, {});
    assert.equal(jobFor('mesh', 'p_1').phase, 'queued');

    // The first job ends: its toast says what failed in it, but the prop's tile is the second job's now.
    hub.props = [prop(1), prop(2, true)];
    hub.jobs = [second({ p_1: 'waiting' }, { status: 'queued', current_prop: null }),
      first({ p_1: 'failed', p_2: 'done' }, { status: 'error', current_prop: null })];
    await hub.poll();
    assert.deepEqual(said(), ['Barrel: It failed.']);
    assert.deepEqual(store.jobErrors, {});

    // The second job makes it: no failure is left on the prop.
    hub.props = [prop(1, true), prop(2, true)];
    hub.jobs = [second({ p_1: 'done' }, { status: 'done', current_prop: null }), hub.jobs[1]];
    await hub.poll();
    assert.deepEqual(store.jobErrors, {});
    assert.equal(MESH.status(store.project.props[0]).state, 'done');
  } finally {
    hub.close();
  }
});

test('a job started in another window: the props it has already finished show their results at once', async () => {
  const hub = fakeHub('Else');
  try {
    toasts.replaceChildren();
    hub.props = [prop(1), prop(2)];
    await openProject('Else');
    await hub.tick();
    const stamp = store.openStamp;
    // The page sees the job for the first time here, with one prop through: its model shows now, not when
    // the whole batch ends.
    hub.props = [prop(1, true), prop(2)];
    hub.jobs = [{ ...batchJob('e1', { p_1: 'done', p_2: ['working', 0.4] }, { part: 'hr', position: 2 }), project: 'Else' }];
    await hub.poll();
    assert.equal(hub.reads, 2, 'the project is read again, once');
    assert.notEqual(bustFor('mesh', 'p_1'), stamp);
    assert.equal(MESH.status(store.project.props[0]).state, 'done');
    assert.deepEqual(said(), [], 'the job goes on: no toast yet');
  } finally {
    hub.close();
  }
});

test('a newer job of the same stage that has ended does not swallow an older job’s failure of the same prop', async () => {
  const hub = fakeHub('Both');
  try {
    toasts.replaceChildren();
    hub.props = [prop(1), prop(2)];
    const older = (states, extra = {}) => ({ ...batchJob('b1', states, { part: 'hr', errors: { p_1: 'It failed.' } }), project: 'Both', ...extra });
    hub.next = () => older({ p_1: 'waiting', p_2: 'waiting' }, { status: 'queued', current_prop: null, errors: {} });
    await openProject('Both');
    await hub.tick();
    await startJob('mesh', ['p_1', 'p_2']);
    await hub.tick();
    // The artist asked for that prop again in a second job and stopped it: it is over, and has no word on
    // the prop any more - the job still running does.
    const stopped = { ...batchJob('b2', { p_1: 'cancelled' }), project: 'Both', status: 'cancelled', current_prop: null, part: null };
    hub.jobs = [stopped, older({ p_1: 'failed', p_2: ['working', 0.4] })];
    await hub.poll();
    assert.deepEqual(store.jobErrors, { 'mesh:p_1': 'It failed.' });
    assert.deepEqual(jobState('mesh', store.project.props[0]), { state: 'error', message: 'It failed.' });
  } finally {
    hub.close();
  }
});

test('a page opened while a job runs: a prop that failed says so again, and nothing is read twice', async () => {
  const hub = fakeHub('Open');
  try {
    toasts.replaceChildren();
    hub.props = [prop(1, true), prop(2), prop(3)];
    hub.jobs = [{ ...batchJob('o1', { p_1: 'done', p_2: 'failed', p_3: ['working', 0.4] }, { part: 'hr', errors: { p_2: 'It failed.' } }), project: 'Open' }];
    await openProject('Open');
    await hub.tick();
    assert.deepEqual(store.jobErrors, { 'mesh:p_2': 'It failed.' });
    assert.equal(hub.reads, 1, 'the project just read has the finished model already');
    assert.equal(bustFor('mesh', 'p_1'), store.openStamp);
    assert.deepEqual(said(), []);
  } finally {
    hub.close();
  }
});

test('jobs gone with a restarted hub: the props they were not through with say so, the finished and the failed keep what they had', async () => {
  const hub = fakeHub('Gone');
  try {
    toasts.replaceChildren();
    hub.props = [prop(1, true), prop(2), prop(3), prop(4), prop(5)];
    hub.jobs = [{ ...batchJob('g1', { p_1: 'done', p_2: 'failed', p_3: 'resting', p_4: ['working', 0.4], p_5: 'waiting' },
      { part: 'hr', position: 2, errors: { p_2: 'It failed.' } }), project: 'Gone' }];
    await openProject('Gone');
    await hub.tick();
    hub.jobs = [];
    await hub.poll();
    const note = 'Spellcast3D was closed while this was being made, so it was not finished. Press the button again to make it.';
    assert.deepEqual(store.jobErrors, { 'mesh:p_2': 'It failed.', 'mesh:p_3': note, 'mesh:p_4': note, 'mesh:p_5': note });
    assert.deepEqual(said(), ['Spellcast3D was restarted, so 3 props were not finished. Their tiles say which; run them again.']);
  } finally {
    hub.close();
  }
});

// ---- the "faster" switches (stages/faster.js) -------------------------------------------------------------

test('while faster image generation is on, the progress of a picture says so first, in Images and in Omni-Texture; never with it off or for another tool', () => {
  const said = (stage, tool, on) => {
    store.settings = { qwen: { int8_matmul: on } };
    reset({ jobs: [batchJob('j1', { p_1: ['working', 0.4, 'Drawing · step 3 of 8'] }, { stage, tool })] });
    const layer = busyLayer(stage);
    layer.sync(store.project.props[0]);
    return layer.children.filter((line) => /^sv-busy-/.test(line.className) && !line.hidden).map((line) => line.textContent);
  };
  try {
    assert.deepEqual(said('images', 'qwen', true), ['Faster image generation is on', 'Drawing · step 3 of 8']);
    assert.deepEqual(said('omnitexture', 'qwen', true), ['Faster image generation is on', 'Drawing · step 3 of 8']);
    assert.deepEqual(said('images', 'qwen', false), ['Drawing · step 3 of 8']);
    assert.equal(said('mesh', 'pixal3d', true).length, 1, 'the 3D generator has a switch of its own');
  } finally {
    store.settings = undefined;
  }
});

test('Faster image generation changes only after a window that says it holds for all pictures and Omni-Texture, then restarts a running Qwen-Image', async () => {
  const sent = [];
  let saved = false;
  globalThis.fetch = async (path, init = {}) => {
    if (init.method === 'PATCH') saved = JSON.parse(init.body).qwen.int8_matmul;
    if (init.method !== 'GET') sent.push(init.body ? [path, JSON.parse(init.body)] : [path]);
    return { status: 200, text: async () => JSON.stringify(path === '/api/settings' ? { qwen: { int8_matmul: saved } } : { state: 'stopped' }) };
  };
  const timers = holdSlowTimers();
  const quick = globalThis.setTimeout;        // a restart asks for the status in 700 ms: that poll never starts here
  globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 500 ? 0 : quick(fn, ms, ...rest));
  const asking = () => document.body.children.find((d) => d.open && /faster image generation\?/.test(d.textContent));
  const press = (label) => asking().querySelectorAll('.modal-actions .btn').find((b) => b.textContent === label).click();
  const flip = async (sw, to) => {
    sw.input.checked = to;
    sw.input.fire('change');
    await timers.tick();
  };
  try {
    store.settings = undefined;
    reset({ states: { qwen: 'ready' } });
    document.body.replaceChildren();
    const sw = fasterSwitch('images');
    await timers.tick();
    assert.equal(!!sw.input.checked, false, 'off unless the hub\'s settings say otherwise');

    // Nothing changes before the artist agrees; Cancel leaves it as it was.
    await flip(sw, true);
    assert.equal(sw.input.checked, false);
    assert.match(asking().textContent, /all image generation: every picture in Images and every repaint in Omni-Texture/);
    assert.match(asking().textContent, /The image generator restarts now \(about a minute\)\./, 'it picks its kernels when it starts');
    // Three short paragraphs, not one block: what it changes, what it costs, the restart. What it costs says in bold
    // that a prompt's instructions may go unheeded (a sheet of four views came out as two with it on).
    const paragraphs = asking().all((e) => e.className === 'modal-text');
    assert.deepEqual(paragraphs.map((p) => p.textContent.slice(0, 13)), ['This changes ', 'Pictures take', 'The image gen']);
    assert.deepEqual(paragraphs[1].all((e) => e.tagName === 'B').map((b) => b.textContent),
      ['Some instructions may be ignored, for example a character sheet of four views.']);
    press('Cancel');
    await timers.tick();
    assert.deepEqual(sent, []);
    assert.equal(sw.input.checked, false);

    // Agreed: saved as the hub's qwen.int8_matmul, and the running image generator started again.
    await flip(sw, true);
    press('Turn on');
    await timers.tick();
    assert.deepEqual(sent, [['/api/settings', { qwen: { int8_matmul: true } }], ['/api/tools/qwen/stop'], ['/api/tools/qwen/start']]);
    assert.equal(sw.input.checked, true);
    assert.equal(store.settings.qwen.int8_matmul, true);

    // Off again with the image generator stopped: the same question, no restart.
    sent.length = 0;
    reset({ states: { qwen: 'stopped' } });
    await flip(sw, false);
    assert.match(asking().textContent, /exactly as the model draws them again, and take longer\./);
    // One paragraph of two lines: what it changes, then - after a line break - what that does.
    const [off, ...rest] = asking().all((e) => e.className === 'modal-text');
    assert.deepEqual([rest.length, off.all((e) => e.tagName === 'BR').length], [0, 1]);
    assert.match(off.textContent, /for every prop and project\.The pictures come out/);
    assert.doesNotMatch(asking().textContent, /restarts now/, 'it is not running: no restart to announce');
    press('Turn off');
    await timers.tick();
    assert.deepEqual(sent, [['/api/settings', { qwen: { int8_matmul: false } }]]);
    assert.equal(sw.input.checked, false);

    // While it draws, the switch waits: no window, nothing saved.
    sent.length = 0;
    reset({ states: { qwen: 'busy' } });
    await flip(sw, true);
    assert.equal(asking(), undefined);
    assert.equal(sw.input.checked, false);
    assert.deepEqual(sent, []);
    assert.deepEqual(said(), ['The image generator is drawing right now. Change this when it is done.']);
  } finally {
    timers.release();
    store.settings = undefined;
  }
});

test('Faster 3D generation asks the same way: all 3D models, a seed no longer the original\'s; saved as coarse_exact false, then Pixal3D restarts', async () => {
  const sent = [];
  globalThis.fetch = async (path, init = {}) => {
    if (init.method !== 'GET') sent.push(init.body ? [path, JSON.parse(init.body)] : [path]);
    const body = init.method === 'PATCH' ? { pixal3d: JSON.parse(init.body).pixal3d } : { state: 'stopped' };
    return { status: 200, text: async () => JSON.stringify(body) };
  };
  const timers = holdSlowTimers();
  const quick = globalThis.setTimeout;
  globalThis.setTimeout = (fn, ms, ...rest) => (ms >= 500 ? 0 : quick(fn, ms, ...rest));
  const asking = () => document.body.children.find((d) => d.open && /faster 3D generation\?/.test(d.textContent));
  const press = (label) => asking().querySelectorAll('.modal-actions .btn').find((b) => b.textContent === label).click();
  try {
    reset({ states: { pixal3d: 'ready' } });
    store.status.gpu = { compute_cap: 8.6 };
    store.settings = { pixal3d: { coarse_exact: true } };
    document.body.replaceChildren();
    const sw = fasterSwitch('mesh');
    assert.equal(sw.input.checked, false, 'off: the coarse stages exact, the default');
    assert.equal(sw.applies(), true, 'RTX 30xx and newer');

    sw.input.checked = true;
    sw.input.fire('change');
    await timers.tick();
    assert.equal(sw.input.checked, false, 'nothing before the artist agrees');
    assert.match(asking().textContent, /all 3D generation: every 3D model made in the 3D stage, for every prop and project/);
    assert.match(asking().textContent, /a seed no longer gives the model the original Pixal3D makes/);
    assert.match(asking().textContent, /Pixal3D restarts now \(about half a minute\)\./);
    press('Turn on');
    await timers.tick();
    assert.deepEqual(sent, [['/api/settings', { pixal3d: { coarse_exact: false } }], ['/api/tools/pixal3d/stop'], ['/api/tools/pixal3d/start']]);
    assert.equal(sw.input.checked, true);

    // While Pixal3D works, the switch waits.
    sent.length = 0;
    reset({ states: { pixal3d: 'busy' } });
    sw.input.checked = false;
    sw.input.fire('change');
    await timers.tick();
    assert.equal(asking(), undefined);
    assert.equal(sw.input.checked, true);
    assert.deepEqual(sent, []);
    assert.deepEqual(said(), ['Pixal3D is making 3D models right now. Change this when it is done.']);

    // Below RTX 30xx it changes nothing: the panel does not show it, and it asks the hub for nothing.
    store.settings = undefined;
    store.status.gpu = { compute_cap: 7.5 };
    const old = fasterSwitch('mesh');
    assert.equal(old.applies(), false);
    assert.deepEqual(sent, []);
  } finally {
    timers.release();
    store.settings = undefined;
  }
});

// ---- words and settings ---------------------------------------------------------------------------------

test('the 3D generator starts in about half a minute: it reads its models when a job needs them', () => {
  assert.equal(FIRST_START.pixal3d, 'about half a minute');
  assert.equal(TOOL_ETA.pixal3d, 'about half a minute');
  reset({ states: { pixal3d: 'starting' } });
  assert.equal(headline().text, 'Loading the 3D generator… (about half a minute)');
});

test('Settings has no low-video-memory switch for the 3D generator, and saving never writes one', async () => {
  const sent = [];
  globalThis.fetch = async (path, init = {}) => {
    if (init.method === 'PATCH') sent.push(JSON.parse(init.body));
    return { status: 200, text: async () => JSON.stringify({ gpu_exclusive: sent.length ? false : true, forge: { ckpt_dirs: [] }, pixal3d: { low_vram: true, wave_size: 0 } }) };
  };
  const timers = holdSlowTimers();
  try {
    reset();
    document.body.replaceChildren();
    await openSettings();
    const [dialog] = document.body.children;
    assert.ok(dialog.open);
    const labels = dialog.byClass('field-label').map((l) => l.textContent);
    // Advanced's own, then the folded Omni-Texture test's (shell/omnilab.js).
    assert.deepEqual(labels, ['Animations', '3D model', 'Its pictures', 'Qwen-Image’s second picture',
      'Its other pictures sent to Qwen-Image', 'Cameras', 'The views are painted', 'Qwen-Image is shown',
      'Words for Qwen-Image'],
      'no "Low video memory mode"');
    assert.doesNotMatch(dialog.textContent, /low video memory|low-vram/i);
    // Another setting changed and saved: only that one is written (the low_vram an older settings file has stays).
    const [share] = dialog.all((e) => e.tagName === 'INPUT' && e.attrs.type === 'checkbox');
    share.checked = false;
    share.fire('change');
    dialog.querySelectorAll('.modal-actions .btn').at(-1).click();
    await timers.tick();
    assert.deepEqual(sent, [{ gpu_exclusive: false }]);
    assert.equal(dialog.open, false);
    assert.deepEqual(said(), ['Settings saved.']);
  } finally {
    timers.release();
  }
});

test('Settings\' "Faster 3D view" is kept, and the page loaded again, only once the save went through', async () => {
  const { plainEdges, setPlainEdges } = await load('shell/graphics.js');
  const kept = new Map();
  const had = { storage: Object.getOwnPropertyDescriptor(globalThis, 'localStorage'), location: globalThis.location };
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: { getItem: (k) => kept.get(k) ?? null, setItem: (k, v) => kept.set(k, String(v)), removeItem: (k) => kept.delete(k) },
  });
  let reloads = 0;
  globalThis.location = { ...had.location, reload: () => { reloads += 1; } };
  let refuse = true;
  globalThis.fetch = async (path, init = {}) => (init.method === 'PATCH' && refuse
    ? { status: 500, text: async () => JSON.stringify({ detail: 'The settings could not be written.' }) }
    : { status: 200, text: async () => JSON.stringify({ gpu_exclusive: true, forge: { ckpt_dirs: [] } }) });
  const timers = holdSlowTimers();
  try {
    reset();
    document.body.replaceChildren();
    await openSettings();
    const [dialog] = document.body.children;
    const save = () => dialog.querySelectorAll('.modal-actions .btn').at(-1).click();
    const [faster] = dialog.all((e) => e.classList.contains('switch') && e.textContent.includes('Faster 3D view'));
    assert.ok(!faster.input.checked, 'off unless chosen');
    faster.input.checked = true;
    faster.input.fire('change');
    const [share] = dialog.all((e) => e.tagName === 'INPUT' && e.attrs.type === 'checkbox');
    share.checked = false;
    share.fire('change');
    save();
    await timers.tick();
    assert.deepEqual([dialog.open, plainEdges(), kept.size, reloads], [true, false, 0, 0], 'the hub refused: nothing kept yet');
    refuse = false;
    save();
    await timers.tick();
    assert.deepEqual([dialog.open, plainEdges(), kept.get('spellcast3d.plainEdges'), reloads], [false, true, 'on', 1]);
  } finally {
    timers.release();
    setPlainEdges(false);
    globalThis.location = had.location;
    if (had.storage) Object.defineProperty(globalThis, 'localStorage', had.storage);
    else delete globalThis.localStorage;
  }
});

test('the steps the fixtures name are the launcher\'s: each prop\'s fraction rests at the end of its last step', () => {
  const ends = Object.values(PARTS).map(([, end]) => end);
  assert.deepEqual(ends, [...ends].sort((a, b) => a - b));
  assert.deepEqual(Object.keys(PARTS), ['cutout', 'camera', 'features', 'ss', 'cond_lr', 'lr', 'upsample', 'cond_hr', 'hr', 'tex', 'finish']);
});
