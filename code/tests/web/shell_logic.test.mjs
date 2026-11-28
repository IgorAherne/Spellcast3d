// Logic of the web UI shell that needs no browser: the props list's pictures, decks and status marks, the
// stages before and after, the status line's wording, per-prop job state and the small formatters.
//   node --test code/tests/web/shell_logic.test.mjs
// (code/tests/test_web.py runs every code/tests/web/*.test.mjs when Node.js is installed)
import assert from 'node:assert/strict';
import { test } from 'node:test';

// What a browser page has, as far as these modules touch it when they load (actions.js listens for the
// page being hidden or closed).
globalThis.addEventListener ??= () => {};
globalThis.document ??= { addEventListener: () => {}, hidden: false };

const STATIC = new URL('../../spellcast/static/js/', import.meta.url);
const load = (path) => import(new URL(path, STATIC));

const { store, jobFor, stepTime, STAGES } = await load('store.js');
const { hubJob } = await import('./hub_job.mjs');
const { badgeOf, deckOf, tileViews } = await load('shell/proplist.js');
const { propPicture, cropBox, fillSide, frameStyle } = await load('cropped.js');
const { VIEWS } = await load('stages/slicer.js');
const tilePicture = (p) => propPicture(p)?.url ?? null;
const { nextStage, prevStage, STAGE_INFO } = await load('shell/stages.js');
const { headline } = await load('shell/statusline.js');
const { formatBytes, formatElapsed, plural, timeAgo } = await load('dom.js');

const prop = (extra = {}) => ({ id: 'p_1', name: 'Barrel', images: { chosen: 'images/0001.png' }, ...extra });
const tools = (states = {}) => Object.fromEntries(['forge', 'pixal3d', 'imesh', 'lato2', 'triflow']
  .map((t) => [t, { state: states[t] || 'stopped', message: '' }]));

function reset({ jobs = [], tools: t = {}, install = null, connected = true } = {}) {
  store.connected = connected;
  store.jobs = jobs.map((job) => hubJob(job));   // as the hub sends them: each says where its props stand
  store.status = { tools: tools(t), install };
  store.project = { id: 'Props', name: 'Props', props: [prop(), prop({ id: 'p_2', name: 'Lantern' })] };
  store.bust = {};
  store.openStamp = 7;
}

test('a tile shows the smallest picture of its prop there is', () => {
  reset();
  assert.equal(tilePicture(prop()), '/files/Props/props/p_1/images/0001.png?v=7', 'the picture, until its thumbnail is made');
  const small = prop({ thumbs: { images: { path: 'thumbs/images.webp', v: 12, of: 'images/0001.png' }, mesh: { path: 'thumbs/mesh.webp', v: 3 } } });
  assert.deepEqual(propPicture(small), { url: '/files/Props/props/p_1/thumbs/images.webp?v=12', crop: null });
  store.bust['images:p_1'] = 99;
  assert.equal(tilePicture(prop()), '/files/Props/props/p_1/images/0001.png?v=99', 'a new picture is never an old one from the cache');
});

test('a part of a picture is laid out in its box by CSS: the picture scaled and moved, the part filling the box', () => {
  assert.deepEqual(cropBox([512, 0, 512, 512], { w: 1024, h: 512 }), { width: 200, height: 100, left: -100, top: -0 });
  assert.deepEqual(cropBox([0, 256, 256, 256], { w: 1024, h: 1024 }), { width: 400, height: 400, left: -0, top: -100 });
  // A square slice in a tile (4:3): its width fills the tile, as object-fit: cover has a square thumbnail.
  assert.equal(fillSide(512, 512, 4 / 3, 'cover'), 'width');
  assert.equal(fillSide(1024, 512, 4 / 3, 'cover'), 'height');
  assert.equal(fillSide(512, 512, 4 / 3, 'contain'), 'height');
  assert.equal(fillSide(300, 600, 1, 'contain'), 'height');
  assert.equal(fillSide(600, 300, 1, 'contain'), 'width');
  // A tall slice inside a square box (the views beside a hovered tile): its frame is as high as the box, of the
  // slice's shape, and cuts the picture - the figures beside the slice do not show in the room left over.
  const frame = frameStyle(300, 600, 1, 'contain');
  assert.ok(frame.includes('overflow:hidden') && frame.includes('aspect-ratio:300 / 600') && frame.endsWith('height:100%'), frame);
});

test('hovering a deck shows its views: front, left, back, right, each cut to its slice and named by its side', () => {
  reset();
  const sheet = 'images/0001.png';
  const views = [
    { view: 'right', src: sheet, rect: [512, 512, 512, 512] }, { view: 'front', src: sheet, rect: [0, 0, 512, 512] },
    { view: 'back', src: 'images/0002.png', rect: null }, { view: 'left', src: sheet, rect: [0, 512, 512, 512] },
  ];
  const shown = tileViews(prop({ images: { chosen: sheet, views } }));
  assert.deepEqual(shown.map((v) => v.view), ['front', 'left', 'back', 'right']);
  assert.deepEqual(shown.map((v) => v.label), ['front', 'left', 'back', 'right'].map((v) => VIEWS[v].short));
  assert.deepEqual([shown[0].url, shown[0].crop], ['/files/Props/props/p_1/images/0001.png?v=7', [0, 0, 512, 512]]);
  assert.deepEqual([shown[2].url, shown[2].crop], ['/files/Props/props/p_1/images/0002.png?v=7', null], 'a whole picture as a view');
  assert.deepEqual(tileViews(prop()), [], 'one picture: nothing to show');
  assert.deepEqual(tileViews(prop({ images: { chosen: sheet, views: views.slice(0, 1) } })), [], 'one view is one picture');
  assert.deepEqual(tileViews(prop({ images: { chosen: 'images/0003.png', views } })), [], 'views that wait aside');
});

test('a prop made from the artist\'s own model shows its model, and an icon until its snapshot is made', () => {
  reset();
  const own = prop({ images: { chosen: null }, mesh: { glb: 'mesh/imported_001.glb', imported: { materials: [] } } });
  assert.equal(tilePicture(own), null);
  const snapped = { ...own, thumbs: { mesh: { path: 'thumbs/mesh.webp', v: 'a' } } };
  assert.equal(tilePicture(snapped), '/files/Props/props/p_1/thumbs/mesh.webp?v=a');
  assert.equal(tilePicture(prop({ images: {} })), null);
});

test('a tile marks the status of its prop in the stage on screen; the check never shows in Images', () => {
  assert.equal(badgeOf('images', 'done'), '', 'every prop in the list has its picture');
  assert.equal(badgeOf('mesh', 'done'), 'done');
  assert.equal(badgeOf('retopo', 'review'), 'review', 'a result to accept is a hollow dot, not a check');
  assert.equal(badgeOf('retopo', 'stale'), 'stale');
  assert.equal(badgeOf('images', 'waiting'), 'waiting');
  assert.equal(badgeOf('projection', 'error'), 'error');
  assert.equal(badgeOf('mesh', 'processing'), '', 'being made: a progress bar, no mark');
  assert.equal(badgeOf('mesh', 'idle'), '');
  assert.equal(badgeOf('mesh', 'unreachable'), '');
  assert.equal(badgeOf('mesh', 'nonsense'), '');
  assert.equal(badgeOf('mesh', undefined), '');
});

test('the two big buttons are the way between the stages: the call to action goes down one, "↑" back up one', () => {
  assert.equal(nextStage('images'), 'mesh');
  assert.equal(nextStage('mesh'), 'omnitexture', 'the 3D stage goes on to Omni-Texture (optional) before Retopo');
  assert.equal(nextStage('omnitexture'), 'retopo');
  assert.equal(nextStage('retopo'), 'projection');
  assert.equal(nextStage('projection'), 'rig', '"Export all" goes down to the last stage');
  assert.equal(nextStage('rig'), null);
  assert.equal(prevStage('images'), null);
  assert.equal(prevStage('mesh'), 'images');
  assert.equal(prevStage('retopo'), 'omnitexture');
  assert.equal(prevStage('rig'), 'projection');
  // every stage is reached going down from the first, and left going up from the last
  const walk = (from, step) => { const seen = []; for (let s = from; s; s = step(s)) seen.push(s); return seen; };
  assert.deepEqual(walk('images', nextStage), STAGES);
  assert.deepEqual(walk('rig', prevStage), [...STAGES].reverse());
  assert.deepEqual(Object.values(STAGE_INFO).map((s) => s.title), ['Images', '3D', 'Omni-Texture', 'Retopo', 'Projection',
    'Export']);
});

test('the props column\'s big heading is what the stage does to every prop, without a number', () => {
  assert.deepEqual(STAGES.map((s) => STAGE_INFO[s].heading),
    ['All Images', 'All 3D', 'Omni-Texture All', 'All Retopo', 'Project All', 'Export All']);
});

test('a prop with several views is drawn as a deck; one picture, or views that wait, as a single tile', () => {
  const views = (...kinds) => kinds.map((view) => ({ view, src: 'images/0001.png', rect: null }));
  assert.equal(deckOf(prop()), 0);
  assert.equal(deckOf(prop({ images: { chosen: 'images/0001.png', views: views('front') } })), 0, 'one view is one picture');
  assert.equal(deckOf(prop({ images: { chosen: 'images/0001.png', views: views('front', 'left', 'back', 'right') } })), 4);
  assert.equal(deckOf(prop({ images: { chosen: 'images/0002.png', views: views('front', 'back') } })), 0,
    'the chosen picture is none of theirs: the model is made from it alone');
  assert.equal(deckOf(prop({ images: { chosen: null }, mesh: { glb: 'mesh/imported_001.glb' } })), 0);
});

test('the status line says nothing while nothing happens', () => {
  reset();
  assert.equal(headline(), null);
});

test('the status line: offline first, then the running job with its prop and progress', () => {
  reset({ connected: false });
  assert.equal(headline().kind, 'offline');
  reset({
    jobs: [{ id: 'j1', stage: 'mesh', status: 'running', prop_ids: ['p_1', 'p_2'], current_prop: 'p_2',
      progress: { done: 1, total: 2, fraction: 0.6, message: 'Sampling 12/25' } }],
    tools: { forge: 'error' },
  });
  const line = headline();
  assert.equal(line.kind, 'busy');
  assert.equal(line.text, 'Making 3D models · 2 of 2 · Lantern · Sampling 12/25');
  reset({ jobs: [{ id: 'j2', stage: 'projection', status: 'running', prop_ids: ['p_1'], current_prop: 'p_1',
    progress: { done: 0, total: 1, fraction: 0.42, message: 'Baking the textures… 42%' } }] });
  assert.equal(headline().text, 'Baking textures · Barrel', 'the verb and the percentage are not said twice');
  assert.equal(line.fraction, 0.6);
  assert.equal(line.stage, 'mesh');
  assert.ok(line.clock);
});

test('the status line: a job waiting for its tool says loading, or installing', () => {
  reset({ jobs: [{ id: 'j1', stage: 'images', status: 'starting_tool', prop_ids: ['p_1'], progress: {} }], tools: { forge: 'starting' } });
  assert.equal(headline().kind, 'loading');
  assert.match(headline().text, /^Loading Forge Neo…/, 'the image generator by its name: there are two');
  reset({
    jobs: [{ id: 'j1', stage: 'mesh', status: 'starting_tool', prop_ids: ['p_1'],
      progress: { message: 'Waiting for the 3D generator to finish installing (45%)' } }],
    tools: { pixal3d: 'installing' },
  });
  assert.deepEqual([headline().kind, headline().text], ['install', 'Waiting for the 3D generator to finish installing (45%)']);
  // Jobs the hub holds in the queue until the installer is done say why; waiting for other jobs is no news.
  const wait = 'Spellcast3D is being installed in another window; waiting for that to finish…';
  reset({ jobs: [{ id: 'j1', stage: 'mesh', status: 'queued', prop_ids: ['p_1'], progress: { message: wait } }] });
  assert.deepEqual([headline().kind, headline().text, headline().stage], ['install', wait, 'mesh']);
  reset({ jobs: [{ id: 'j1', stage: 'mesh', status: 'queued', prop_ids: ['p_1'], progress: { message: 'Waiting for the jobs before it…' } }] });
  assert.equal(headline(), null);
});

test('the status line: a tool error with details, then install trouble and progress', () => {
  reset({ tools: { pixal3d: 'error' } });
  assert.deepEqual([headline().kind, headline().tool], ['error', 'pixal3d']);
  const comp = (status, extra = {}) => ({ status, step: '', fraction: null, bytes_done: null, bytes_total: null, message: '', ...extra });
  reset({ install: { running: false, current: null, components: { forge: comp('failed', { message: 'No internet.' }) } } });
  assert.equal(headline().kind, 'error');
  assert.equal(headline().text, 'Forge Neo could not be installed: No internet.');
  reset({
    install: {
      running: true, current: 'pixal3d',
      components: { pixal3d: comp('running', { fraction: 0.25, bytes_done: 1024 ** 3, bytes_total: 4 * 1024 ** 3 }) },
    },
  });
  // short (the column is narrow): with the download's size the bar says how far it is
  assert.equal(headline().text, 'Installing the 3D generator · 1.0 GB of 4.0 GB');
  // A step with no number moving (pip unpacking for minutes at one percentage) never looks stuck: the line has the
  // step's clock, which the status line ticks, and what the step does now on hover. A new step starts it anew; what
  // the step does right now (after " · ") does not.
  const step = (text) => reset({ install: { running: true, current: 'lato2',
    components: { lato2: comp('running', { fraction: 0, step: text }) } } });
  step('Installing its Python packages');
  const seen = new Set();
  const line = headline(seen);
  assert.deepEqual([line.text, line.clock, line.detail], ['Installing LATO.2… 0%', 'install:lato2:Installing its Python packages',
    'Installing its Python packages']);
  assert.ok(seen.has(line.clock), 'its time is kept while the line shows');
  step('Installing its Python packages · torch');
  assert.equal(headline().clock, line.clock);
  step('Downloading its AI models');
  assert.notEqual(headline().clock, line.clock);
  // The install card's and the workbench's line of a tool: how long its step has gone on, after a quarter of a minute.
  const c = { status: 'running', step: 'Installing its Python packages · torch', fraction: 0 };
  assert.deepEqual([stepTime('x1', c, 1000), stepTime('x1', c, 10_000), stepTime('x1', c, 135_000)], ['', '', '2:14 on this step']);
  assert.equal(stepTime('x1', { ...c, step: 'Installing its Python packages · numpy' }, 136_000), '2:15 on this step');
  assert.equal(stepTime('x1', { ...c, step: 'Downloading its AI models' }, 137_000), '', 'another step: from the start');
  // A download that is through is no news: "583 KB of 583 KB" standing for minutes read as stuck.
  reset({ install: { running: true, current: 'triflow',
    components: { triflow: comp('running', { fraction: 0.3, step: 'Installing its Python packages', bytes_done: 583_000, bytes_total: 583_000 }) } } });
  assert.equal(headline().text, 'Installing TriFlow… 30%');
});

test('the status line leaves out what the workbench shows: the prop it shows being made, the tool of its stage loading', () => {
  const job = { id: 'j1', stage: 'mesh', status: 'running', prop_ids: ['p_1', 'p_2'], current_prop: 'p_2',
    progress: { done: 1, total: 2, fraction: 0.6 } };
  reset({ jobs: [job] });
  assert.equal(headline(new Set(), { stage: 'mesh', prop: 'p_2' }), null);
  assert.equal(headline(new Set(), { stage: 'mesh', prop: 'p_1' }).kind, 'busy');
  assert.equal(headline(new Set(), { stage: 'retopo', prop: 'p_2' }).kind, 'busy');
  reset({ jobs: [{ ...job, status: 'starting_tool', current_prop: null }] });
  assert.equal(headline(new Set(), { stage: 'mesh', prop: 'p_1' }), null);
  reset({ tools: { forge: 'starting' } });
  assert.equal(headline(new Set(), { stage: 'images', prop: 'p_1' }), null);
  assert.equal(headline(new Set(), { stage: 'mesh', prop: 'p_1' }).kind, 'loading');
  // Retopo: the tool of the prop on screen is its method (the workbench says it loads), not Instant Meshes
  reset({ tools: { triflow: 'starting' } });
  store.project.props[0].retopo = { settings: { method: 'triflow' } };
  store.project.props[1].retopo = { settings: { method: 'imesh' } };
  assert.equal(headline(new Set(), { stage: 'retopo', prop: 'p_1' }), null);
  assert.equal(headline(new Set(), { stage: 'retopo', prop: 'p_2' }).text, 'Loading TriFlow… (this can take a minute)');
});

test('an Omni-Texture job says what it does, and loads Qwen-Image 2.1 by its name', () => {
  reset({ jobs: [{ id: 'j1', stage: 'omnitexture', tool: 'qwen', status: 'running', prop_ids: ['p_1', 'p_2'],
    current_prop: 'p_2', progress: { done: 1, total: 2, fraction: 0.6 } }] });
  assert.equal(headline().text, 'Painting Omni-Textures · 2 of 2 · Lantern');
  reset({ jobs: [{ id: 'j1', stage: 'omnitexture', tool: 'qwen', status: 'starting_tool', prop_ids: ['p_1'], progress: {} }],
    tools: { qwen: 'starting' } });
  assert.equal(headline().text, 'Loading Qwen-Image 2.1… (this can take a minute)');
  assert.equal(jobFor('omnitexture', 'p_1')?.phase, 'starting');
  assert.equal(jobFor('images', 'p_1'), null, 'an Omni-Texture job is no Images job, although both run on Qwen-Image');
});

test('the status line names the retopology tool a job runs on', () => {
  reset({ jobs: [{ id: 'j1', stage: 'retopo', tool: 'lato2', status: 'starting_tool', prop_ids: ['p_1'], progress: {} }],
    tools: { lato2: 'starting' } });
  assert.deepEqual([headline().kind, headline().text], ['loading', 'Loading LATO.2… (this can take a minute)']);
  reset({ jobs: [{ id: 'j1', stage: 'retopo', tool: 'triflow', status: 'running', prop_ids: ['p_1', 'p_2'], current_prop: 'p_2',
    progress: { done: 1, total: 2, fraction: 0.7 } }] });
  assert.equal(headline().text, 'Retopologizing with TriFlow · 2 of 2 · Lantern');
  reset({ jobs: [{ id: 'j1', stage: 'retopo', tool: 'imesh', status: 'running', prop_ids: ['p_1'], current_prop: 'p_1', progress: {} }] });
  assert.equal(headline().text, 'Retopologizing with Instant Meshes · Barrel');
  // A job from before jobs named their tool: the stage's own.
  reset({ jobs: [{ id: 'j1', stage: 'retopo', status: 'starting_tool', prop_ids: ['p_1'], progress: {} }], tools: { imesh: 'starting' } });
  assert.equal(headline().text, 'Loading the retopology tool…');
  reset({ jobs: [{ id: 'j1', stage: 'retopo', tool: 'triflow', status: 'starting_tool', prop_ids: ['p_1'], progress: {} }],
    tools: { triflow: 'installing' } });
  assert.deepEqual([headline().kind, headline().text], ['install', 'Waiting for TriFlow to finish installing']);
  // The retopology tools that load, fail or install by themselves say their names.
  reset({ tools: { lato2: 'starting' } });
  assert.equal(headline().text, 'Loading LATO.2… (this can take a minute)');
  reset({ tools: { triflow: 'error' } });
  assert.deepEqual([headline().kind, headline().tool, headline().text], ['error', 'triflow', 'TriFlow stopped with an error.']);
  reset({ tools: { forge: 'error' } });
  assert.equal(headline().text, 'Forge Neo stopped with an error.');
  const comp = (status, extra = {}) => ({ status, step: '', fraction: null, bytes_done: null, bytes_total: null, message: '', ...extra });
  reset({ install: { running: true, current: 'lato2', components: { lato2: comp('running', { fraction: 0.5 }) } } });
  assert.equal(headline().text, 'Installing LATO.2… 50%');
  reset({ install: { running: false, current: null, components: { triflow: comp('failed', { message: 'No internet.' }) } } });
  assert.equal(headline().text, 'TriFlow could not be installed: No internet.');
  reset({ install: { running: true, current: 'hub', components: { hub: comp('running') } } });
  assert.equal(headline().text, 'Installing the tools…');
});

test('a prop of a running batch knows whether it waits, runs or is done', () => {
  reset({
    jobs: [{ id: 'j1', stage: 'retopo', status: 'running', prop_ids: ['p_1', 'p_2'], current_prop: 'p_2',
      progress: { done: 1, total: 2, fraction: 0.75 } }],
  });
  assert.equal(jobFor('retopo', 'p_1'), null);
  const running = jobFor('retopo', 'p_2');
  assert.equal(running.phase, 'running');
  assert.equal(running.fraction, 0.5);
  assert.equal(jobFor('mesh', 'p_2'), null);
});

test('the formatters speak plainly', () => {
  assert.equal(formatElapsed(72_000), '1:12');
  assert.equal(formatElapsed(3_725_000), '1:02:05');
  assert.equal(formatBytes(1_288_490_188), '1.2 GB');
  assert.equal(formatBytes(5 * 1024 ** 2), '5 MB');
  assert.equal(plural(1, 'prop'), '1 prop');
  assert.equal(plural(3, 'prop'), '3 props');
  assert.equal(timeAgo(new Date().toISOString()), 'just now');
  assert.equal(timeAgo(new Date(Date.now() - 5 * 60000).toISOString()), '5 min ago');
  assert.equal(timeAgo('not a date'), '');
});

test('Ctrl+Z takes back the newest change of all the undo lists registered: each list acts only on its newest', async () => {
  const { isNewest, isUndoKey, undoList, undoStamp } = await load('undo.js');
  const a = [];
  const b = [];
  const dropA = undoList(() => a.at(-1));
  const dropB = undoList(() => b.at(-1));
  a.push(undoStamp());
  b.push(undoStamp());
  assert.deepEqual([isNewest(a.at(-1)), isNewest(b.at(-1))], [false, true]);
  b.pop();
  assert.ok(isNewest(a.at(-1)), 'then the one before');
  a.pop();
  assert.equal(isNewest(a.at(-1)), false, 'nothing to undo');
  b.push(undoStamp());
  dropB();
  a.push(undoStamp() - 10);
  assert.ok(isNewest(a.at(-1)), 'a list gone (its stage left) holds nothing back');
  dropA();
  assert.deepEqual([isUndoKey({ key: 'z', ctrlKey: true }), isUndoKey({ key: 'Z', metaKey: true }),
    isUndoKey({ key: 'z', ctrlKey: true, shiftKey: true }), isUndoKey({ key: 'z' })], [true, true, false, false]);
});
