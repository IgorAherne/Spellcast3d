// The props' small pictures: a picture's thumbnail counts only while it is of what the prop shows now (the
// chosen picture and the part of it), whoever changed the prop - and what thumbs.js draws next follows from that.
//   node --test code/tests/web/thumbs.test.mjs
// (code/tests/test_web.py runs every code/tests/web/*.test.mjs when Node.js is installed)
import assert from 'node:assert/strict';
import { test } from 'node:test';

// What a browser page has, as far as these modules touch it when they load (actions.js listens for the
// page being hidden or closed).
globalThis.addEventListener ??= () => {};
globalThis.document ??= { addEventListener: () => {}, hidden: false };

const STATIC = new URL('../../spellcast/static/js/', import.meta.url);
const load = (path) => import(new URL(path, STATIC));

const { store, emit, getProp, thumbSignature, thumbOf, dropStaleThumb } = await load('store.js');
const { replaceProp } = await load('actions.js');
const { hubJob } = await import('./hub_job.mjs');
const { propPicture } = await load('cropped.js');
const { thumbUrl } = await load('stages/common.js');
const { thumbTask, kept, tryOutcome, noteTry, resting } = await load('stages/thumbs.js');
const tilePicture = (p) => propPicture(p)?.url ?? null;

const SHEET = 'images/0002.png';
const quadrants = (front = [0, 0, 512, 512]) => [
  { view: 'front', src: SHEET, rect: front },
  { view: 'back', src: SHEET, rect: [512, 0, 512, 512] },
];
const thumb = (of, v = 5) => ({ path: 'thumbs/images.webp', v, of });
const prop = (images, thumbs = {}) => ({
  id: 'p_1', name: 'Character', images: { candidates: ['images/0001.png', SHEET], views: [], references: [], ...images }, thumbs,
});

function open(...props) {
  store.jobs = [];
  store.bust = {};
  store.openStamp = 7;
  store.project = { id: 'Props', name: 'Props', props };
}

test('what a picture thumbnail must show: the chosen picture and, of a sliced sheet, its Front slice', () => {
  assert.equal(thumbSignature(undefined), null);
  assert.equal(thumbSignature({ chosen: null, views: quadrants() }), null);
  assert.equal(thumbSignature({ chosen: 'images/0001.png' }), 'images/0001.png');
  assert.equal(thumbSignature({ chosen: SHEET, views: quadrants() }), `${SHEET}#0,0,512,512`);
  assert.equal(thumbSignature({ chosen: SHEET, views: quadrants([16, 0, 512, 512]) }), `${SHEET}#16,0,512,512`);
  // The views wait aside while another picture is chosen; a whole picture as the front view has no part.
  assert.equal(thumbSignature({ chosen: 'images/0001.png', views: quadrants() }), 'images/0001.png');
  assert.equal(thumbSignature({ chosen: SHEET, views: [{ view: 'front', src: SHEET, rect: null }] }), SHEET);
  assert.equal(thumbSignature({ chosen: SHEET, views: [{ view: 'back', src: SHEET, rect: [0, 0, 9, 9] }] }), SHEET);
});

test('a thumbnail of another picture, or of another part of it, is no thumbnail', () => {
  const first = thumb('images/0001.png');
  assert.equal(thumbOf(prop({ chosen: 'images/0001.png' }, { images: first }), 'images'), first);
  // A generation finished (the hub chose its new picture), another picture was picked, the sheet was sliced,
  // the Front slice moved, the slices were cleared: the thumbnail from before does not count.
  assert.equal(thumbOf(prop({ chosen: SHEET }, { images: first }), 'images'), null);
  assert.equal(thumbOf(prop({ chosen: SHEET, views: quadrants() }, { images: thumb(SHEET) }), 'images'), null);
  assert.equal(thumbOf(prop({ chosen: SHEET, views: quadrants([16, 0, 512, 512]) }, { images: thumb(`${SHEET}#0,0,512,512`) }), 'images'), null);
  assert.equal(thumbOf(prop({ chosen: SHEET }, { images: thumb(`${SHEET}#0,0,512,512`) }), 'images'), null);
  assert.ok(thumbOf(prop({ chosen: SHEET, views: quadrants() }, { images: thumb(`${SHEET}#0,0,512,512`) }), 'images'));
  // One stored before thumbnails said what they show cannot be trusted; a prop without one has none.
  assert.equal(thumbOf(prop({ chosen: SHEET }, { images: { path: 'thumbs/images.webp', v: 1 } }), 'images'), null);
  assert.equal(thumbOf(prop({ chosen: SHEET }), 'images'), null);
  assert.equal(thumbOf({ id: 'p_2' }, 'images'), null);
  // A 3D snapshot has no such mark: the hub drops it when its stage gets a new result.
  const snap = { path: 'thumbs/mesh.webp', v: 3 };
  assert.equal(thumbOf(prop({ chosen: SHEET }, { mesh: snap }), 'mesh'), snap);

  const stale = prop({ chosen: SHEET }, { images: first, mesh: snap });
  assert.deepEqual(dropStaleThumb(stale).thumbs, { mesh: snap });
  const current = prop({ chosen: 'images/0001.png' }, { images: first, mesh: snap });
  assert.deepEqual(dropStaleThumb(current).thumbs, { images: first, mesh: snap });
});

test('the tile never shows the thumbnail of a picture the prop no longer shows', () => {
  open(prop({ chosen: 'images/0001.png' }, { images: thumb('images/0001.png') }));
  emit('project', { hard: false });
  assert.equal(tilePicture(getProp('p_1')), '/files/Props/props/p_1/thumbs/images.webp?v=5');

  // The hub's prop after a generation, with a thumbnail of the picture before (stored by an older hub, or by
  // a tab that drew it late): the tile and the headings show the chosen picture itself.
  replaceProp(prop({ chosen: SHEET }, { images: thumb('images/0001.png') }));
  assert.equal(tilePicture(getProp('p_1')), '/files/Props/props/p_1/images/0002.png?v=7');
  assert.equal(thumbUrl(getProp('p_1'), 'images'), null);

  // The page changes a prop itself before the hub has answered (a picture chosen, the sheet sliced): the next
  // thing any view hears, the thumbnail from before is gone.
  replaceProp(prop({ chosen: SHEET }, { images: thumb(SHEET, 6) }));
  assert.equal(tilePicture(getProp('p_1')), '/files/Props/props/p_1/thumbs/images.webp?v=6');
  const shown = getProp('p_1');
  shown.images = { ...shown.images, views: quadrants() };
  emit('prop', shown);
  assert.equal(tilePicture(getProp('p_1')), '/files/Props/props/p_1/images/0002.png?v=7');
  shown.images = { ...shown.images, chosen: 'images/0001.png' };
  shown.thumbs = { images: thumb(`${SHEET}#0,0,512,512`, 8) };
  emit('jobs');
  assert.equal(tilePicture(getProp('p_1')), '/files/Props/props/p_1/images/0001.png?v=7');
});

test('a prop\'s small picture is its chosen picture by construction: asked of the prop alone, whatever thumbnail it carries', () => {
  open();
  const whole = '/files/Props/props/p_1/images/0002.png?v=7';
  const small = '/files/Props/props/p_1/thumbs/images.webp?v=5';
  // Its thumbnail, while that shows the chosen picture (and the part of it).
  assert.deepEqual(propPicture(prop({ chosen: SHEET }, { images: thumb(SHEET) })), { url: small, crop: null });
  assert.deepEqual(propPicture(prop({ chosen: SHEET, views: quadrants() }, { images: thumb(`${SHEET}#0,0,512,512`) })), { url: small, crop: null });
  // A thumbnail of another picture sits on the prop and nothing took it off (no emit ran): the chosen picture.
  const stale = prop({ chosen: SHEET }, { images: thumb('images/0001.png') });
  assert.deepEqual(propPicture(stale), { url: whole, crop: null });
  assert.ok(stale.thumbs.images, 'the prop was not touched');
  assert.equal(thumbUrl(stale, 'images'), null, 'the headings ask the same way');
  // A sheet sliced this instant: its Front slice at once, with the thumbnail of the whole sheet still on it,
  // with none at all, and after the Front slice moved.
  assert.deepEqual(propPicture(prop({ chosen: SHEET, views: quadrants() }, { images: thumb(SHEET) })), { url: whole, crop: [0, 0, 512, 512] });
  assert.deepEqual(propPicture(prop({ chosen: SHEET, views: quadrants() })), { url: whole, crop: [0, 0, 512, 512] });
  assert.deepEqual(propPicture(prop({ chosen: SHEET, views: quadrants([16, 0, 512, 512]) }, { images: thumb(`${SHEET}#0,0,512,512`) })),
    { url: whole, crop: [16, 0, 512, 512] });
  // The slices cleared, or another picture chosen (the views wait aside): all of the chosen picture again.
  assert.deepEqual(propPicture(prop({ chosen: SHEET }, { images: thumb(`${SHEET}#0,0,512,512`) })), { url: whole, crop: null });
  assert.deepEqual(propPicture(prop({ chosen: 'images/0001.png', views: quadrants() }, { images: thumb(`${SHEET}#0,0,512,512`) })),
    { url: '/files/Props/props/p_1/images/0001.png?v=7', crop: null });
  // Views that are whole pictures (one file per side): the front one is the chosen picture, all of it.
  const sides = [{ view: 'front', src: SHEET, rect: null }, { view: 'back', src: 'images/0001.png', rect: null }];
  assert.deepEqual(propPicture(prop({ chosen: SHEET, views: sides })), { url: whole, crop: null });
  assert.deepEqual(propPicture(prop({ chosen: SHEET, views: sides }, { images: thumb(SHEET) })), { url: small, crop: null });
  // No chosen picture: none, also with a thumbnail left behind; the artist's own 3D model: its snapshot.
  assert.equal(propPicture(prop({ chosen: null }, { images: thumb(SHEET) })), null);
  const own = { ...prop({ chosen: null }), mesh: { glb: 'mesh/imported_001.glb', imported: { materials: [] } } };
  assert.equal(propPicture(own), null, 'until its snapshot is made');
  assert.deepEqual(propPicture({ ...own, thumbs: { mesh: { path: 'thumbs/mesh.webp', v: 3 } } }),
    { url: '/files/Props/props/p_1/thumbs/mesh.webp?v=3', crop: null });
  assert.equal(propPicture(null), null);
});

test('a 3D snapshot waits while a model is painted by hand: loading a whole model beside it would stall the stroke', async () => {
  const { createSaver, paintingNow } = await load('stages/paintbar.js');
  open();
  const model = { ...prop({ chosen: SHEET }), mesh: { glb: 'mesh/mesh.glb' } };
  assert.ok(thumbTask('Props', model, 'mesh'), 'drawn while nothing is painted');
  // A painting that waits to be saved (nothing left to save once it is flushed).
  const session = { version: 1, saved: 1, hold: () => () => {} };
  const saver = createSaver({ project: 'Props', propId: 'p_1', glb: 'x.glb', session, save: async () => null });
  saver.schedule();
  assert.ok(paintingNow());
  assert.equal(thumbTask('Props', model, 'mesh'), null, 'not while a painting is on its way into its files');
  assert.ok(thumbTask('Props', model, 'images'), 'a picture\'s thumbnail loads no model: drawn all the same');
  await saver.flush();
  assert.ok(!paintingNow());
  assert.ok(thumbTask('Props', model, 'mesh'), 'drawn once the painting is in its files');
});

test('no thumbnail is given up for good: a refusal for a changed prop does not count, failures rest and are tried again', () => {
  open();
  const whole = thumbTask('Props', prop({ chosen: SHEET }), 'images');
  // The hub's answer to the upload: kept; refused because its prop shows another picture (chosen a moment
  // ago here, or elsewhere) or another part of it; refused although it shows the same (a failure).
  assert.equal(tryOutcome(prop({ chosen: SHEET }, { images: thumb(SHEET) }), whole), 'kept');
  assert.equal(tryOutcome(prop({ chosen: 'images/0001.png' }), whole), 'changed');
  assert.equal(tryOutcome(prop({ chosen: 'images/0001.png' }, { images: thumb('images/0001.png') }), whole), 'changed');
  assert.equal(tryOutcome(prop({ chosen: SHEET, views: quadrants() }), whole), 'changed');
  assert.equal(tryOutcome(prop({ chosen: SHEET }), whole), 'failed');
  const model = { ...prop({ chosen: null }), mesh: { glb: 'mesh/mesh.glb' } };
  assert.equal(tryOutcome(model, thumbTask('Props', model, 'mesh')), 'failed', 'a 3D snapshot the hub did not keep');

  // Clicking back and forth between two pictures, each try refused: neither picture is ever left alone.
  const t0 = 1_000_000;
  for (let i = 0; i < 20; i++) noteTry(whole.result, 'changed', t0 + i);
  assert.equal(resting(whole.result, t0 + 20), false);
  // Failures: a few tries in a row, then a rest, then one more try per rest - never a hot loop, never for good.
  noteTry(whole.result, 'failed', t0);
  noteTry(whole.result, 'failed', t0 + 1000);
  assert.equal(resting(whole.result, t0 + 1001), false);
  noteTry(whole.result, 'failed', t0 + 2000);
  assert.equal(resting(whole.result, t0 + 2001), true);
  assert.equal(resting(whole.result, t0 + 20_000), true);
  assert.equal(resting(whole.result, t0 + 2000 + 30_000), false, 'tried again after the pause');
  noteTry(whole.result, 'failed', t0 + 40_000);
  assert.equal(resting(whole.result, t0 + 40_001), true, 'one try, then it rests again');
  assert.equal(resting('images|another|picture', t0), false, 'another picture has its own count');
  noteTry(whole.result, 'kept', t0 + 80_000);
  noteTry(whole.result, 'failed', t0 + 80_001);
  assert.equal(resting(whole.result, t0 + 80_002), false, 'kept once: the failures before are forgotten');
  noteTry(whole.result, 'kept', t0 + 80_003);
});

test('what is drawn next: the thumbnail the prop lacks now, marked with what it shows', () => {
  open();
  const done = prop({ chosen: 'images/0001.png' }, { images: thumb('images/0001.png') });
  assert.equal(thumbTask('Props', done, 'images'), null, 'its thumbnail shows it');
  assert.equal(thumbTask('Props', prop({ chosen: null }), 'images'), null, 'no picture yet');

  const generated = prop({ chosen: SHEET }, { images: thumb('images/0001.png') });
  const whole = thumbTask('Props', generated, 'images');
  assert.deepEqual([whole.url, whole.crop, whole.of], ['/files/Props/props/p_1/images/0002.png?v=7', null, SHEET]);

  const sliced = thumbTask('Props', prop({ chosen: SHEET, views: quadrants() }, { images: thumb(SHEET) }), 'images');
  assert.deepEqual([sliced.crop, sliced.of], [[0, 0, 512, 512], `${SHEET}#0,0,512,512`]);
  assert.notEqual(sliced.result, whole.result, 'the same picture, another part: another thumbnail');

  // The same picture chosen again later lacks its thumbnail again (the hub dropped it meanwhile).
  assert.equal(thumbTask('Props', prop({ chosen: SHEET }), 'images').result, whole.result);

  // While its pictures are being made, the prop's picture is about to change: drawn once it is there.
  store.jobs = [hubJob({ id: 'j1', stage: 'images', status: 'running', prop_ids: ['p_1'], current_prop: 'p_1', progress: {} })];
  assert.equal(thumbTask('Props', generated, 'images'), null);
  store.jobs = [];

  // A 3D snapshot waits while a job of its stage runs: drawing one loads the whole model into a second 3D
  // context, and the tool needs that video memory. The sweep after the job ends draws it.
  const model = { ...prop({ chosen: null }), mesh: { glb: 'mesh/mesh.glb' } };
  const batch = { id: 'j1', stage: 'mesh', project: 'Props', status: 'running', prop_ids: ['p_2'],
    props: { p_2: { state: 'working', fraction: 0.4, message: '' } }, part: { key: 'hr', label: 'Refining shape details', position: 1, of: 1 } };
  store.jobs = [batch];
  assert.equal(thumbTask('Props', model, 'mesh'), null, 'another prop is on the 3D generator');
  assert.ok(thumbTask('Props', prop({ chosen: SHEET }), 'images'), 'the pictures are drawn as ever');
  store.jobs = [{ ...batch, status: 'done' }];
  assert.ok(thumbTask('Props', model, 'mesh'), 'the job is over');
  store.jobs = [];

  // The hub's answer to an upload: kept only when the prop has the thumbnail of what was drawn.
  assert.equal(kept(prop({ chosen: SHEET }, { images: thumb(SHEET) }), whole), true);
  assert.equal(kept(prop({ chosen: 'images/0001.png' }, {}), whole), false, 'refused: the prop shows another picture');
  assert.equal(kept(prop({ chosen: 'images/0001.png' }, { images: thumb('images/0001.png') }), whole), false);
  const snapshot = thumbTask('Props', model, 'mesh');
  assert.deepEqual([snapshot.kind, snapshot.of], ['glb', null]);
  assert.equal(kept({ ...model, thumbs: { mesh: { path: 'thumbs/mesh.webp', v: 1 } } }, snapshot), true);
});
