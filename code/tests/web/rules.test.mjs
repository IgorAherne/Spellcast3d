// Tests of the stage views' rules (static/js/stages/rules.js): node --test code/tests/web/rules.test.mjs
// (run by code/tests/test_web.py when Node.js is installed).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  RETOPO_ORDER, STAGE_TOOLS, bakeFresh, carryPlan, effectiveViews, frontOnly, highPoly, meshFresh, meshSource, nameFromPrompt,
  planCount, asideHint, planSummary, reaches, retopoFresh, retopoMethod, retopoMissing, retopoPlan, retopoReady,
  retopoUsed, retopoVariant, rowPlan, omniBatch, omniChosen, omniRepaints,
  omniOwn, omniStatus, omniTakeOff, omniTakenOff, omniTextured, omniViews, omniViewsEye, omniViewsHidden, omnitextureFresh,
  omnitextureUsed, toolOf, viewsAside,
  tileable, viewsSignature,
  warmAsk,
} from '../../spellcast/static/js/stages/rules.js';

/**
 * A prop with the results given (paths as the hub stores them). method: the retopology method it picked
 * (retopo.settings.method; Instant Meshes here unless given, null: none picked, so TriFlow, the default);
 * madeWith: the method that made its retopo (retopo.method). sizes: the sizes it asks for; made: those it has a clean
 * mesh of (its variants; the first is retopo.obj unless `retopo` names it), picked: whether the one used is picked.
 */
function box(id, { image, mesh, meshFrom, imported, retopo, retopoFrom, uv, bake, stale, method, madeWith, sizes, made = [],
  picked = false } = {}) {
  const variants = made.map((size) => ({ size, obj: `retopo/${size}k/mesh.obj`, stats: { vertices: size * 1000 } }));
  const obj = retopo ?? variants[0]?.obj ?? null;
  return {
    id,
    images: { chosen: image ?? null },
    mesh: { glb: mesh ?? null, source: meshFrom ?? image ?? null, imported: imported ? { materials: [] } : null },
    retopo: {
      obj, source: retopoFrom ?? mesh ?? null, stats: uv === undefined ? null : { has_uv: uv },
      method: madeWith ?? (obj ? 'imesh' : null), accepted: picked, variants,
      settings: { ...(method === null ? {} : { method: method ?? 'imesh' }), ...(sizes ? { sizes } : {}) },
    },
    projection: { glb: bake ?? null, stale: !!stale },
  };
}

/** closed: stages ("mesh") or single tools ("retopo:triflow") that can't take work here. */
const deps = (pendingJobs = [], closed = []) => ({
  pending: (stage, b) => pendingJobs.includes(`${stage}:${b.id}`),
  open: (stage, tool) => !closed.includes(stage) && !closed.includes(`${stage}:${tool}`),
});

test('a 3D model is fresh while it was made from the chosen picture, or is the artist\'s own', () => {
  assert.equal(meshFresh(box('a', { image: 'images/1.png', mesh: 'mesh/mesh.glb' })), true);
  assert.equal(meshFresh(box('a', { image: 'images/2.png', mesh: 'mesh/mesh.glb', meshFrom: 'images/1.png' })), false);
  assert.equal(meshFresh(box('a', { mesh: 'mesh/imported_001.glb', imported: true, meshFrom: 'mesh/import_001/a.fbx' })), true);
  assert.equal(meshFresh(box('a', { image: 'images/1.png' })), false);
});

test('a prop with views makes its 3D model from them: a new view or a moved slice makes the model old', () => {
  const views = [
    { view: 'back', src: 'images/0001.png', rect: [512, 0, 512, 512] },
    { view: 'front', src: 'images/0001.png', rect: [0, 0, 512, 512] },
    { view: 'left', src: 'images/upload_002.png', rect: null },
  ];
  assert.equal(viewsSignature(views),
    'front:images/0001.png:0,0,512,512|left:images/upload_002.png:*|back:images/0001.png:512,0,512,512',
    'in the views\' order (front, left, back, right) whatever order they were saved in; * = the whole picture');
  assert.equal(viewsSignature([]), '');
  assert.equal(meshSource({ chosen: 'images/0001.png', views: [] }), 'images/0001.png', 'no views: the chosen picture');
  assert.equal(meshSource({ chosen: 'images/0001.png' }), 'images/0001.png');
  assert.equal(meshSource({ chosen: null }), null);
  assert.equal(meshSource({ chosen: 'images/0001.png', views: views.slice(1, 2) }), 'views:front:images/0001.png:0,0,512,512',
    'one view is enough');

  const made = { id: 'a', images: { chosen: 'images/0001.png', views }, mesh: { glb: 'mesh/mesh.glb', source: `views:${viewsSignature(views)}` } };
  assert.equal(meshFresh(made), true);
  assert.equal(meshFresh({ ...made, images: { ...made.images, views: [...views].reverse() } }), true, 'the order they are saved in does not matter');
  const moved = views.map((v) => (v.view === 'back' ? { ...v, rect: [520, 0, 504, 512] } : v));
  assert.equal(meshFresh({ ...made, images: { ...made.images, views: moved } }), false);
  assert.equal(meshFresh({ ...made, images: { ...made.images, views: views.slice(1) } }), false, 'a view removed');
  assert.equal(meshFresh({ ...made, images: { ...made.images, views: [] } }), false, 'every view removed: the chosen picture now');
  const fromPicture = { id: 'b', images: { chosen: 'images/0001.png', views: [] }, mesh: { glb: 'mesh/mesh.glb', source: 'images/0001.png' } };
  assert.equal(meshFresh(fromPicture), true);
  assert.equal(meshFresh({ ...fromPicture, images: { ...fromPicture.images, views: views.slice(1, 2) } }), false, 'sliced after');
});

test('the views count while the chosen picture is one of theirs; another picture is made into 3D alone', () => {
  const views = [
    { view: 'front', src: 'images/upload_001.png', rect: null },
    { view: 'left', src: 'images/upload_002.png', rect: null },
  ];
  assert.deepEqual(effectiveViews({ chosen: 'images/upload_001.png', views }), views);
  assert.deepEqual(effectiveViews({ chosen: 'images/upload_002.png', views }), views, 'any picture of theirs');
  assert.deepEqual(effectiveViews({ chosen: 'images/0003.png', views }), []);
  assert.deepEqual(effectiveViews({ chosen: null, views }), []);
  assert.deepEqual(effectiveViews({ views }), []);
  assert.deepEqual(effectiveViews(undefined), []);
  assert.equal(meshSource({ chosen: 'images/0003.png', views }), 'images/0003.png');

  // Made from the views; a generated picture chosen after: the model is old. Picking a view's picture again: fresh.
  const made = { id: 'a', images: { chosen: 'images/upload_001.png', views }, mesh: { glb: 'mesh/mesh.glb', source: `views:${viewsSignature(views)}` } };
  assert.equal(meshFresh(made), true);
  assert.equal(meshFresh({ ...made, images: { ...made.images, chosen: 'images/0003.png' } }), false);
  assert.equal(meshFresh({ ...made, images: { ...made.images, chosen: 'images/upload_002.png' } }), true);
  // Made from that other picture alone: fresh while it is chosen, whatever views wait.
  const alone = { id: 'b', images: { chosen: 'images/0003.png', views }, mesh: { glb: 'mesh/mesh.glb', source: 'images/0003.png' } };
  assert.equal(meshFresh(alone), true);
});

test('going back to Images makes no 3D model "to do" again: only another picture or other views do', () => {
  const [a, b] = ['images/upload_009.png', 'images/0004.png'];
  const slice = (view, rect) => ({ view, src: a, rect });
  // A sheet cut into "4 quadrants" (slicer.presetViews): the front and the back on top, the sides under them.
  const quad = [slice('front', [0, 0, 512, 512]), slice('left', [0, 512, 512, 512]),
    slice('back', [512, 0, 512, 512]), slice('right', [512, 512, 512, 512])];
  const row = ['front', 'left', 'back', 'right'].map((view, i) => slice(view, [i * 256, 0, 256, 1024]));
  const prop = (images, mesh = {}) => ({
    id: 'p',
    images: { candidates: [a, b], settings: { prompt: 'a bear', use_references: true }, ...images },
    mesh: { glb: 'mesh/mesh.glb', settings: { seed: 1, resolution: 1024 }, ...mesh },
    retopo: { obj: 'retopo/triflow.obj', method: 'triflow', source: 'mesh/mesh.glb', stale: false, settings: { method: 'triflow' } },
    projection: { glb: 'projection/bake_001/projection.glb', stale: false },
  });
  const live = { pending: () => false, open: () => true };
  const nothing = (p, why) => {
    assert.equal(meshFresh(p), true, why);
    assert.deepEqual(carryPlan('projection', [p], live), [], `${why}: nothing is queued, down to the bake`);
  };
  const again = (p, why) => {
    assert.equal(meshFresh(p), false, why);
    assert.deepEqual(carryPlan('mesh', [p], live), [['mesh', ['p'], 'pixal3d']], why);
  };

  // Made from the sheet's four views.
  const source = meshSource({ chosen: a, views: quad });
  assert.equal(source, 'views:front:images/upload_009.png:0,0,512,512|left:images/upload_009.png:0,512,512,512|'
    + 'back:images/upload_009.png:512,0,512,512|right:images/upload_009.png:512,512,512,512');
  const made = (images) => prop(images, { source });
  nothing(made({ chosen: a, views: quad }), 'the same picture chosen again');
  again(made({ chosen: b, views: quad }), 'another picture chosen: it is made into 3D alone');
  nothing(made({ chosen: a, views: quad }), '... and the first one again');
  nothing(made({ chosen: a, views: [...quad].reverse() }), 'the same views saved in another order');
  nothing(made({ chosen: a, views: JSON.parse(JSON.stringify(quad).replaceAll('512', '512.0')) }), 'the same numbers written 512.0');
  nothing(made({ chosen: a, views: quad.map((v) => ({ ...v, rect: [...v.rect], extra: 1 })) }), 'the same views sent again as new objects');
  nothing(made({ chosen: a, views: quad, settings: { prompt: 'a fox', use_references: false, from_sketch: true } }),
    'another tab of Images, other words: only its pictures are made into 3D');
  nothing(made({ chosen: a, views: quad, candidates: [a, b, 'images/0005.png'], references: ['reference/reference_001.png'], sketch: 'sketch/sketch.png' }),
    'more pictures, an example, a sketch: the chosen one is the same');
  nothing({ ...made({ chosen: a, views: quad }), thumbs: {} }, 'its thumbnail drawn again');
  nothing(prop({ chosen: a, views: quad }, { source, settings: { seed: 7, resolution: 1536, decimation_target: 50000, texture_size: 4096 } }),
    'other 3D settings: "Make 3D again" asks for them, they make nothing old');
  again(made({ chosen: a, views: quad.map((v) => (v.view === 'back' ? { ...v, rect: [520, 0, 504, 512] } : v)) }), 'a slice moved');
  again(made({ chosen: a, views: quad.map((v) => ({ ...v, view: { front: 'back', back: 'front' }[v.view] || v.view })) }), 'two slices swapped their sides');
  again(made({ chosen: a, views: row }), '"4 in a row" pressed on a sheet made as "4 quadrants"');
  nothing(made({ chosen: a, views: quad }), '... and "4 quadrants" again: the model it has is that one');
  again(made({ chosen: a, views: [] }), 'the views cleared');

  // Made from one picture alone: the same there.
  const alone = (images) => prop(images, { source: a });
  nothing(alone({ chosen: a }), 'the same picture chosen again');
  again(alone({ chosen: b }), 'another picture');
  nothing(alone({ chosen: a, views: [] }), 'back to the first');
  nothing(alone({ chosen: a, views: [{ view: 'front', src: b, rect: null }] }), 'views marked on another picture wait');
  again(alone({ chosen: a, views: [slice('front', [0, 0, 512, 512])] }), 'a slice of it marked: that part is made into 3D');
});

test('views that wait on another picture are said from what the artist sees, and only while that picture is there', () => {
  const sheet = ['front', 'left', 'back', 'right'].map((view, i) => ({ view, src: 'images/0001.png', rect: [i * 256, 0, 256, 512] }));
  const pictures = ['images/0001.png', 'images/0002.png', 'images/upload_003.png'];
  const other = { chosen: 'images/0002.png', candidates: pictures, views: sheet };
  assert.equal(viewsAside(other).length, 4);
  assert.equal(asideHint(other), 'This picture has no views marked, so it is made into 3D on its own. '
    + 'Another picture of this prop has 4 views: choose that picture in Images to use them.');
  assert.equal(asideHint({ ...other, views: sheet.slice(0, 1) }), 'This picture has no views marked, so it is made into 3D on its own. '
    + 'Another picture of this prop has 1 view: choose that picture in Images to use it.');
  const spread = [sheet[0], { view: 'back', src: 'images/upload_003.png', rect: null }];
  assert.equal(asideHint({ ...other, views: spread }), 'This picture has no views marked, so it is made into 3D on its own. '
    + 'Other pictures of this prop have 2 views: choose one of them in Images to use them.');
  // Nothing to say: the views are used, the prop has none, no picture is chosen, or their picture is gone.
  assert.equal(asideHint({ ...other, chosen: 'images/0001.png' }), '');
  assert.equal(asideHint({ ...other, views: [] }), '');
  assert.equal(asideHint({ chosen: 'images/0002.png', candidates: pictures }), '');
  assert.equal(asideHint({ ...other, chosen: null }), '');
  assert.equal(asideHint({ ...other, candidates: ['images/0002.png'] }), '', 'the sheet was removed');
  assert.equal(asideHint({ ...other, views: [null, {}, { view: 'top', src: 'images/0001.png' }] }), '', 'no real view');
  assert.deepEqual(viewsAside(undefined), []);
});

test('a model made from the front view alone is out of date once the multi-view models are there', () => {
  const views = [
    { view: 'front', src: 'images/upload_001.png', rect: null },
    { view: 'back', src: 'images/upload_002.png', rect: null },
  ];
  const source = `views:${viewsSignature(views)}`;
  const images = { chosen: 'images/upload_001.png', views };
  const single = { id: 's', images, mesh: { glb: 'mesh/mesh.glb', source, stats: { mode: 'single', views: 1 } } };
  const multi = { id: 'm', images, mesh: { glb: 'mesh/mesh.glb', source, stats: { mode: 'multi', views: 2 } } };
  assert.equal(frontOnly(single), true);
  assert.equal(frontOnly(multi), false);
  assert.equal(meshFresh(single), true, 'without the models, making it again changes nothing');
  assert.equal(meshFresh(single, { multiview: false }), true);
  assert.equal(meshFresh(single, { multiview: true }), false);
  assert.equal(meshFresh(multi, { multiview: true }), true);
  // One view alone is made as a picture: no multi-view models needed.
  const one = { id: 'o', images: { chosen: 'images/upload_001.png', views: views.slice(0, 1) },
    mesh: { glb: 'mesh/mesh.glb', source: `views:${viewsSignature(views.slice(0, 1))}`, stats: { mode: 'single' } } };
  assert.equal(meshFresh(one, { multiview: true }), true);
  // Generate 3D and "Make 3D for N props" make it again then, and only then.
  const deps = (multiview) => ({ pending: () => false, open: () => true, multiview });
  assert.deepEqual(carryPlan('mesh', [single, multi, one], deps(false)), []);
  assert.deepEqual(carryPlan('mesh', [single, multi, one], deps(true)), [['mesh', ['s'], 'pixal3d']]);
  assert.deepEqual(rowPlan('mesh', [single, multi, one], deps(true)), [['mesh', ['s'], 'pixal3d']]);
});

test('a retopo is fresh for the 3D model it was made from; a bake until the hub marks it stale', () => {
  assert.equal(retopoFresh(box('a', { image: 'i.png', mesh: 'mesh/mesh.glb', retopo: 'retopo/retopo.obj' })), true);
  assert.equal(retopoFresh(box('a', { mesh: 'mesh/imported_002.glb', retopo: 'r.obj', retopoFrom: 'mesh/imported_001.glb' })), false);
  const staleRetopo = box('a', { mesh: 'm.glb', retopo: 'r.obj' });
  staleRetopo.retopo.stale = true;
  assert.equal(retopoFresh(staleRetopo), false);
  assert.equal(bakeFresh(box('a', { bake: 'projection/bake_001/projection.glb' })), true);
  assert.equal(bakeFresh(box('a', { bake: 'projection/bake_001/projection.glb', stale: true })), false);
});

test('a retopo is fresh only for the method the prop is set to now: another one makes it "to do" again', () => {
  const made = box('a', { image: 'i.png', mesh: 'm.glb', retopo: 'retopo/retopo.obj' });
  assert.equal(retopoMethod(made), 'imesh', 'the method it picked');
  assert.equal(retopoMethod(box('a', { method: null })), 'triflow', 'no method picked: TriFlow, the default');
  assert.equal(retopoMethod(undefined), 'triflow');
  assert.equal(retopoFresh(made), true);
  const older = { ...made, retopo: { ...made.retopo, method: null } };
  assert.equal(retopoFresh(older), true, 'a clean mesh from before the choice was made by Instant Meshes');
  assert.equal(retopoFresh(box('a', { mesh: 'm.glb', retopo: 'retopo/retopo.obj', method: 'lato2' })), false,
    'Instant Meshes made it, LATO.2 is picked now');
  assert.equal(retopoFresh(box('a', { mesh: 'm.glb', retopo: 'retopo/lato2.obj', madeWith: 'lato2', method: 'lato2' })), true);
  assert.equal(retopoFresh(box('a', { mesh: 'm.glb', retopo: 'retopo/triflow.obj', madeWith: 'triflow', method: 'lato2' })), false);
  assert.equal(retopoFresh(box('a', { mesh: 'm.glb', retopo: 'retopo/lato2.obj', madeWith: 'lato2', method: 'imesh' })), false,
    'Instant Meshes picked again: its viewer makes the next one');
  assert.equal(retopoFresh(box('a', { mesh: 'm.glb', retopo: 'retopo/lato2.obj', madeWith: 'lato2', method: null })), false, 'unset is TriFlow');
  assert.equal(retopoFresh(box('a', { mesh: 'm.glb', retopo: 'retopo/triflow.obj', madeWith: 'triflow', method: null })), true);
  assert.equal(retopoFresh(box('a', { mesh: 'm.glb', retopo: 'retopo/retopo.obj', method: null })), false,
    'Instant Meshes made it, nothing picked: TriFlow makes the next one');
  const stale = box('a', { mesh: 'm.glb', retopo: 'retopo/triflow.obj', madeWith: 'triflow', method: 'triflow' });
  stale.retopo.stale = true;
  assert.equal(retopoFresh(stale), false, 'a new 3D model still makes it old');
});

test('each stage\'s work runs on its tool; Retopo on the method of each prop', () => {
  assert.deepEqual(STAGE_TOOLS.retopo, ['triflow', 'lato2', 'imesh'], 'the default first (paths.STAGE_TOOLS)');
  assert.deepEqual(STAGE_TOOLS.images, ['qwen', 'forge'], 'Qwen-Image 2.1 by default, Forge Neo when spellcast3d-config.ini says so');
  assert.deepEqual([...RETOPO_ORDER].sort(), [...STAGE_TOOLS.retopo].sort(), 'every method has its turn in a carry');
  assert.deepEqual(RETOPO_ORDER, ['triflow', 'lato2', 'imesh'],
    'the GPU tools right after Pixal3D (TriFlow, the default, first), Instant Meshes (CPU) last');
  const lato = box('a', { image: 'i.png', method: 'lato2' });
  assert.equal(toolOf('retopo', lato), 'lato2');
  assert.equal(toolOf('retopo', box('b')), 'imesh');
  assert.equal(toolOf('retopo', box('b', { method: null })), 'triflow', 'nothing picked: the default');
  assert.equal(toolOf('mesh', lato), 'pixal3d');
  assert.equal(toolOf('projection', lato), 'imesh');
  assert.equal(toolOf('images', lato), 'qwen', 'the default image generator (the page asks the hub which one runs)');
});

test('Retopo All queues one job per method, TriFlow then LATO.2 then Instant Meshes, for the props that lack a size', () => {
  const boxes = [
    box('a', { image: 'i.png', mesh: 'm.glb', sizes: [2] }),
    box('b', { image: 'i.png', mesh: 'm.glb', method: 'triflow', sizes: [4, 22] }),
    box('c', { image: 'i.png', method: 'lato2', sizes: [2] }),   // no 3D model yet
    box('d', { image: 'i.png', mesh: 'm.glb', method: 'lato2', sizes: [2] }),
    // Instant Meshes made it, TriFlow is picked now: made again with TriFlow.
    box('e', { image: 'i.png', mesh: 'm.glb', made: [2], picked: true, method: 'triflow', sizes: [2] }),
    box('f', { image: 'i.png', mesh: 'm.glb', made: [2], picked: true, madeWith: 'lato2', method: 'lato2', sizes: [2] }),
    box('g', { image: 'i.png', mesh: 'm.glb', made: [4], picked: true, madeWith: 'triflow', method: 'triflow', sizes: [4, 22] }),
    box('h', { image: 'i.png', mesh: 'm.glb', method: 'triflow', sizes: [] }),   // every box erased
  ];
  const plan = retopoPlan(boxes, deps());
  assert.deepEqual(plan, [['retopo', ['b', 'e', 'g'], 'triflow'], ['retopo', ['d'], 'lato2'], ['retopo', ['a'], 'imesh']]);
  assert.equal(planCount(plan, 'retopo'), 5);
  const name = (t) => ({ lato2: 'LATO.2', triflow: 'TriFlow', imesh: 'Instant Meshes' })[t];
  assert.equal(planSummary(plan, name),
    'Retopologizes 5 props (3 with TriFlow, then 1 with LATO.2, then 1 with Instant Meshes).');
  // A 3D model on its way: retopologized after it, every size anew.
  assert.deepEqual(retopoPlan(boxes, deps(['mesh:c', 'mesh:f']))[1], ['retopo', ['c', 'd', 'f'], 'lato2']);
  // A tool that can't run here leaves its own props out; queued already: not again.
  assert.deepEqual(retopoPlan(boxes, deps([], ['retopo:triflow'])), [['retopo', ['d'], 'lato2'], ['retopo', ['a'], 'imesh']]);
  assert.deepEqual(retopoPlan(boxes, deps(['retopo:d', 'retopo:b'])), [['retopo', ['e', 'g'], 'triflow'], ['retopo', ['a'], 'imesh']]);
  // Nothing left to do: "again" makes every prop that asks for a size anew.
  assert.deepEqual(retopoPlan(boxes, deps(), { again: true }),
    [['retopo', ['b', 'e', 'g'], 'triflow'], ['retopo', ['d', 'f'], 'lato2'], ['retopo', ['a'], 'imesh']]);
  // Going down to Retopo, or on to the bake, never retopologizes.
  assert.deepEqual(carryPlan('retopo', boxes, deps()), [['mesh', ['c'], 'pixal3d']]);
  assert.ok(![...carryPlan('projection', boxes, deps()), ...rowPlan('projection', boxes, deps()), ...rowPlan('retopo', boxes, deps())]
    .some(([stage]) => stage === 'retopo'));
});

test('a prop is ready for the next stage once it uses a picked clean mesh of its 3D model; it lacks the sizes not made', () => {
  const made = (more = {}) => box('a', { image: 'i.png', mesh: 'm.glb', method: 'lato2', madeWith: 'lato2', sizes: [4, 22, 6],
    made: [4, 22], ...more });
  assert.deepEqual([retopoReady(made({ picked: true })), retopoReady(made())], [true, false]);
  assert.equal(retopoReady(made({ picked: true, method: 'triflow' })), false, 'another method picked');
  const old = made({ picked: true });
  old.retopo.stale = true;
  assert.equal(retopoReady(old), false, 'made from an older 3D model');
  assert.deepEqual([retopoUsed(made()), retopoUsed(made({ retopo: 'retopo/22k/mesh.obj' })), retopoUsed(box('x'))], [4, 22, null]);
  assert.deepEqual(retopoMissing(made()), [6]);
  assert.deepEqual(retopoMissing(made({ made: [4, 22, 6] })), []);
  assert.deepEqual(retopoMissing(old), [4, 22, 6], 'old ones count for none');
  assert.deepEqual(retopoMissing(made({ made: [] })), [4, 22, 6]);
  assert.deepEqual(retopoMissing(box('x')), [], 'it asks for none');
  // Sizes under a thousand vertices among whole ones: numbers, compared as numbers.
  const mixed = made({ sizes: [0.25, 2, 0.3], made: [2, 0.25], retopo: 'retopo/0.25k/mesh.obj' });
  assert.deepEqual([retopoUsed(mixed), retopoVariant(mixed, 0.25)?.obj, retopoVariant(mixed, 0.3)],
    [0.25, 'retopo/0.25k/mesh.obj', null]);
  assert.deepEqual(retopoMissing(mixed), [0.3]);
});

test('Generate 3D queues a 3D model for every prop with a picture and no fresh model', () => {
  const boxes = [
    box('new', { image: 'i.png' }),
    box('done', { image: 'i.png', mesh: 'm.glb' }),
    box('older', { image: 'i2.png', mesh: 'm.glb', meshFrom: 'i1.png' }),
    box('busy', { image: 'i.png' }),
    box('own', { mesh: 'mesh/imported_001.glb', imported: true }),
    box('empty'),
  ];
  assert.deepEqual(carryPlan('mesh', boxes, deps(['mesh:busy'])), [['mesh', ['new', 'older'], 'pixal3d']]);
  assert.deepEqual(carryPlan('mesh', boxes, deps([], ['mesh'])), [], 'a 3D generator that cannot run here queues nothing');
});

test('a tileable texture is no object: no button makes it a 3D model, a clean mesh, an Omni-Texture or a bake', () => {
  const texture = (b, settings = { tileable: true }) => ({ ...b, images: { ...b.images, settings } });
  assert.deepEqual([{ tileable: true }, { tileable: true, qwen_model: 'turbo' }, { tileable: true, qwen_model: 'flux3' },
    { tileable: false }, {}].map((settings) => tileable(texture(box('t'), settings))), [true, true, false, false, false],
  'the tick, for a model that draws such pictures');
  assert.equal(tileable(box('t')), false);
  const fresh = box('object', { image: 'i.png' });
  const drawn = texture(box('texture', { image: 'i.png' }));
  for (const upTo of ['mesh', 'retopo']) {
    assert.deepEqual(carryPlan(upTo, [fresh, drawn], deps()), [['mesh', ['object'], 'pixal3d']], upTo);
  }
  assert.deepEqual(rowPlan('mesh', [drawn], deps()), []);
  assert.equal(reaches('mesh', [drawn], deps()), false);
  // What it had before the tick stays, and nothing more is made of it.
  const meshed = texture(box('texture', { image: 'i.png', mesh: 'm.glb', method: 'triflow', sizes: [2] }));
  assert.deepEqual(retopoPlan([meshed, box('object', { image: 'i.png', mesh: 'm.glb', method: 'triflow', sizes: [2] })], deps()),
    [['retopo', ['object'], 'triflow']]);
  const clean = { image: 'i.png', mesh: 'm.glb', made: [2], picked: true, uv: true };
  assert.deepEqual(carryPlan('projection', [texture(box('texture', clean)), box('object', clean)], deps()),
    [['projection', ['object'], STAGE_TOOLS.projection[0]]]);
  const chosen = (b) => ({ ...b, omnitexture: { settings: { chosen: true } } });
  assert.deepEqual(omniBatch([chosen(meshed), chosen(box('object', { image: 'i.png', mesh: 'm.glb' }))], deps()).map((b) => b.id),
    ['object']);
});

test('going down to Retopo makes the missing 3D models and retopologizes nothing', () => {
  const boxes = [
    box('new', { image: 'i.png' }),
    box('meshed', { image: 'i.png', mesh: 'm.glb', sizes: [2] }),
    box('clean', { image: 'i.png', mesh: 'm.glb', retopo: 'r.obj' }),
    box('own', { mesh: 'mesh/imported_001.glb', imported: true }),
    box('meshing', { image: 'i.png' }),
    box('empty'),
  ];
  assert.deepEqual(carryPlan('retopo', boxes, deps(['mesh:meshing'])), [['mesh', ['new'], 'pixal3d']]);
  assert.deepEqual(carryPlan('retopo', boxes, deps([], ['mesh'])), [], 'without a 3D generator: nothing');
});

test('Project textures bakes every prop with (or getting) a clean mesh with UVs, and no fresh bake: it makes no clean mesh', () => {
  const boxes = [
    box('ready', { image: 'i.png', mesh: 'm.glb', retopo: 'r.obj', uv: true }),
    box('baked', { image: 'i.png', mesh: 'm.glb', retopo: 'r.obj', uv: true, bake: 'b.glb' }),
    box('stale', { image: 'i.png', mesh: 'm.glb', retopo: 'r.obj', uv: true, bake: 'b.glb', stale: true }),
    box('no-uv', { image: 'i.png', mesh: 'm.glb', retopo: 'r.obj', uv: false }),
    box('meshed', { image: 'i.png', mesh: 'm.glb' }),
    box('new', { image: 'i.png' }),
  ];
  assert.deepEqual(carryPlan('projection', boxes, deps()), [['projection', ['ready', 'stale'], 'imesh']]);
  // One whose clean mesh is being made (again) is baked after it.
  assert.deepEqual(carryPlan('projection', boxes, deps(['retopo:no-uv', 'retopo:meshed'])),
    [['projection', ['ready', 'stale', 'no-uv', 'meshed'], 'imesh']]);
});

test('a stage\'s own button works on the props that arrived in it, without the work of the stages above', () => {
  const boxes = [
    box('new', { image: 'i.png' }),
    box('meshed', { image: 'i.png', mesh: 'm.glb' }),
    box('clean', { image: 'i.png', mesh: 'm.glb', retopo: 'r.obj', uv: true }),
    box('own', { mesh: 'mesh/imported_001.glb', imported: true }),
    box('meshing', { image: 'i.png' }),
    box('stale', { image: 'i.png', mesh: 'm.glb', retopo: 'r.obj', uv: true, bake: 'b.glb', stale: true }),
  ];
  assert.deepEqual(rowPlan('mesh', boxes, deps(['mesh:meshing'])), [['mesh', ['new'], 'pixal3d']]);
  assert.deepEqual(rowPlan('retopo', boxes, deps(['mesh:meshing'])), [], 'Retopo\'s own button is in its viewport');
  assert.deepEqual(rowPlan('projection', boxes, deps(['retopo:meshed'])), [['projection', ['meshed', 'clean', 'stale'], 'imesh']]);
  assert.deepEqual(rowPlan('projection', boxes, deps([], ['projection'])), [], 'a baker that cannot run here bakes nothing');
});

test('a stage can be reached when some prop has its result, has it on the way, or going down there queues work', () => {
  const pictureOnly = [box('a', { image: 'i.png' })];
  assert.equal(reaches('retopo', pictureOnly, deps()), true, 'its 3D model is made on the way');
  assert.equal(reaches('retopo', pictureOnly, deps([], ['mesh'])), false);
  assert.equal(reaches('retopo', [box('a')], deps()), false);
  assert.equal(reaches('retopo', [box('a', { mesh: 'm.glb', retopo: 'r.obj' })], deps()), true);
  assert.equal(reaches('projection', [box('a', { mesh: 'm.glb', retopo: 'r.obj', bake: 'b.glb' })], deps([], ['projection'])), true);
});

test('prop names from a prompt', () => {
  assert.equal(nameFromPrompt('a wooden barrel with iron bands, game asset'), 'Wooden barrel');
  assert.equal(nameFromPrompt('The old brass lantern standing on a table'), 'Old brass lantern standing');
  assert.equal(nameFromPrompt('masterpiece, best quality, (stone well:1.3) <lora:props:0.8>, moss'), 'Stone well');
  assert.equal(nameFromPrompt('8k, highly detailed'), '');
  assert.equal(nameFromPrompt('   '), '');
});

test('the tooltip of a stage\'s own button says all the work it queues, in order', () => {
  assert.equal(planSummary([]), '');
  assert.equal(planSummary([['retopo', ['a', 'b', 'c']]]), 'Retopologizes 3 props.');
  assert.equal(planSummary([['mesh', ['a', 'b']], ['retopo', ['a', 'b', 'c']]]),
    'Makes the 3D models of 2 props first, then retopologizes 3 props.');
  assert.equal(planSummary([['retopo', ['a']], ['projection', ['a', 'b']]]), 'Retopologizes 1 prop first, then bakes 2 props.');
  assert.equal(planSummary([['mesh', ['a']], ['retopo', ['a', 'b']], ['projection', ['a', 'b', 'c']]]),
    'Makes the 3D model of 1 prop, retopologizes 2 props, then bakes 3 props.');
  // With the tools' names, Retopo's groups in the order they run (one tool: nothing more to say).
  const name = (t) => ({ lato2: 'LATO.2', triflow: 'TriFlow', imesh: 'Instant Meshes' })[t];
  const plan = [['mesh', ['a'], 'pixal3d'], ['retopo', ['a', 'b', 'c'], 'lato2'], ['retopo', ['d', 'e'], 'triflow']];
  assert.equal(planSummary(plan, name),
    'Makes the 3D model of 1 prop first, then retopologizes 5 props (3 with LATO.2, then 2 with TriFlow).');
  assert.equal(planSummary([['retopo', ['a', 'b'], 'lato2']], name), 'Retopologizes 2 props.');
});

test('a multi-view prop made from small slices says so in the 3D stage', async () => {
  const { SMALL_VIEW_PX, smallViewsHint } = await import('../../spellcast/static/js/stages/rules.js');
  const slice = (view, rect) => ({ view, src: 'images/0001.png', rect });
  const images = (views) => ({ chosen: 'images/0001.png', candidates: ['images/0001.png'], views });
  const quadrants = [slice('front', [0, 0, 512, 512]), slice('back', [512, 0, 512, 512])];
  assert.equal(smallViewsHint(images(quadrants)), 'Its views are small slices (about 512 px): the 3D model gets softer '
    + 'than from full pictures. One picture per side or higher image resolution gives more detail.');
  assert.equal(smallViewsHint(images(quadrants.slice(0, 1))), '', 'one view: the picture is made into 3D alone');
  const big = [slice('front', [0, 0, SMALL_VIEW_PX, 1024]), slice('back', [1024, 0, 1024, 1024])];
  assert.equal(smallViewsHint(images(big)), '', 'big enough');
  const whole = [slice('front', null), { view: 'back', src: 'images/0002.png', rect: null }];
  assert.equal(smallViewsHint(images(whole)), '', 'whole pictures: their size is not known here');
  assert.equal(smallViewsHint({ ...images(quadrants), chosen: 'images/0002.png' }), '', 'views not in use');
});

// ---- Omni-Texture ------------------------------------------------------------------------------------------------

/** A prop as box() makes it, chosen for Omni-Texture or not, with an Omni-Texture result when `st` is given. */
/**
 * A prop in the Omni-Texture stage: `st` its result - a job's Omni-Texture (it has painted views), or with `own` the
 * 3D model's own texture as the artist painted it (none; `byHand`: hand paint was saved, `repaints`: how many).
 */
function painted(id, { chosen = true, st = null, stFrom = 'm.glb', stStale = false, bakeFrom = null, own = false, byHand = false, repaints = 0, ...more } = {}) {
  const b = box(id, { image: 'i.png', ...more });
  b.omnitexture = {
    settings: { chosen }, glb: st, source_mesh: st ? stFrom : null, stale: stStale,
    views: st && !own ? ['omnitexture/ot_001/views/0-front/answer.png'] : [], painted: byHand,
    repaints: Array.from({ length: repaints }, (_, i) => ({ glb: `omnitexture/ot_00${i + 2}/omnitexture.glb` })),
  };
  if (bakeFrom) b.projection.source_mesh = bakeFrom;
  return b;
}

test('an Omni-Texture is fresh while made from the 3D model the prop has; only a chosen prop uses it', () => {
  const st = 'omnitexture/ot_001/omnitexture.glb';
  assert.equal(omnitextureFresh(painted('a', { mesh: 'm.glb', st })), true);
  assert.equal(omnitextureFresh(painted('a', { mesh: 'm.glb', st, stStale: true })), false, 'a new 3D model made it stale');
  assert.equal(omnitextureFresh(painted('a', { mesh: 'mesh/imported_002.glb', st })), false, 'made from another 3D model');
  assert.equal(omnitextureFresh(painted('a', { mesh: 'm.glb' })), false);
  assert.equal(omnitextureFresh(box('a', { mesh: 'm.glb' })), false, 'a prop from before the stage');
  assert.equal(omniChosen(box('a')), false);
  assert.equal(highPoly(painted('a', { mesh: 'm.glb', st })), st, 'the bake reads the chosen prop\'s Omni-Texture');
  assert.equal(highPoly(painted('a', { mesh: 'm.glb', st, chosen: false })), 'm.glb', 'left out: its own texture again');
  assert.equal(highPoly(painted('a', { mesh: 'm.glb', st, stStale: true })), 'm.glb', 'a stale one is not used');
  assert.equal(omnitextureUsed(painted('a', { mesh: 'm.glb', st, chosen: false })), false);
  assert.deepEqual([omniOwn(painted('a', { mesh: 'm.glb', st })), omniTextured(painted('a', { mesh: 'm.glb', st }))], [false, true]);
});

test('a prop painted or masked without an Omni-Texture: its own texture is a result, used once something is on it, chosen or not', () => {
  const st = 'omnitexture/ot_001/omnitexture.glb';
  const mine = (more = {}) => painted('a', { mesh: 'm.glb', st, own: true, chosen: false, ...more });
  assert.deepEqual([omniOwn(mine()), omnitextureFresh(mine()), omniTextured(mine())], [true, true, false], 'a result, but no Omni-Texture');
  assert.equal(omniOwn(painted('a', { mesh: 'm.glb' })), false, 'no result at all');
  // A tool was taken up and nothing painted: the bake still reads the 3D model (a bake made from it stays fresh).
  assert.deepEqual([omnitextureUsed(mine()), highPoly(mine())], [false, 'm.glb']);
  assert.deepEqual([omnitextureUsed(mine({ byHand: true })), highPoly(mine({ byHand: true }))], [true, st], 'painted by hand');
  assert.deepEqual([omnitextureUsed(mine({ repaints: 1 })), highPoly(mine({ repaints: 1 }))], [true, st], 'a part painted again');
  assert.equal(omnitextureUsed(mine({ repaints: 1, chosen: true })), true, 'chosen and still waiting for its job: used meanwhile');
  assert.equal(highPoly(mine({ byHand: true, stStale: true })), 'm.glb', 'painted on a 3D model made again since: not used');
  // Chosen, it still waits for its Omni-Texture: the job is offered, and its tile says so.
  const waiting = mine({ repaints: 1, chosen: true });
  assert.deepEqual(omniBatch([waiting], deps()).map((b) => b.id), ['a']);
  assert.deepEqual(omniStatus(waiting), { state: 'idle', message: 'Chosen: waits for “Omni-Texture the chosen”' });
  assert.deepEqual(omniStatus(mine({ byHand: true })),
    { state: 'idle', message: 'Not chosen: its own texture, with what you painted on it' });
  assert.deepEqual(omniStatus(mine()), { state: 'idle', message: 'Not chosen: keeps its 3D model’s own texture' });
  assert.deepEqual(omniStatus(mine({ byHand: true, stStale: true, chosen: true })),
    { state: 'idle', message: 'Chosen: waits for “Omni-Texture the chosen”' }, 'never "stale": there is no Omni-Texture to make again');
  assert.equal(omniTakeOff(mine({ repaints: 1 })).views, false);
});

test('a bake is old once the Omni-Texture it should read changed: made, chosen or left out', () => {
  const st = 'omnitexture/ot_001/omnitexture.glb';
  const baked = { mesh: 'm.glb', retopo: 'r.obj', uv: true, bake: 'b.glb' };
  assert.equal(bakeFresh(painted('a', { ...baked, st, bakeFrom: st })), true);
  assert.equal(bakeFresh(painted('a', { ...baked, st, bakeFrom: 'm.glb' })), false, 'an Omni-Texture came after the bake');
  assert.equal(bakeFresh(painted('a', { ...baked, st, bakeFrom: st, chosen: false })), false, 'left out: bake the 3D model');
  assert.equal(bakeFresh(painted('a', { ...baked, st, bakeFrom: 'm.glb', chosen: false })), true);
  assert.equal(bakeFresh(painted('a', { ...baked, bakeFrom: 'm.glb' })), true, 'chosen, not painted yet: the 3D model is what it reads');
  assert.equal(bakeFresh(box('a', baked)), true, 'a bake from before source_mesh was kept');
});

test('the early start asks for a stopped tool, and for a running one only when the wish names a model', () => {
  const tools = {
    qwen: { installed: true, supported: true, state: 'stopped' },
    pixal3d: { installed: true, supported: true, state: 'ready' },
    lato2: { installed: false, supported: true, state: 'not_installed' },
    triflow: { installed: true, supported: false, state: 'unsupported' },
  };
  assert.deepEqual(warmAsk(tools, { tool: 'qwen' }), { tool: 'qwen', wish: {} });
  assert.deepEqual(warmAsk(tools, { tool: 'qwen', model: 'full', displace: true }), { tool: 'qwen', wish: { model: 'full', displace: true } });
  assert.equal(warmAsk(tools, { tool: 'pixal3d' }), null, 'running already');
  for (const state of ['starting', 'ready', 'busy']) {
    const running = { ...tools, qwen: { ...tools.qwen, state } };
    assert.equal(warmAsk(running, { tool: 'qwen', displace: true }), null, `${state}: nothing to ask without a model`);
    assert.deepEqual(warmAsk(running, { tool: 'qwen', model: 'turbo' }), { tool: 'qwen', wish: { model: 'turbo' } },
      `${state}: it may hold another model - the hub decides`);
  }
  for (const wish of [null, {}, { tool: null }, { tool: 'lato2' }, { tool: 'lato2', model: 'x' }, { tool: 'triflow', model: 'x' }, { tool: 'nope' }]) {
    assert.equal(warmAsk(tools, wish), null);
  }
  assert.equal(warmAsk(undefined, { tool: 'qwen' }), null, 'before the hub has answered');
});

test('going on to Retopo or to the bake never starts an Omni-Texture: only its own button paints', () => {
  // The artist chose props and moved on without pressing "Omni-Texture the chosen": nothing is painted for them.
  const boxes = [
    painted('chosen', { mesh: 'm.glb' }),
    painted('left-out', { mesh: 'm.glb', chosen: false }),
    painted('done', { mesh: 'm.glb', st: 's.glb' }),
    painted('stale', { mesh: 'm.glb', st: 's.glb', stStale: true }),
    painted('new', {}),
    painted('painting', { mesh: 'm.glb' }),
  ];
  const plan = carryPlan('retopo', boxes, deps(['omnitexture:painting']));
  assert.deepEqual(plan, [['mesh', ['new'], 'pixal3d']]);
  assert.equal(planSummary(carryPlan('retopo', boxes, deps())), 'Makes the 3D model of 1 prop.');
  const clean = { mesh: 'm.glb', retopo: 'r.obj', uv: true };
  const baked = [painted('chosen', clean), painted('left-out', { ...clean, chosen: false }), painted('done', { ...clean, st: 's.glb' })];
  for (const live of [deps(), deps(['omnitexture:chosen'])]) {
    assert.deepEqual(carryPlan('projection', baked, live), [['projection', ['chosen', 'left-out', 'done'], 'imesh']],
      'a prop chosen but not painted is baked with the texture it has; one being painted is baked after it (jobs queue)');
    assert.deepEqual(rowPlan('projection', baked, live), [['projection', ['chosen', 'left-out', 'done'], 'imesh']]);
  }
  assert.ok(![...carryPlan('projection', boxes, deps()), ...carryPlan('mesh', boxes, deps())].some(([stage]) => stage === 'omnitexture'));
});

test('"Omni-Texture the chosen" sends the chosen props with a 3D model and no fresh Omni-Texture', () => {
  const boxes = [
    painted('chosen', { mesh: 'm.glb' }),
    painted('left-out', { mesh: 'm.glb', chosen: false }),
    painted('done', { mesh: 'm.glb', st: 's.glb' }),
    painted('stale', { mesh: 'm.glb', st: 's.glb', stStale: true }),
    painted('no-model', {}),
    painted('meshing', {}),
    painted('queued', { mesh: 'm.glb' }),
  ];
  const live = deps(['mesh:meshing', 'omnitexture:queued']);
  assert.deepEqual(omniBatch(boxes, live).map((b) => b.id), ['chosen', 'stale', 'meshing']);
  assert.deepEqual(omniBatch(boxes, live, { again: true }).map((b) => b.id), ['chosen', 'done', 'stale', 'meshing']);
  assert.deepEqual(omniBatch([painted('a', { chosen: false, mesh: 'm.glb' })], deps()), []);
});

test('a tile in Omni-Texture says whether its prop is chosen, waits, is done or is stale', () => {
  assert.deepEqual(omniStatus(painted('a', { mesh: 'm.glb', chosen: false })),
    { state: 'idle', message: 'Not chosen: keeps its 3D model’s own texture' });
  assert.deepEqual(omniStatus(painted('a', { mesh: 'm.glb' })), { state: 'idle', message: 'Chosen: waits for “Omni-Texture the chosen”' });
  assert.equal(omniStatus(painted('a', { mesh: 'm.glb', st: 's.glb' })).state, 'done');
  assert.deepEqual(omniStatus(painted('a', { mesh: 'm.glb', st: 's.glb', stStale: true })),
    { state: 'stale', message: '3D model changed: Omni-Texture again' });
  assert.equal(omniStatus(painted('a', {}), { meshComing: true }).state, 'waiting');
  assert.equal(omniStatus(painted('a', {})).state, 'idle');
  assert.equal(omniStatus(painted('a', { mesh: 'm.glb' }), { canRun: false }).message,
    'Omni-Texture needs Qwen-Image 2.1 as the image generator');
  assert.deepEqual(STAGE_TOOLS.omnitexture, ['qwen']);
  assert.equal(toolOf('omnitexture', painted('a')), 'qwen');
});

test('the painted views of an Omni-Texture: what went onto the model, the layer, the sheet they were painted in', () => {
  const st = 'omnitexture/ot_001';
  const block = {
    glb: `${st}/omnitexture.glb`,
    views: [`${st}/views/0-front/answer.png`, `${st}/views/1-back/answer.png`, `${st}/views/2-front_right/answer.png`],
    projected: [`${st}/views/0-front/projected.png`, `${st}/views/2-front_right/projected.png`],
    layers: [`${st}/views/0-front/layer.webp`],
    sheets: [{ folder: `${st}/sheets/1`, views: [`${st}/views/0-front`, `${st}/views/2-front_right`] }],
  };
  const [front, back, right] = omniViews(block);
  assert.deepEqual(front, {
    folder: `${st}/views/0-front`, side: 'front', picture: `${st}/views/0-front/projected.png`,
    albedo: `${st}/views/0-front/albedo.png`, layer: `${st}/views/0-front/layer.webp`,
    sheet: `${st}/sheets/1`, together: ['front', 'front right'],
  });
  // without the lighting taken off the painting itself went on; a view painted alone has no sheet
  assert.deepEqual([back.picture, back.albedo, back.layer, back.sheet], [`${st}/views/1-back/answer.png`, null, null, null]);
  assert.equal(right.side, 'front right');
  assert.deepEqual(omniViews({ ...block, glb: null }), []);
  assert.deepEqual(omniViews(undefined), []);
});

test('the repaints the model shown holds: each on the one before it, not those taken back or taken off', () => {
  const steps = ['a/s.glb', 'b/s.glb', 'c/s.glb', 'd/s.glb'];
  const made = (glb, under, extra = {}) => ({ glb, under, picture: `${glb}.png`, layer: `${glb}.webp`, fade: 2.5, ...extra });
  // b, then c on it; d: c put on again onto a, b taken off
  const repaints = [made(steps[1], steps[0]), made(steps[2], steps[1]), made(steps[3], steps[0]), { glb: 7 }, null];
  const block = { glb: steps[1], steps, repaints };
  assert.deepEqual(omniRepaints(block), [{ glb: steps[1], picture: 'b/s.glb.png', layer: 'b/s.glb.webp', fade: 2.5, fadable: true, hidden: false }]);
  assert.deepEqual(omniRepaints({ ...block, glb: steps[2] }).map((r) => r.glb), [steps[1], steps[2]]);
  assert.deepEqual(omniRepaints({ ...block, glb: steps[3] }).map((r) => r.glb), [steps[3]]);
  assert.deepEqual(omniRepaints({ ...block, glb: steps[0] }), []);
  // one whose model under it is no longer known is still shown, but cannot be put on again
  const old = omniRepaints({ ...block, glb: 'z/s.glb', repaints: [made('z/s.glb', null, { layer: null, fade: 'x' })] });
  assert.deepEqual(old, [{ glb: 'z/s.glb', picture: 'z/s.glb.png', layer: null, fade: 1, fadable: false, hidden: false }]);
  assert.deepEqual(omniRepaints({ steps, repaints: [made(steps[1], steps[0])] }), []);
  // one the artist hid keeps its place among them
  const hid = [made(steps[1], steps[0], { hidden: true }), made(steps[2], steps[1], { hidden: 'yes' })];
  assert.deepEqual(omniRepaints({ glb: steps[2], steps, repaints: hid }).map((r) => [r.glb, r.hidden]), [[steps[1], true], [steps[2], false]]);
});

test('the painted views hidden: still listed, the eye shows them again, and none is taken off meanwhile', () => {
  const st = 'omnitexture/ot_001';
  const views = [`${st}/views/0-front/answer.png`, `${st}/views/1-back/answer.png`];
  const bare = 'omnitexture/ot_002/omnitexture.glb';
  const repaint = 'omnitexture/ot_003/omnitexture.glb';
  const shown = { mesh: { glb: 'mesh/mesh.glb' }, omnitexture: { glb: `${st}/omnitexture.glb`, source_mesh: 'mesh/mesh.glb', views } };
  const hidden = { ...shown, omnitexture: { ...shown.omnitexture, glb: repaint,
    bases: [{ glb: bare, hidden: true, shown: `${st}/omnitexture.glb`, views: views.map((v) => ({ folder: v.slice(0, v.lastIndexOf('/')), picture: v })) }],
    repaints: [{ glb: repaint, under: bare, picture: 'r.png' }] } };
  assert.deepEqual([omniViewsHidden(shown.omnitexture), omniViewsHidden(hidden.omnitexture), omniViewsHidden(undefined)], [false, true, false]);
  assert.deepEqual(omniViews(hidden.omnitexture).map((v) => v.side), ['front', 'back'], 'listed as before');
  assert.deepEqual([omniTakeOff(shown).views, omniTakeOff(hidden).views, omniTakeOff(hidden).whole, omniTakeOff(hidden).repaints],
    [true, false, true, true]);
  // The eye: to hide them only while the 3D model is the one they were painted on; to show them again always.
  const stale = (box) => ({ ...box, omnitexture: { ...box.omnitexture, stale: true } });
  assert.deepEqual([shown, hidden, stale(shown), stale(hidden)].map(omniViewsEye), [true, true, false, true]);
});

test('a view taken off: the model composed again holds the others, each with its own picture and layer now', () => {
  const st = 'omnitexture/ot_001';
  const [front, back] = [`${st}/views/0-front`, `${st}/views/1-back`];
  const composed = 'omnitexture/ot_002/omnitexture.glb';
  const repaint = 'omnitexture/ot_003/omnitexture.glb';
  const block = {
    glb: repaint,
    steps: [`${st}/omnitexture.glb`, composed, repaint],
    views: [`${front}/answer.png`, `${back}/answer.png`],
    projected: [`${front}/projected.png`, `${back}/projected.png`],
    layers: [`${front}/layer.webp`, `${back}/layer.webp`],
    bases: [{ glb: composed, views: [{ folder: back, picture: 'omnitexture/ot_002/views/1-back/projected.png',
      layer: 'omnitexture/ot_002/views/1-back/layer.webp' }] }],
    repaints: [{ glb: repaint, under: composed, picture: 'r.png' }],
  };
  const shown = omniViews(block);
  assert.deepEqual(shown.map((v) => [v.folder, v.picture, v.layer, v.albedo]), [[back,
    'omnitexture/ot_002/views/1-back/projected.png', 'omnitexture/ot_002/views/1-back/layer.webp', `${back}/albedo.png`]]);
  assert.equal(omniRepaints(block).length, 1);
  assert.deepEqual(omniViews({ ...block, glb: composed }).map((v) => v.folder), [back]);
  // Ctrl+Z to the job's result: both views as the job painted them
  assert.deepEqual(omniViews({ ...block, glb: `${st}/omnitexture.glb` }).map((v) => [v.picture, v.layer]),
    [[`${front}/projected.png`, `${front}/layer.webp`], [`${back}/projected.png`, `${back}/layer.webp`]]);
});

test("the × takes a repaint off, a view but the only one, the whole of a result with painted views, and waits while the prop's Omni-Texture changes", () => {
  const st = 'omnitexture/ot_001';
  const prop = (views, extra = {}) => ({
    mesh: { glb: 'mesh/mesh.glb' },
    omnitexture: { glb: `${st}/omnitexture.glb`, source_mesh: 'mesh/mesh.glb', views: views.map((v) => `${st}/views/${v}/answer.png`), ...extra },
  });
  const both = prop(['0-front', '1-back']);
  assert.deepEqual(omniTakeOff(both), { views: true, repaints: true, whole: true, waits: false });
  assert.deepEqual([omniTakeOff(prop(['0-front'])).views, omniTakeOff(prop(['0-front'])).whole], [false, true],
    'the only view stays: with it the whole result goes');
  assert.equal(omniTakeOff(prop([])).whole, false, 'the prop\'s own texture is no Omni-Texture to take off');
  // Taken off, the prop has no result, and says so while Ctrl+Z can still bring it back.
  const off = prop(['0-front'], { glb: '', steps: [`${st}/omnitexture.glb`, ''] });
  assert.deepEqual([omniTakenOff(off), omniTakenOff(both), omniTakenOff(prop([], { glb: '' })), omniTextured(off), omniTakeOff(off).whole],
    [true, false, false, false, false]);
  assert.equal(omniTakeOff(prop(['0-front', '1-back'], { stale: true })).views, false, 'its 3D model was made again');
  assert.equal(omniTakeOff({ ...both, mesh: { glb: 'mesh/other.glb' } }).views, false);
  // one view left on the model shown: a result composed again without the other
  const composed = prop(['0-front', '1-back'], { glb: 'omnitexture/ot_002/omnitexture.glb',
    bases: [{ glb: 'omnitexture/ot_002/omnitexture.glb', views: [{ folder: `${st}/views/1-back` }] }] });
  assert.equal(omniTakeOff(composed).views, false);
  // where Ctrl+Z cannot bring it back (Omni-Texture cannot run here), none
  assert.deepEqual(omniTakeOff(both, { canRun: false }), { views: false, repaints: false, whole: false, waits: false });
  for (const now of [{ job: { status: 'running' } }, { changing: true }, { changing: {} }]) {
    assert.equal(omniTakeOff(both, now).waits, true, JSON.stringify(now));
  }
});
