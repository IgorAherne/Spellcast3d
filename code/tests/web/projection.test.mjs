// The Projection panel's numbers (no browser): the model the bake made first, then how the projection went.
//   node --test code/tests/web/projection.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

globalThis.addEventListener ??= () => {};
globalThis.document ??= { addEventListener() {}, removeEventListener() {}, createElement: () => ({}) };
const { bakeFacts } = await import(new URL('../../spellcast/static/js/stages/projection.js', import.meta.url));

test('a bake\'s numbers: the model\'s vertices and texture, then the projection\'s own', () => {
  const stats = {
    render_vertices: 10511, triangles: 9064, resolution: 2048, coverage: 0.4579, miss_fraction: 0.0000537,
    ray_distance_percent: 1.1394, seconds: 15.36, warnings: ['A chart is small'],
  };
  const facts = bakeFacts({ projection: { glb: 'projection/bake_001/projection.glb', stats, settings: { ray_distance: null } } });
  // A bake made before the clean mesh's own count was kept: the files' vertices alone.
  assert.deepEqual(facts.model, [['Vertices', '10,511'], ['Split at UV seams', null], ['Texture', '2048 × 2048 px']]);
  // The count Retopo shows first, then what the files hold of it (a seam's points are a vertex per side).
  const counted = bakeFacts({ projection: { glb: 'p.glb', stats: { ...stats, vertices: 2160, render_vertices: 3079 } } });
  assert.deepEqual(counted.model.slice(0, 2), [['Vertices', '2,160'], ['Split at UV seams', '3,079 in the saved files']]);
  const whole = bakeFacts({ projection: { glb: 'p.glb', stats: { ...stats, vertices: 500, render_vertices: 500 } } });
  assert.deepEqual(whole.model.slice(0, 2), [['Vertices', '500'], ['Split at UV seams', null]]);
  assert.deepEqual(facts.projection, [['Covered', '46 %'], ['Missed', '0.01 %'],
    ['Ray distance', '1.14 % of the size (automatic)'], ['Time', '15.4 s']]);
  assert.deepEqual(facts.warnings, ['A chart is small']);
  // No bake yet: nothing to say.
  const none = bakeFacts({ projection: { glb: null, stats } });
  assert.ok([...none.model, ...none.projection].every(([, v]) => v == null) && none.warnings.length === 0);
});
