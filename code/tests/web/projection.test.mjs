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
  assert.deepEqual(facts.model, [['Vertices', '10,511'], ['Texture', '2048 × 2048 px']]);
  assert.deepEqual(facts.projection, [['Covered', '46 %'], ['Missed', '0.01 %'],
    ['Ray distance', '1.14 % of the size (automatic)'], ['Time', '15.4 s']]);
  assert.deepEqual(facts.warnings, ['A chart is small']);
  // No bake yet: nothing to say.
  const none = bakeFacts({ projection: { glb: null, stats } });
  assert.ok([...none.model, ...none.projection].every(([, v]) => v == null) && none.warnings.length === 0);
});
