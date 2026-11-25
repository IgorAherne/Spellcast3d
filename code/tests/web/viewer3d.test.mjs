// The 3D viewer's lights in "All" without a GPU (viewer3d.js; three.js objects need no renderer): six point lights
// start evenly spread on a sphere round the model and go round it on great circles at right angles to each other
// (not all round the equator), at a distance tied to the model's size; they never bunch up, pass near every side of
// the model in a turn, are the same for a seed, and their glows are never picked by a ray. Then the viewer itself, on
// models already in its cache: a model shown in another's place keeps the camera, and a held view.
//   node --test code/tests/web/viewer3d.test.mjs
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { test } from 'node:test';

// The page's import map (index.html), for Node.
const STATIC = new URL('../../spellcast/static/', import.meta.url);
registerHooks({
  resolve(specifier, context, next) {
    if (specifier === 'three') return { url: new URL('vendor/three/build/three.module.js', STATIC).href, shortCircuit: true };
    if (specifier.startsWith('three/addons/')) {
      return { url: new URL(`vendor/three/examples/jsm/${specifier.slice('three/addons/'.length)}`, STATIC).href, shortCircuit: true };
    }
    return next(specifier, context);
  },
});
const THREE = await import('three');
const { OrbitLights, orbitPaths, orbitPlace } = await import(new URL('js/viewer3d.js', STATIC));

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const length = (v) => Math.hypot(...v);
const degrees = (a, b) => (Math.acos(Math.max(-1, Math.min(1, dot(a, b) / (length(a) * length(b))))) * 180) / Math.PI;
const near = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

const STEPS = 720;   // places looked at in one turn
const turn = (fn) => { for (let s = 0; s < STEPS; s++) fn((s / STEPS) * Math.PI * 2); };

test('the lights start evenly spread: the six corners of an octahedron, each a quarter turn from its four neighbours', () => {
  const paths = orbitPaths();
  assert.equal(paths.length, 6);
  for (const { start } of paths) {
    assert.ok(near(length(start), 1));
    const angles = paths.map((p) => degrees(start, p.start)).sort((a, b) => a - b);
    assert.deepEqual(angles.map((a) => Math.round(a)), [0, 90, 90, 90, 90, 180]);
  }
});

test('each goes round a great circle: always as far from the centre, back where it began after a turn', () => {
  for (const path of orbitPaths()) {
    assert.ok(near(length(path.travel), 1) && near(dot(path.start, path.travel), 0), 'it sets off at right angles');
    turn((a) => assert.ok(near(length(orbitPlace(path, a)), 1)));
    orbitPlace(path, Math.PI * 2).forEach((v, i) => assert.ok(near(v, path.start[i])));
  }
});

test('three circles at right angles to each other, none the equator: one goes over the top and under the bottom', () => {
  const axes = orbitPaths().map(({ start, travel }) => cross(start, travel));
  const circles = axes.filter((a, i) => axes.findIndex((b) => near(Math.abs(dot(a, b)), 1)) === i);
  assert.equal(circles.length, 3, 'two lights to a circle');
  for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) assert.ok(near(dot(circles[i], circles[j]), 0));
  for (const axis of circles) assert.ok(Math.abs(axis[1]) < 0.75, `a circle leans ${degrees(axis, [0, 1, 0]).toFixed(0)} degrees`);
  let top = -1;
  let bottom = 1;
  for (const path of orbitPaths()) turn((a) => { const y = orbitPlace(path, a)[1]; top = Math.max(top, y); bottom = Math.min(bottom, y); });
  assert.ok(top > 0.999 && bottom < -0.999);
});

test('they never bunch up: no two lights closer than 60 degrees at any moment of a turn', () => {
  for (const seed of [1, 2, 3, 42]) {
    const paths = orbitPaths(seed);
    let closest = 180;
    turn((a) => {
      const places = paths.map((p) => orbitPlace(p, a));
      for (let i = 0; i < places.length; i++) for (let j = i + 1; j < places.length; j++) closest = Math.min(closest, degrees(places[i], places[j]));
    });
    assert.ok(closest > 59.9, `seed ${seed}: ${closest.toFixed(1)} degrees`);
  }
});

test('in a turn a light passes within 36 degrees of every side of the model, its top and bottom too', () => {
  const paths = orbitPaths();
  const sides = Array.from({ length: 400 }, (_, i) => {   // a Fibonacci sphere of directions
    const y = 1 - (2 * i + 1) / 400;
    const r = Math.sqrt(1 - y * y);
    const phi = i * Math.PI * (3 - Math.sqrt(5));
    return [Math.cos(phi) * r, y, Math.sin(phi) * r];
  });
  const nearest = sides.map(() => 180);
  turn((a) => {
    const places = paths.map((p) => orbitPlace(p, a));
    sides.forEach((side, k) => { for (const place of places) nearest[k] = Math.min(nearest[k], degrees(side, place)); });
  });
  assert.ok(Math.max(...nearest) < 36, `${Math.max(...nearest).toFixed(1)} degrees`);
});

test('the same seed gives the same paths, other seeds other ones', () => {
  assert.deepEqual(orbitPaths(7), orbitPaths(7));
  assert.deepEqual(orbitPaths(), orbitPaths());
  assert.notDeepEqual(orbitPaths(1), orbitPaths(2));
});

/** A scene with a box of `size` at `at`, and the lights fitted round it. */
function boxScene(size, at = [0, 0, 0]) {
  const scene = new THREE.Scene();
  const box = new THREE.Mesh(new THREE.BoxGeometry(size, size, size), new THREE.MeshStandardMaterial());
  box.position.fromArray(at);
  scene.add(box);
  const lights = new OrbitLights(scene);
  lights.fit(box);
  return { scene, box, lights, centre: new THREE.Vector3().fromArray(at) };
}

test('the lights go round the model at the same multiple of its radius, whatever its size and place', () => {
  const ratios = [];
  for (const [size, at] of [[2, [0, 0, 0]], [40, [5, -3, 12]]]) {
    const { lights, centre } = boxScene(size, at);
    const radius = (size * Math.sqrt(3)) / 2;   // half the box's diagonal
    for (let step = 0; step < 50; step++) {
      for (const light of lights.lights) ratios.push(light.position.distanceTo(centre) / radius);
      lights.turn(0.37);
    }
  }
  assert.ok(ratios.every((r) => near(r, ratios[0], 1e-6)), 'one distance for every light, size and moment');
  assert.ok(ratios[0] > 1.2, 'outside the model');
});

test('dark and without glows until lit; lit, every light shines alike and shows its glow', () => {
  const { lights } = boxScene(2);
  assert.equal(lights.lights.length, 6);
  assert.ok(lights.lights.every((l) => l.isPointLight && l.intensity === 0 && l.decay === 0 && !l.castShadow));
  assert.ok(lights.glows.every((g) => !g.visible));
  lights.light(true);
  assert.ok(lights.lights.every((l) => l.intensity > 0 && l.intensity === lights.lights[0].intensity));
  assert.ok(lights.glows.every((g) => g.visible));
  lights.light(false);
  assert.ok(lights.lights.every((l) => l.intensity === 0) && lights.glows.every((g) => !g.visible));
});

test('a ray through a glow finds the model, never the glow; a glow near the camera stays a dot', () => {
  const { scene, box, lights } = boxScene(2);
  lights.light(true);
  const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 100);
  camera.position.set(0, 0, 8);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  scene.updateMatrixWorld(true);
  const ray = new THREE.Raycaster();
  ray.camera = camera;
  for (const light of lights.lights) {
    ray.set(camera.position, light.position.clone().sub(camera.position).normalize());
    for (const hit of ray.intersectObject(scene, true)) assert.equal(hit.object, box);
  }
  lights.faceCamera(camera);
  const far = lights.glows.map((g) => g.scale.x);
  assert.ok(far.every((s) => s > 0 && near(s, far[0])), 'far off, all one size, tied to the model');
  camera.position.copy(lights.lights[0].position).add(new THREE.Vector3(0, 0, 0.01));
  lights.faceCamera(camera);
  assert.ok(lights.glows[0].scale.x < far[0] / 10, 'close by, a dot');
});

// ---- the shared viewer itself, with stand-ins for what needs a graphics card -----------------------------------

const { viewer } = await import(new URL('js/viewer3d.js', STATIC));

/** The viewer with a camera and controls but no renderer, and `urls` (two: all its cache keeps) in it as models. */
function rig(urls) {
  Object.assign(viewer, {
    renderer: {}, root: new THREE.Group(), orbit: new OrbitLights(new THREE.Scene()), current: null, place: null,
    scene: { userData: { key: {}, rim: {}, ambient: {} } },
    camera: new THREE.PerspectiveCamera(32, 1, 0.01, 1000),
    controls: { target: new THREE.Vector3(), enabled: true, update() {} },
  });
  viewer.cache.clear();
  for (const [i, url] of urls.entries()) {
    const box = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), new THREE.MeshStandardMaterial());
    box.geometry.translate(3 + i, 1, 0);           // (frame() moves each to the origin: a place of its own)
    const pivot = new THREE.Group().add(box);
    pivot.userData.views = { setExtraMaps() {}, apply() {}, setWire() {}, dispose() {}, edges: null };
    viewer.cache.set(url, pivot);
  }
}
const modelAt = () => viewer.current.children[0].position.toArray();

test('a model shown with keepView stands where the last one stood and the camera stays - also when a model still loading had taken that one off the screen', async () => {
  rig(['a.glb', 'b.glb']);
  assert.equal(await viewer.show('a.glb', 'glb'), true);
  const place = modelAt();
  viewer.camera.position.set(1, 2, 3);               // the artist goes round it
  const camera = viewer.camera.position.toArray();
  assert.equal(await viewer.show('b.glb', 'glb', { keepView: true }), true);
  assert.deepEqual([modelAt(), viewer.camera.position.toArray()], [place, camera]);
  // A model not in the cache is asked for, and another before it is there: a job's end does this (the old result
  // under its new address, then the new result).
  const superseded = viewer.show('not-loaded.glb', 'glb', { keepView: true });
  assert.equal(viewer.current, null, 'nothing on screen while it loads');
  assert.equal(await viewer.show('a.glb', 'glb', { keepView: true }), true);
  assert.deepEqual([modelAt(), viewer.camera.position.toArray()], [place, camera]);
  assert.equal(await superseded, false);
  // Without keepView a model is framed anew.
  assert.equal(await viewer.show('b.glb', 'glb'), true);
  assert.notDeepEqual(viewer.camera.position.toArray(), camera);
});

test('a held view: the controls are off and nothing frames the model, until it is let go or another model is shown', async () => {
  rig(['a.glb', 'b.glb']);
  await viewer.show('a.glb', 'glb');
  viewer.camera.position.set(1, 2, 3);
  const camera = viewer.camera.position.toArray();
  viewer.holdView(true);
  assert.equal(viewer.controls.enabled, false);
  assert.equal(viewer.frame(), false, '"reset the view"');
  assert.equal(viewer.frame({ keepDirection: true }), false, 'the F key');
  assert.deepEqual(viewer.camera.position.toArray(), camera);
  viewer.holdView(false);
  assert.equal(viewer.controls.enabled, true);
  assert.equal(viewer.frame(), true);
  viewer.holdView(true);
  await viewer.show('b.glb', 'glb');
  assert.deepEqual([viewer.held, viewer.controls.enabled], [false, true], 'a hold is for the model that was on screen');
});
