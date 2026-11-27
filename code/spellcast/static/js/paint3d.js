// Painting on the baked low poly in the Projection viewer (stages/paintbar.js is its toolbar): a marker of any colour
// and a clone stamp, painted into the base colour texture on the graphics card, the way Blender's and Substance's
// projection painting does it. Loaded with viewer3d.js (both need three.js).
//
// A stroke is painted on the screen first: its dabs (stages/paintstroke.js) go into a screen-sized coverage mask.
// Then the model is drawn into its own texture, triangle by triangle at their UV places: each texel finds where it
// lies on the screen, and takes the stroke's colour by the mask there if the camera sees it (a depth and UV picture
// of the model, made when the stroke began, says which surface is in front) and it faces the camera. Each texel is
// computed from the texture as it was when the stroke began, so a stroke never builds up over itself. The triangles
// are also drawn grown by a few texels into the gutter around their UV islands, so the seams and the far mip levels
// get the new colours too. Only the triangles that lie under the stroke on the screen are drawn (the camera holds
// still for a stroke, so where each lies is worked out once, as it begins): no dab reaches any other, so the texture
// comes out the same, and a model of several hundred thousand triangles costs a dab no more than a small one.
// The clone stamp reads the same picture for the UV under the brush plus the offset to its
// source, and copies the texture from there. Alt + a click (no drag) with the marker picks the texture's colour
// under the pointer; Alt + a drag still orbits.
//
// The texture keeps the PNG's own sRGB values (a raw RGBA8 target; the lit material decodes them itself), so every
// texel a stroke does not touch keeps its exact bytes. Undo keeps, per stroke, the tiles of the texture it may have
// changed (worked out on the CPU from the triangles under the stroke), as they were before it.
//
// A third tool, the overlay, paints a layer of its own over the base colour, in one colour (Omni-Texture's mask, in
// magenta): the same strokes, into a coverage target. It is made the first time it is painted, together with the
// texture the model then shows - the base with the overlay's colour over it, drawn again after each change of either
// - so a model never painted with it costs nothing more. The marker and the clone stamp never read or write it, the
// saved painting (readAll) never holds it, and it lives on the page alone. One undo history holds the strokes of both
// layers, each step knowing its own.
//
// The texture lives on the graphics card alone. When the browser loses the WebGL context (a driver reset, the card's
// memory running out), three.js gets it back on "webglcontextrestored" and uploads every texture again from its
// picture - but a render target comes back empty, transparent black, and three.js reads it back as zeros without a
// word. So a session keeps what the bake's files hold (the picture the model came with, then the pixels saved last),
// never paints, takes back or reads its texture while the context is gone or across a loss, and fills it from that
// again once the context is back: strokes not saved by then are lost, never saved as black. The overlay, which no
// file holds, goes with the loss: the model shows the base again, and a later stroke makes a new one.
//
// A painting is saved whole (readAll's pixels), or - by a session made for patches - as the texels alone that differ
// from what the files hold (readAll's `patch`; Omni-Texture's hub keeps them as a layer of hand paint). Such a session
// keeps the files' bytes as the card holds them: read back from its texture once it is filled from their picture, then
// the pixels saved last; readAll compares them with the texture's own, read at that moment (stages/paintstroke.js
// patchOf). Never a picture decoded a second way: a texel nobody painted is never sent.

import * as THREE from 'three';
import {
  createFootprint, createStroke, createUndo, dabsBox, dirtyTiles, patchOf, screenCorners, texelHex, tileRects,
} from './stages/paintstroke.js';
import { undoStamp } from './undo.js';

const TILES = 32;            // undo tiles per side of the texture
const FACING_SOFT = 0.2;     // the paint fades out where the surface turns away from the camera (cos of the angle)
const SEAM = 0.02;           // UVs this far apart between screen neighbours: a seam (the clone stamp does not blend there)
const HARDNESS = { marker: 1, clone: 0.55, overlay: 1 };   // the dab's edge: 1 = sharp (antialiased), lower = softer
const BASE = 'base';         // the layers: the base colour texture (saved into the files) ...
const OVERLAY = 'overlay';   // ... and the overlay over it (its tool has its name)
const CLICK_PX = 3;          // Alt + a press let go less than this far from where it began: a click (the eyedropper)
const LOST = 'The graphics card was reset, so what was painted on the model and not saved yet is gone. The model '
  + 'shows again in a moment; paint it again then.';
const LOST_WHY = 'The graphics card was reset: painting waits until the page has it back.';

/** The texture is not the one its session filled any more (the WebGL context went): error.lost says so. */
function lostError() {
  const error = new Error(LOST);
  error.lost = true;
  return error;
}

/** The layer an undo step (or a kept tile) belongs to. */
const layerOf = (step) => (step.layer === OVERLAY ? OVERLAY : BASE);

/** '#rrggbb' as the bytes the texture holds (its sRGB values themselves) -> `colour`. */
function rawColour(colour, hex) {
  const n = parseInt(String(hex).slice(1), 16) || 0;
  return colour.setRGB(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, THREE.LinearSRGBColorSpace);
}

// ---- shaders ------------------------------------------------------------------------------------------------

const COPY_VERTEX = 'void main() { gl_Position = vec4( position.xy, 0.0, 1.0 ); }';
const COPY_FRAGMENT = `
uniform sampler2D src;
void main() { gl_FragColor = texelFetch( src, ivec2( gl_FragCoord.xy ), 0 ); }`;

// The model at its UV places (paintUv), carrying its own 3D point, smooth normal and face normal.
const UV_VERTEX = `
attribute vec2 paintUv;
attribute vec3 faceNormal;
varying vec3 vWorld;
varying vec3 vNormal;
varying vec3 vFace;
void main() {
	vWorld = ( modelMatrix * vec4( position, 1.0 ) ).xyz;
	vNormal = mat3( modelMatrix ) * normal;
	vFace = mat3( modelMatrix ) * faceNormal;
	gl_Position = vec4( paintUv * 2.0 - 1.0, 0.0, 1.0 );
}`;

const ISLAND_FRAGMENT = 'void main() { gl_FragColor = vec4( 1.0 ); }';

const PAINT_FRAGMENT = `
uniform sampler2D before;
uniform sampler2D islands;
uniform sampler2D mask;
uniform sampler2D snap;
uniform mat4 viewProj;
uniform mat4 fromClip;
uniform vec4 depthOf;
uniform vec2 screenSize;
uniform vec3 colour;
uniform vec2 cloneOffset;
uniform float twoSided;
varying vec3 vWorld;
varying vec3 vNormal;
varying vec3 vFace;

#ifdef CLONE
// The texture under screen point p (GL pixels) as the stroke began: the UV the picture holds there, blended between
// its four pixels unless they straddle a seam or the edge of the model.
bool cloneColour( vec2 p, out vec4 result ) {
	if ( any( lessThan( p, vec2( 0.0 ) ) ) || any( greaterThanEqual( p, screenSize ) ) ) return false;
	vec2 f = p - 0.5;
	vec2 t = f - floor( f );
	ivec2 i = ivec2( floor( f ) );
	ivec2 hi = ivec2( screenSize ) - 1;
	vec4 a = texelFetch( snap, clamp( i, ivec2( 0 ), hi ), 0 );
	vec4 b = texelFetch( snap, clamp( i + ivec2( 1, 0 ), ivec2( 0 ), hi ), 0 );
	vec4 c = texelFetch( snap, clamp( i + ivec2( 0, 1 ), ivec2( 0 ), hi ), 0 );
	vec4 d = texelFetch( snap, clamp( i + ivec2( 1, 1 ), ivec2( 0 ), hi ), 0 );
	vec4 n = t.y < 0.5 ? ( t.x < 0.5 ? a : b ) : ( t.x < 0.5 ? c : d );
	if ( n.z <= 0.0 ) return false;
	vec2 uv = n.xy;
	if ( a.z > 0.0 && b.z > 0.0 && c.z > 0.0 && d.z > 0.0
		&& max( max( distance( a.xy, b.xy ), distance( a.xy, c.xy ) ), distance( a.xy, d.xy ) ) < ${SEAM} ) {
		uv = mix( mix( a.xy, b.xy, t.x ), mix( c.xy, d.xy, t.x ), t.y );
	}
	result = texture( before, uv );
	return true;
}
#endif

void main() {
	ivec2 texel = ivec2( gl_FragCoord.xy );
#ifdef GUTTER
	if ( texelFetch( islands, texel, 0 ).r > 0.5 ) discard;   // inside an island: its own triangle paints it
#endif
	vec4 clip = viewProj * vec4( vWorld, 1.0 );
	if ( clip.w <= 0.0 ) discard;
	vec2 ndc = clip.xy / clip.w;
	if ( any( greaterThan( abs( ndc ), vec2( 1.0 ) ) ) ) discard;
	vec2 st = ndc * 0.5 + 0.5;
	float cover = texture( mask, st ).r;
	if ( cover <= 0.0 ) discard;
	// Back along this point's own line of sight: a bent lens (viewer3d.js LensCamera) has no one eye to look from.
	vec4 nearer = fromClip * vec4( ndc, -1.0, 1.0 );
	vec3 toCam = normalize( nearer.xyz / nearer.w - vWorld );
	float face = dot( normalize( vFace ), toCam );
	float lean = dot( normalize( vNormal ), toCam );
	if ( twoSided > 0.5 ) {
		face = abs( face );
		lean = abs( lean );
	}
	if ( face <= 0.0 ) discard;
	vec2 px = st * screenSize;
	vec4 s = texelFetch( snap, min( ivec2( px ), ivec2( screenSize ) - 1 ), 0 );
	float depth = dot( depthOf, vec4( vWorld, 1.0 ) );
	if ( s.z <= 0.0 || depth > s.z + 1.5 * s.w + s.z * 2e-4 ) discard;   // something else is in front
	float w = cover * smoothstep( 0.0, ${FACING_SOFT}, lean );
	if ( w <= 0.0 ) discard;
	vec4 old = texelFetch( before, texel, 0 );
#ifdef CLONE
	vec4 src;
	if ( ! cloneColour( px + cloneOffset, src ) ) discard;
	gl_FragColor = vec4( mix( old.rgb, src.rgb, w ), old.a );
#elif defined( OVERLAY )
	gl_FragColor = vec4( mix( old.r, 1.0, w ) );   // the overlay's coverage, in every channel
#else
	gl_FragColor = vec4( mix( old.rgb, colour, w ), old.a );
#endif
}`;

// What the model shows while it has an overlay: the base colour with the overlay's colour over it by its coverage.
const COMPOSITE_FRAGMENT = `
uniform sampler2D base;
uniform sampler2D overlay;
uniform vec3 colour;
void main() {
	ivec2 texel = ivec2( gl_FragCoord.xy );
	vec4 b = texelFetch( base, texel, 0 );
	gl_FragColor = vec4( mix( b.rgb, colour, texelFetch( overlay, texel, 0 ).r ), b.a );
}`;

// Per screen pixel: the UV, the view depth and how fast it changes there (the depth test's tolerance).
const SNAP_VERTEX = `
varying vec2 vUv;
varying float vDepth;
void main() {
	vUv = uv;
	vec4 mv = modelViewMatrix * vec4( position, 1.0 );
	vDepth = - mv.z;
	gl_Position = projectionMatrix * mv;
}`;
const SNAP_FRAGMENT = `
varying vec2 vUv;
varying float vDepth;
void main() { gl_FragColor = vec4( vUv, vDepth, fwidth( vDepth ) ); }`;

// The model as the camera sees it, small (Session.sight): green where it is, red as far as the overlay covers it
// (Omni-Texture's box for the mask's words goes beside the mask, off the model: stages/maskbar.js).
const SIGHT_VERTEX = `
varying vec2 vUv;
void main() {
	vUv = uv;
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`;
const SIGHT_FRAGMENT = `
uniform sampler2D overlay;
varying vec2 vUv;
void main() { gl_FragColor = vec4( texture( overlay, vUv ).r, 1.0, 0.0, 1.0 ); }`;

// One dab of the stroke's mask: a disc, antialiased, its opacity the pen's (kept as the largest per pixel).
const DAB_VERTEX = `
attribute vec4 dab;
uniform vec2 screenSize;
varying vec2 vLocal;
varying float vRadius;
varying float vAlpha;
void main() {
	float reach = dab.z + 1.0;
	vLocal = position.xy * reach;
	vRadius = dab.z;
	vAlpha = dab.w;
	gl_Position = vec4( ( dab.xy + vLocal ) / screenSize * 2.0 - 1.0, 0.0, 1.0 );
}`;
const DAB_FRAGMENT = `
uniform float hardness;
varying vec2 vLocal;
varying float vRadius;
varying float vAlpha;
void main() {
	float d = length( vLocal );
	float cover = 1.0 - smoothstep( vRadius * hardness - 0.5, vRadius + 0.5, d );
	gl_FragColor = vec4( cover * min( 1.0, 2.0 * vRadius ) * vAlpha );
}`;

// The lit material reads the painted texture's raw sRGB values: decoded here, as three.js does for video.
const MAP_FRAGMENT_SRGB = THREE.ShaderChunk.map_fragment.replace(
  'vec4 sampledDiffuseColor = texture2D( map, vMapUv );',
  'vec4 sampledDiffuseColor = sRGBTransferEOTF( texture2D( map, vMapUv ) );',
);

function decodeSrgbMap(material) {
  const before = material.onBeforeCompile;
  const key = material.customProgramCacheKey;
  material.onBeforeCompile = function onBeforeCompile(shader, renderer) {
    before.call(this, shader, renderer);
    shader.fragmentShader = shader.fragmentShader.replace('#include <map_fragment>', MAP_FRAGMENT_SRGB);
  };
  material.customProgramCacheKey = function cacheKey() { return `${key.call(this)}|spellcast-paint-srgb`; };
  material.needsUpdate = true;
}

const fullScreen = () => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  return g;
};

function target(width, height, options = {}) {
  return new THREE.WebGLRenderTarget(width, height, {
    depthBuffer: false, type: THREE.UnsignedByteType, format: THREE.RGBAFormat, colorSpace: THREE.NoColorSpace,
    generateMipmaps: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, ...options,
  });
}

/**
 * The model's paintable part: the meshes whose own material has the one base colour texture (a bake has exactly
 * one), or null with why not.
 */
/**
 * The line of sight through a place of the camera's picture (x, y: -1 .. 1): where it starts on the near plane, its
 * direction, and how far along it the camera sees. From its two ends, whatever the lens: a bent one has no one eye,
 * and an inverse one sees less deep than its far plane.
 */
export function lineOfSight(camera, x, y) {
  const origin = new THREE.Vector3(x, y, -1).unproject(camera);
  const direction = new THREE.Vector3(x, y, 1).unproject(camera).sub(origin);
  const far = direction.length();
  return { origin, direction: direction.divideScalar(far), far };
}

/**
 * The row that gives a world point (x, y, z, 1) its depth before the camera, along its view (the camera looks down
 * its own -z). Clip w is that depth through a camera's own lens only.
 */
export function depthRow(camera, row = new THREE.Vector4()) {
  const view = camera.matrixWorldInverse.elements;
  return row.set(-view[2], -view[6], -view[10], -view[14]);
}

export function paintableParts(views) {
  const maps = new Set();
  const parts = [];
  for (const { mesh, own } of views?.meshes || []) {
    if (Array.isArray(own)) return { why: 'This model has several materials on one mesh.' };
    if (!own.map) continue;
    maps.add(own.map);
    parts.push({ mesh, own });
  }
  if (maps.size !== 1) return { why: maps.size ? 'This model has several colour textures.' : 'This model has no colour texture.' };
  const map = [...maps][0];
  const { width, height } = map.image || {};
  if (!width || width !== height) return { why: 'Only a square colour texture can be painted.' };
  return { map, parts, size: width };
}

/**
 * A mesh's triangles for the UV passes, corner by corner (no index): as they are ("exact") and grown by `grow` texels
 * around their UV triangle ("gutter", its 3D point and normal extended over the same plane) ->
 *   islands: all of the exact triangles, for the islands' mask;
 *   inside, around: the exact and the grown triangles a stroke draws - those `under` lists (an index both share,
 *     with room for every corner; Session.paintMask fills it);
 *   uv, position, grown: the exact corners' UVs and 3D points, and the grown corners' 3D points.
 */
function uvGeometries(geometry, size, grow) {
  const pos = geometry.getAttribute('position');
  const nor = geometry.getAttribute('normal');
  const uvs = geometry.getAttribute('uv');
  const index = geometry.getIndex();
  const count = index ? index.count : pos.count;
  const exact = { paintUv: new Float32Array(count * 2), position: new Float32Array(count * 3), normal: new Float32Array(count * 3), faceNormal: new Float32Array(count * 3) };
  const gutter = { paintUv: new Float32Array(count * 2), position: new Float32Array(count * 3), normal: new Float32Array(count * 3), faceNormal: new Float32Array(count * 3) };
  const at = (k) => (index ? index.getX(k) : k);
  const P = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
  const N = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
  const T = [new THREE.Vector2(), new THREE.Vector2(), new THREE.Vector2()];
  const face = new THREE.Vector3();
  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();
  const n1 = new THREE.Vector2();
  const n2 = new THREE.Vector2();
  const g = new THREE.Vector2();
  /** Writes a corner's UV, 3D point and normal into `into` at corner `o`. */
  const put = (into, o, u, v, p, n) => {
    into.paintUv[o * 2] = u;
    into.paintUv[o * 2 + 1] = v;
    into.position[o * 3] = p.x;
    into.position[o * 3 + 1] = p.y;
    into.position[o * 3 + 2] = p.z;
    into.normal[o * 3] = n.x;
    into.normal[o * 3 + 1] = n.y;
    into.normal[o * 3 + 2] = n.z;
  };
  const grownPoint = new THREE.Vector3();
  const grownNormal = new THREE.Vector3();
  /** l0 * a[0] + l1 * a[1] + l2 * a[2], component by component -> `into`. */
  const blend = (into, a, l0, l1, l2) => into.set(
    l0 * a[0].x + l1 * a[1].x + l2 * a[2].x, l0 * a[0].y + l1 * a[1].y + l2 * a[2].y, l0 * a[0].z + l1 * a[1].z + l2 * a[2].z);
  for (let k = 0; k + 2 < count; k += 3) {
    for (let j = 0; j < 3; j++) {
      const v = at(k + j);
      P[j].fromBufferAttribute(pos, v);
      if (nor) N[j].fromBufferAttribute(nor, v);
      T[j].set(uvs.getX(v), uvs.getY(v));
    }
    face.crossVectors(e1.subVectors(P[1], P[0]), e2.subVectors(P[2], P[0]));
    if (!nor) N[0].copy(face).normalize(), N[1].copy(N[0]), N[2].copy(N[0]);
    // The UV triangle in texels, its signed area, and each corner pushed out along both edges' outward normals.
    const ax = T[0].x * size; const ay = T[0].y * size;
    const bx = T[1].x * size; const by = T[1].y * size;
    const cx = T[2].x * size; const cy = T[2].y * size;
    const area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
    for (let j = 0; j < 3; j++) {
      const o = (k + j);
      put(exact, o, T[j].x, T[j].y, P[j], N[j]);
      face.toArray(exact.faceNormal, o * 3);
      face.toArray(gutter.faceNormal, o * 3);
      if (Math.abs(area) < 1e-6) {           // no area in UV: nothing to grow
        put(gutter, o, T[j].x, T[j].y, P[j], N[j]);
        continue;
      }
      const s = Math.sign(area);
      const cur = T[j];
      const prev = T[(j + 2) % 3];
      const next = T[(j + 1) % 3];
      n1.set((cur.y - prev.y) * s, -(cur.x - prev.x) * s).normalize();
      n2.set((next.y - cur.y) * s, -(next.x - cur.x) * s).normalize();
      const cos = n1.dot(n2);
      g.copy(n1).add(n2).multiplyScalar(grow / Math.max(0.25, 1 + cos));   // a sharp corner reaches at most 4x
      const ux = cur.x * size + g.x;
      const uy = cur.y * size + g.y;
      // Barycentrics of the grown corner in the UV triangle extend the 3D point and normal over the same plane.
      const l1 = ((ux - ax) * (cy - ay) - (cx - ax) * (uy - ay)) / area;
      const l2 = ((bx - ax) * (uy - ay) - (ux - ax) * (by - ay)) / area;
      const l0 = 1 - l1 - l2;
      put(gutter, o, ux / size, uy / size, blend(grownPoint, P, l0, l1, l2), blend(grownNormal, N, l0, l1, l2));
    }
  }
  const attributes = (a) => ({
    paintUv: new THREE.BufferAttribute(a.paintUv, 2), position: new THREE.BufferAttribute(a.position, 3),
    normal: new THREE.BufferAttribute(a.normal, 3), faceNormal: new THREE.BufferAttribute(a.faceNormal, 3),
  });
  const make = (attrs, drawn = null) => {
    const geo = new THREE.BufferGeometry();
    for (const [name, attribute] of Object.entries(attrs)) geo.setAttribute(name, attribute);
    geo.setIndex(drawn);
    return geo;
  };
  const exactAttributes = attributes(exact);
  const under = new THREE.BufferAttribute(new Uint32Array(count), 1).setUsage(THREE.DynamicDrawUsage);
  return {
    islands: make(exactAttributes), inside: make(exactAttributes, under), around: make(attributes(gutter), under), under,
    uv: exact.paintUv, position: exact.position, grown: gutter.position,
  };
}

/** A helper mesh that draws `geometry` where `mesh` is (its world matrix copied before each pass). */
function follower(geometry, material, mesh) {
  const m = new THREE.Mesh(geometry, material);
  m.matrixAutoUpdate = false;
  m.matrixWorldAutoUpdate = false;
  m.frustumCulled = false;
  m.userData.follows = mesh;
  return m;
}

const follow = (scene) => scene.children.forEach((m) => m.matrixWorld.copy(m.userData.follows.matrixWorld));

/**
 * The painting of one model shown in the viewer: its texture, the passes, the undo history. Made when a tool is
 * first used on it; lives as long as the viewer keeps the model (viewer3d.js disposes it with the model).
 */
class Session {
  constructor(viewer, pivot, found, { patches = false } = {}) {
    this.viewer = viewer;
    this.renderer = viewer.renderer;
    this.gl = this.renderer.getContext();
    this.pivot = pivot;
    this.parts = found.parts;
    this.size = found.size;
    this.twoSided = found.parts.some(({ own }) => own.side === THREE.DoubleSide);
    this.history = createUndo();   // the strokes of both layers (it forgets its oldest: record, tally)
    this.strokes = { [BASE]: 0, [OVERLAY]: 0 };   // strokes on each layer since it was last wiped
    this.version = 0;            // changes with every stroke, undo and redo of the base colour ...
    this.saved = 0;              // ... and the one in the bake's files (paintbar.js saves the difference)
    this.holds = 0;              // reads and saves under way (dispose waits for them: hold)
    this.disposed = false;
    this.lost = false;           // the context went since the texture was filled: it is not this one's any more
    this.losses = 0;             // context losses seen: a read across one is not to be trusted
    this.fills = 0;              // times the texture was filled (savedAs: filled again since a read)
    const size = this.size;
    const grow = Math.max(2, Math.min(8, size / 512));
    this.margin = grow + 1;
    const map = found.map;

    // (what the model samples: the painted texture, or the composite while there is an overlay)
    this.shownOptions = {
      generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter, wrapS: map.wrapS, wrapT: map.wrapT,
      anisotropy: map.anisotropy,
    };
    this.paint = target(size, size, this.shownOptions);
    this.before = target(size, size);
    this.islands = target(size, size, { format: THREE.RedFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    this.overlay = null;         // the overlay's coverage (useOverlay) ...
    this.shown = null;           // ... and the base with it composited, which the model then shows
    this.overlayHex = null;      // the colour it shows in
    this.overlayColour = new THREE.Color();
    // Per layer, the tiles strokes, undo and put changed since the fill (keep reads them).
    this.touched = { [BASE]: new Uint8Array(TILES * TILES), [OVERLAY]: new Uint8Array(TILES * TILES) };
    this.flipRows = !!map.flipY;   // the hub wants the PNG's own row order
    this.stored = map.image;       // what the bake's files hold: the picture the model came with, then the pixels saved
    this.patches = patches;        // its saves send the texels painted alone (readAll's patch) ...
    this.files = null;             // ... against these: the bytes the files hold (a promise while they are read back)

    // The model's triangles at their UV places.
    this.uniforms = {
      before: { value: this.before.texture }, islands: { value: this.islands.texture }, mask: { value: null },
      // The camera of the stroke: world -> clip, its inverse, and what gives a world point's depth in view.
      snap: { value: null }, viewProj: { value: new THREE.Matrix4() }, fromClip: { value: new THREE.Matrix4() },
      depthOf: { value: new THREE.Vector4() },
      screenSize: { value: new THREE.Vector2(1, 1) }, colour: { value: new THREE.Color() },
      cloneOffset: { value: new THREE.Vector2() }, twoSided: { value: this.twoSided ? 1 : 0 },
    };
    const pass = (defines) => this.paintMaterial(defines);
    this.materials = { marker: [pass({ GUTTER: '' }), pass({})], clone: [pass({ GUTTER: '', CLONE: '' }), pass({ CLONE: '' })] };
    this.islandMaterial = new THREE.ShaderMaterial({
      vertexShader: UV_VERTEX, fragmentShader: ISLAND_FRAGMENT, side: THREE.DoubleSide, depthTest: false,
      depthWrite: false, blending: THREE.NoBlending,
    });
    this.snapMaterials = new Map();
    this.sightTarget = this.sightScene = this.sightMaterials = null;   // (sight: made on first use)
    this.paintScene = new THREE.Scene();      // the gutter first, then the islands over it: one pass, one mip update
    this.exactScene = new THREE.Scene();      // the islands alone, for their mask
    this.snapScene = new THREE.Scene();
    // Per mesh: its corners (uvGeometries), and where its grown triangles lie on the stroke's screen (aim).
    this.corners = [];
    for (const { mesh, own } of this.parts) {
      const made = uvGeometries(mesh.geometry, size, grow);
      const around = follower(made.around, this.materials.marker[0], mesh);
      const inside = follower(made.inside, this.materials.marker[1], mesh);
      around.renderOrder = 0;
      inside.renderOrder = 1;
      inside.userData.inside = true;
      this.paintScene.add(around, inside);
      this.exactScene.add(follower(made.islands, this.islandMaterial, mesh));
      this.corners.push({ mesh, ...made });
      let snapMaterial = this.snapMaterials.get(own.side);
      if (!snapMaterial) {
        snapMaterial = new THREE.ShaderMaterial({ vertexShader: SNAP_VERTEX, fragmentShader: SNAP_FRAGMENT, side: own.side });
        this.snapMaterials.set(own.side, snapMaterial);
      }
      this.snapScene.add(follower(mesh.geometry, snapMaterial, mesh));
    }

    this.quad = new THREE.Mesh(fullScreen());
    this.quad.frustumCulled = false;
    this.quadScene = new THREE.Scene().add(this.quad);
    this.fill();

    this.canvas = this.renderer.domElement;
    this.onLost = () => {
      this.lost = true;
      this.losses++;
    };
    this.onRestored = () => this.restore();
    this.canvas.addEventListener('webglcontextlost', this.onLost);
    this.canvas.addEventListener('webglcontextrestored', this.onRestored);

    // The model now shows the painted texture; the one it came with is freed (its picture stays: stored).
    this.showMap(map, this.paint.texture, decodeSrgbMap);
    map.dispose();
  }

  /** One pass of the strokes into a layer, at the model's UV places (`defines`: which tool, gutter or islands). */
  paintMaterial(defines) {
    return new THREE.ShaderMaterial({
      uniforms: this.uniforms, defines, vertexShader: UV_VERTEX, fragmentShader: PAINT_FRAGMENT,
      side: THREE.DoubleSide, depthTest: false, depthWrite: false, blending: THREE.NoBlending,
    });
  }

  /** The model's materials take `to` for their colour texture `from` (patch: adapting each), and show it. */
  showMap(from, to, patch) {
    const views = this.pivot.userData.views;
    views.replaceMap(from, to, patch);
    if (this.viewer.current === this.pivot) views.apply(this.viewer.view);   // a base colour view on screen is made again
    this.viewer.redraw();
  }

  /**
   * The overlay, made on first use (until then it costs nothing): its coverage in a target of its own, and the base
   * with the overlay's colour over it in another, which the model shows from then on with its mip levels (the base's
   * own are made no more). `hex`: the colour it shows in ('#rrggbb').
   */
  useOverlay(hex) {
    if (this.overlay && hex === this.overlayHex) return;
    const made = !this.overlay;
    if (made) {
      const size = this.size;
      this.overlay = target(size, size, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
      this.shown = target(size, size, this.shownOptions);
      this.pass(this.overlay, new THREE.Scene(), true);      // nothing covered yet
      this.materials.overlay = [this.paintMaterial({ GUTTER: '', OVERLAY: '' }), this.paintMaterial({ OVERLAY: '' })];
      // The composite, of the whole texture (a quad) and of the texels a stroke's pass just drew (its triangles).
      const composite = {
        uniforms: { base: { value: this.paint.texture }, overlay: { value: this.overlay.texture }, colour: { value: this.overlayColour } },
        fragmentShader: COMPOSITE_FRAGMENT, depthTest: false, depthWrite: false, blending: THREE.NoBlending,
      };
      this.compositeMaterial = new THREE.ShaderMaterial({ ...composite, vertexShader: COPY_VERTEX });
      this.strokeComposite = new THREE.ShaderMaterial({ ...composite, vertexShader: UV_VERTEX, side: THREE.DoubleSide });
      this.strokeScene = new THREE.Scene();
      for (const { around, mesh } of this.corners) this.strokeScene.add(follower(around, this.strokeComposite, mesh));
      this.paint.texture.generateMipmaps = false;
    }
    this.overlayHex = hex;
    rawColour(this.overlayColour, hex);
    this.composite();
    if (made) this.showMap(this.paint.texture, this.shown.texture);
    else this.viewer.redraw();
  }

  /**
   * Draws the composite again, when there is an overlay (three.js then makes its mip levels): all of it, or -
   * `underStroke` - the texels of the triangles the stroke's pass just drew, the only ones it changed.
   */
  composite(underStroke = false) {
    if (!this.shown) return;
    if (underStroke) {
      follow(this.strokeScene);
      this.pass(this.shown, this.strokeScene);
      return;
    }
    this.quad.material = this.compositeMaterial;
    this.pass(this.shown, this.quadScene);
  }

  /** The overlay goes with the context it lived in: the model shows the base colour texture again. */
  dropOverlay() {
    if (!this.overlay) return;
    this.showMap(this.shown.texture, this.paint.texture);
    this.paint.texture.generateMipmaps = true;    // (made by the fill that follows)
    for (const one of [this.overlay, this.shown, this.compositeMaterial, this.strokeComposite, ...this.materials.overlay]) one.dispose();
    delete this.materials.overlay;
    this.overlay = this.shown = this.compositeMaterial = this.strokeComposite = this.strokeScene = this.overlayHex = null;
  }

  /** The target of a layer (null: no overlay yet). */
  layer(name) {
    return name === OVERLAY ? this.overlay : this.paint;
  }

  /**
   * Fills the texture with what the bake's files hold (stored: a picture, or saved RGBA bytes in the PNG's row order;
   * uploaded raw, no sRGB decoding), readies the stroke's "before" target and draws the islands' mask. The undo history
   * starts afresh, with nothing to save, and without an overlay.
   */
  fill() {
    this.dropOverlay();
    const size = this.size;
    const raw = this.stored.data ? new THREE.DataTexture(this.stored.data, size, size) : new THREE.Texture(this.stored);
    raw.flipY = this.flipRows;
    raw.colorSpace = THREE.NoColorSpace;
    raw.generateMipmaps = false;
    raw.minFilter = raw.magFilter = THREE.NearestFilter;
    raw.needsUpdate = true;
    const copy = new THREE.ShaderMaterial({
      uniforms: { src: { value: raw } }, vertexShader: COPY_VERTEX, fragmentShader: COPY_FRAGMENT,
      depthTest: false, depthWrite: false, blending: THREE.NoBlending,
    });
    this.quad.material = copy;
    this.pass(this.paint, this.quadScene);
    copy.dispose();
    raw.dispose();
    this.renderer.initRenderTarget(this.before);   // copied into before it is ever drawn into
    follow(this.exactScene);
    this.pass(this.islands, this.exactScene, true);
    this.history.clear();
    for (const flags of Object.values(this.touched)) flags.fill(0);
    this.strokes = { [BASE]: 0, [OVERLAY]: 0 };
    this.fills++;
    this.saved = ++this.version;
    this.lost = this.gl.isContextLost();
    if (this.patches) this.files = this.stored.data ? Promise.resolve(this.stored.data) : this.readFiles();
    this.viewer.redraw();
  }

  /**
   * The bytes the files hold, read back from the texture just filled from their picture; they then stand in for the
   * picture (a loss under the read: the texture is filled from the picture and read again).
   */
  readFiles() {
    const fills = this.fills;
    const read = this.readWhole(this.paint);
    read.then((data) => { if (fills === this.fills) this.stored = { data }; }, () => {});
    return read;
  }

  /** The context is back after a loss: the texture is filled again (once, whoever hears of it first). */
  restore() {
    if (this.lost && !this.disposed) this.fill();
  }

  /** The texture on the card is still the one this session filled: no context loss since, not freed. */
  intact() {
    return !this.lost && !this.freed && !this.gl.isContextLost();
  }

  /** Keeps the texture while a read or a save of it is under way, also past dispose() -> release() (once). */
  hold() {
    this.holds++;
    let held = true;
    return () => {
      if (!held) return;
      held = false;
      this.holds--;
      if (this.disposed && !this.holds) this.free();
    };
  }

  /** Renders `scene` into `rt` (cleared to transparent black first when asked), leaving the renderer as it was. */
  pass(rt, scene, clear = false, camera = this.viewer.camera) {
    const r = this.renderer;
    const autoClear = r.autoClear;
    const before = r.getRenderTarget();
    const colour = r.getClearColor(new THREE.Color());
    const alpha = r.getClearAlpha();
    r.autoClear = false;
    r.setRenderTarget(rt);
    if (clear) {
      r.setClearColor(0x000000, 0);
      r.clear(true, true, false);
      r.setClearColor(colour, alpha);
    }
    r.render(scene, camera);
    r.setRenderTarget(before);
    r.autoClear = autoClear;
  }

  /** Copies the layer a stroke paints into the stroke's "before" target. */
  keepBefore(layer) {
    this.renderer.copyTextureToTexture(this.layer(layer).texture, this.before.texture);
  }

  /** The depth and UV picture of the model as the camera sees it now, into `snap` (screen sized). */
  snapshot(snap) {
    this.pivot.updateWorldMatrix(true, true);
    follow(this.snapScene);
    this.pass(snap, this.snapScene, true);
  }

  /**
   * A stroke begins, seen through `viewProj` on a `width` x `height` px screen: its footprint there is empty, and
   * each triangle's place on that screen is known - grown into the gutter, which holds the triangle itself.
   */
  aim(viewProj, width, height) {
    this.footprint = createFootprint(width, height);
    const m = new THREE.Matrix4();
    for (const part of this.corners) {
      m.multiplyMatrices(viewProj, part.mesh.matrixWorld);
      part.screen = screenCorners(part.grown, m.elements, width, height, part.screen);
      part.cells = this.footprint.cells(part.screen, part.cells);
    }
  }

  /**
   * Paints the stroke's mask, which `dabs` were just added to, into the tool's layer: of the triangles under the
   * stroke so far the gutter around the islands first, then the islands (then the composite, when there is an
   * overlay).
   */
  paintMask(tool, layer, dabs) {
    const [gutter, exact] = this.materials[tool];
    this.footprint.add(dabs);
    for (const { cells, under, around, inside } of this.corners) {
      const count = this.footprint.under(cells, under.array);
      under.clearUpdateRanges();
      if (count) {
        under.addUpdateRange(0, count);
        under.needsUpdate = true;
      }
      around.setDrawRange(0, count);
      inside.setDrawRange(0, count);
    }
    follow(this.paintScene);
    this.paintScene.children.forEach((m) => { m.material = m.userData.inside ? exact : gutter; });
    this.pass(this.layer(layer), this.paintScene);
    this.composite(true);
    this.viewer.redraw();
  }

  /** What a stroke of `layer` may have changed (the tiles under it), as it was before it -> an undo step, or null. */
  undoStep(viewProj, box, screenW, screenH, layer) {
    const size = this.size;
    const m = new THREE.Matrix4();
    const flags = new Uint8Array(TILES * TILES);
    for (const part of this.corners) {
      m.multiplyMatrices(viewProj, part.mesh.matrixWorld);
      part.screen = screenCorners(part.position, m.elements, screenW, screenH, part.screen);
      const these = dirtyTiles({ uv: part.uv, screen: part.screen, box, size, tiles: TILES, margin: this.margin, twoSided: this.twoSided });
      for (let i = 0; i < flags.length; i++) flags[i] |= these[i];
    }
    const rects = tileRects(flags, TILES, size);
    if (!rects.length) return null;
    let bytes = 0;
    for (const rect of rects) {
      rect.data = this.read(this.before, rect);
      bytes += rect.data.length;
      this.touch(layer, rect);
    }
    return { rects, bytes, at: undoStamp(), layer };
  }

  read(rt, { x, y, w, h }) {
    const data = new Uint8Array(w * h * 4);
    this.renderer.readRenderTargetPixels(rt, x, y, w, h, data);
    return data;
  }

  /** Marks the tiles under `rect` (texels) as changed in `layer`. */
  touch(layer, { x, y, w, h }) {
    const tile = this.size / TILES;
    const flags = this.touched[layer];
    for (let ty = Math.floor(y / tile); ty < Math.ceil((y + h) / tile); ty++) {
      for (let tx = Math.floor(x / tile); tx < Math.ceil((x + w) / tile); tx++) flags[ty * TILES + tx] = 1;
    }
  }

  /** Writes tiles into a layer; with `swap`, each then holds what the layer had there. */
  write(rects, swap = false, layer = BASE) {
    const rt = this.layer(layer);
    const texture = rt.texture;
    const mips = texture.generateMipmaps;
    texture.generateMipmaps = false;           // once for all the tiles, below
    try {
      for (const rect of rects) {
        const now = swap ? this.read(rt, rect) : rect.data;
        const tile = new THREE.DataTexture(rect.data, rect.w, rect.h, THREE.RGBAFormat, THREE.UnsignedByteType);
        this.renderer.copyTextureToTexture(tile, texture, null, new THREE.Vector3(rect.x, rect.y, 0));
        tile.dispose();
        rect.data = now;
        this.touch(layer, rect);
      }
    } finally {
      texture.generateMipmaps = mips;
    }
    if (mips) this.pass(rt, new THREE.Scene());   // three.js makes the mip levels again after a pass into the target
    if (layer === BASE) this.version++;
    this.composite();
    this.viewer.redraw();
  }

  /** Undo and redo are the same swap of a step's tiles with its layer's (none while the texture is not intact). */
  undo() {
    if (!this.intact()) return false;
    const step = this.history.undo();
    if (step) {
      this.write(step.rects, true, layerOf(step));
      this.tally(step, true);
    }
    return !!step;
  }

  redo() {
    if (!this.intact()) return false;
    const step = this.history.redo();
    if (step) {
      this.write(step.rects, true, layerOf(step));
      this.tally(step);
    }
    return !!step;
  }

  /** A stroke, or a wipe, done on the model joins the history. */
  record(step) {
    this.history.add(step);
    this.tally(step);
  }

  /**
   * A step went onto its layer (`back`: was taken back): the strokes the layer holds since it was last wiped. Kept
   * apart from the history, which forgets its oldest steps while their paint stays.
   */
  tally(step, back = false) {
    const layer = layerOf(step);
    if (step.wipe) this.strokes[layer] = back ? step.wiped : 0;
    else this.strokes[layer] += back ? -1 : 1;
  }

  /** Strokes of `layer` are on the model, not taken back, since it was last wiped. */
  painted(layer = BASE) {
    return this.strokes[layer] > 0;
  }

  /**
   * Wipes `layer` wherever it was painted, as one step of the history: Ctrl+Z brings the painting back -> whether
   * there was anything to wipe.
   */
  wipe(layer = OVERLAY) {
    if (!this.intact() || !this.painted(layer)) return false;
    const rects = tileRects(this.touched[layer], TILES, this.size);
    let bytes = 0;
    for (const rect of rects) {
      rect.data = new Uint8Array(rect.w * rect.h * 4);
      bytes += rect.data.length;
    }
    this.write(rects, true, layer);            // each tile now holds what the layer had there
    this.record({ rects, bytes, at: undoStamp(), layer, wipe: true, wiped: this.strokes[layer] });
    return true;
  }

  /**
   * The painting on this model, for when the same model is loaded again (put): the history's steps and each layer
   * as it is wherever it was changed, of `layer` alone when given (OVERLAY: the base comes back from the files).
   * Null while the texture is not intact: what the card holds then is not the painting.
   */
  keep(layer = null) {
    if (!this.intact()) return null;
    const layers = layer ? [layer] : [BASE, OVERLAY];
    const tiles = [];
    for (const name of layers) {
      const rt = this.layer(name);
      if (!rt) continue;
      for (const rect of tileRects(this.touched[name], TILES, this.size)) {
        tiles.push({ ...rect, layer: name, data: this.read(rt, rect) });
      }
    }
    const mine = (step) => layers.includes(layerOf(step));
    const { steps, undone } = this.history.all;
    return {
      size: this.size, colour: this.overlayHex, tiles, steps: steps.filter(mine), undone: undone.filter(mine),
      strokes: Object.fromEntries(layers.map((name) => [name, this.strokes[name]])),
    };
  }

  /** A painting kept of this model (keep) onto it as loaded, before any stroke: layers and history -> whether it went on. */
  put(kept) {
    if (kept?.size !== this.size || !this.intact() || this.history.canUndo || this.history.canRedo) return false;
    const { tiles, steps, undone } = kept;
    if ([...tiles, ...steps, ...undone].some((one) => layerOf(one) === OVERLAY)) this.useOverlay(kept.colour);
    for (const layer of [BASE, OVERLAY]) {
      const these = tiles.filter((tile) => layerOf(tile) === layer);
      if (these.length) this.write(these, false, layer);
    }
    this.history.take(kept);
    Object.assign(this.strokes, kept.strokes);
    return true;
  }

  /**
   * The whole base colour texture as RGBA8 in the PNG's row order (never the overlay) -> {width, height, data, version,
   * fills}, and in a session for patches `patch`: what of it differs from the files, as patchOf gives it (null: none).
   * Refused (lostError) when the texture is not intact, before the read or after it: pixels read across a context
   * loss are zeros.
   */
  async readAll() {
    const { size, version, fills, files } = this;      // (the card reads the texture as it is at this call)
    const image = { width: size, height: size, data: await this.readWhole(this.paint), version, fills };
    if (files) image.patch = patchOf(image.data, await files);
    return image;
  }

  /**
   * The overlay's coverage, a byte per texel in the PNG's row order (0 none, 255 whole) -> {width, height, data}: from
   * 128 on a texel is covered at least half way (Omni-Texture: masked). Zeros while there is no overlay; refused as
   * readAll is.
   */
  async readOverlay() {
    const size = this.size;
    const data = new Uint8Array(size * size);
    if (!this.intact()) throw lostError();
    if (this.overlay) {
      const rgba = await this.readWhole(this.overlay);
      for (let t = 0; t < data.length; t++) data[t] = rgba[t * 4];
    }
    return { width: size, height: size, data };
  }

  /**
   * The model as the camera sees it now, `width` x `height` pixels (the viewport's shape) -> {width, height, model,
   * cover}: a byte per pixel each, rows from the top - 1 where the model is, and how far the overlay covers it there
   * (0 .. 255; 0 without an overlay). Refused as readAll is.
   */
  async sight(width, height) {
    if (!this.intact()) throw lostError();
    if (this.sightTarget?.width !== width || this.sightTarget?.height !== height) {
      this.sightTarget?.dispose();
      this.sightTarget = target(width, height, { depthBuffer: true, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    }
    if (!this.sightScene) {
      this.sightScene = new THREE.Scene();
      const materials = new Map();
      for (const { mesh, own } of this.parts) {
        if (!materials.has(own.side)) {
          materials.set(own.side, new THREE.ShaderMaterial({
            uniforms: { overlay: { value: null } }, vertexShader: SIGHT_VERTEX, fragmentShader: SIGHT_FRAGMENT, side: own.side,
          }));
        }
        this.sightScene.add(follower(mesh.geometry, materials.get(own.side), mesh));
      }
      this.sightMaterials = [...materials.values()];
    }
    for (const m of this.sightMaterials) m.uniforms.overlay.value = this.overlay?.texture || null;
    this.pivot.updateWorldMatrix(true, true);
    follow(this.sightScene);
    this.pass(this.sightTarget, this.sightScene, true);
    const rgba = new Uint8Array(width * height * 4);
    const losses = this.losses;
    const release = this.hold();
    try {
      await this.renderer.readRenderTargetPixelsAsync(this.sightTarget, 0, 0, width, height, rgba);
    } finally {
      release();
    }
    if (this.losses !== losses || !this.intact()) throw lostError();
    const model = new Uint8Array(width * height);
    const cover = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0, from = (height - 1 - y) * width, to = y * width; x < width; x++) {   // (GL's rows from the bottom)
        model[to + x] = rgba[(from + x) * 4 + 1] ? 1 : 0;
        cover[to + x] = rgba[(from + x) * 4];
      }
    }
    return { width, height, model, cover };
  }

  /** All of a layer's target as RGBA8 in the PNG's row order (readAll). */
  async readWhole(rt) {
    if (!this.intact()) throw lostError();
    const { size, losses } = this;
    const data = new Uint8Array(size * size * 4);
    const release = this.hold();
    let failed = null;
    try {
      await this.renderer.readRenderTargetPixelsAsync(rt, 0, 0, size, size, data);
    } catch (e) {
      failed = e instanceof Error ? e : new Error('The painting could not be read back from the graphics card.');
    } finally {
      if (this.losses !== losses || !this.intact()) failed = lostError();
      release();
    }
    if (failed) throw failed;
    if (this.flipRows) {
      const row = size * 4;
      const tmp = new Uint8Array(row);
      for (let a = 0, b = size - 1; a < b; a++, b--) {
        tmp.set(data.subarray(a * row, a * row + row));
        data.copyWithin(a * row, b * row, b * row + row);
        data.set(tmp, b * row);
      }
    }
    return data;
  }

  /**
   * The pixels `image` (readAll's) are in the bake's files now: what a context loss brings back. When the texture
   * was filled again since they were read (a loss meanwhile), it is filled with them: the model shows the files.
   */
  savedAs(image) {
    this.stored = { data: image.data };       // (its patch, if it had one, is not kept)
    if (image.fills === this.fills) {
      this.saved = image.version;
      if (this.patches) this.files = Promise.resolve(image.data);
    } else if (!this.disposed && this.intact()) {
      this.fill();
    }
  }

  dispose() {
    this.disposed = true;
    this.history.clear();
    if (!this.holds) this.free();
  }

  free() {
    if (this.freed) return;
    this.freed = true;
    this.canvas.removeEventListener('webglcontextlost', this.onLost);
    this.canvas.removeEventListener('webglcontextrestored', this.onRestored);
    this.stored = null;
    for (const rt of [this.paint, this.before, this.islands, this.overlay, this.shown, this.sightTarget]) rt?.dispose();
    for (const list of Object.values(this.materials)) list.forEach((m) => m.dispose());
    this.compositeMaterial?.dispose();
    this.strokeComposite?.dispose();
    this.islandMaterial.dispose();
    this.snapMaterials.forEach((m) => m.dispose());
    this.sightMaterials?.forEach((m) => m.dispose());
    for (const { islands, inside, around } of this.corners) for (const geometry of [islands, inside, around]) geometry.dispose();
    this.quad.geometry.dispose();
  }
}

/**
 * The painter of a model viewer: createPainter(viewer, ui) takes the viewer's left button while a tool is up.
 * ui: {tool() -> 'marker' | 'clone' | 'overlay' | null, size() (brush diameter, CSS px), colour() ('#rrggbb': the
 * marker's, or the overlay's while that tool is up), bySize(), byFade(), pressure(e) -> 0..1 | null, strokeStart(),
 * strokeEnd(), changed(layer) (a stroke of 'base' - something to save - or of 'overlay'), source() (the clone stamp's
 * source was set or moved), picked('#rrggbb') (the eyedropper picked a colour), and optionally lost() (the graphics
 * card was reset under a painted model, or a press found it gone: painting waits), restored() (it is back after
 * lost(): the model shows the painting saved last, without an overlay) and patches (true: the sessions it makes are
 * saved as patches, readAll's `patch`)}.
 */
export function createPainter(viewer, ui) {
  const renderer = viewer.renderer;
  const canvas = renderer.domElement;
  const floatOk = renderer.extensions.has('EXT_color_buffer_float');
  let snap = null;            // screen-sized: the depth / UV picture (float) ...
  let mask = null;            // ... and the stroke's coverage
  let stroke = null;          // the stroke being painted
  let sourceWorld = null;     // the clone stamp's source on the model, until the first stroke after picking it
  let cloneOffset = null;     // then: from the brush to the source, in screen pixels (kept from stroke to stroke)
  let eyedrop = null;         // {id, x, y}: an Alt + press with the marker, a click if it is let go where it began
  let why = '';

  // The dabs: one quad per dab, drawn with MAX blending into the mask.
  const dabGeometry = new THREE.InstancedBufferGeometry();
  dabGeometry.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0], 3));
  let dabArray = new Float32Array(4 * 256);
  let dabAttribute = new THREE.InstancedBufferAttribute(dabArray, 4);
  dabGeometry.setAttribute('dab', dabAttribute);
  const dabMaterial = new THREE.ShaderMaterial({
    uniforms: { screenSize: { value: new THREE.Vector2(1, 1) }, hardness: { value: 1 } },
    vertexShader: DAB_VERTEX, fragmentShader: DAB_FRAGMENT, depthTest: false, depthWrite: false,
    blending: THREE.CustomBlending, blendEquation: THREE.MaxEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
  });
  const dabMesh = new THREE.Mesh(dabGeometry, dabMaterial);
  dabMesh.frustumCulled = false;
  const dabScene = new THREE.Scene().add(dabMesh);

  /** The session of the model on screen (made on first use), or null with `why`. */
  function session(make = true) {
    const pivot = viewer.current;
    if (!pivot) {
      why = 'No model is shown.';
      return null;
    }
    if (pivot.userData.paint) return pivot.userData.paint;
    const found = paintableParts(pivot.userData.views);
    const lost = renderer.getContext().isContextLost();    // a texture filled now would hold nothing
    why = found.why || (lost ? LOST_WHY : '');
    if (!found.map || !make || lost) return null;
    pivot.userData.paint = new Session(viewer, pivot, found, { patches: !!ui.patches });
    return pivot.userData.paint;
  }

  function screenTargets() {
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    if (!snap || snap.width !== size.x || snap.height !== size.y) {
      snap?.dispose();
      mask?.dispose();
      snap = target(size.x, size.y, {
        type: floatOk ? THREE.FloatType : THREE.HalfFloatType, depthBuffer: true,
        minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
      });
      mask = target(size.x, size.y, { format: THREE.RedFormat });
    }
    return size;
  }

  /** A pointer event's place in the drawing buffer's pixels (y down), and the buffer's pixels per CSS pixel. */
  function bufferPoint(e) {
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / (rect.width || 1);
    const sy = canvas.height / (rect.height || 1);
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy, scale: sx };
  }

  /** The model's surface under a pointer event (three.js's intersection: point, uv...), or null. */
  function hitAt(e) {
    const s = session(false) || null;
    const pivot = viewer.current;
    if (!pivot) return null;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    const { origin, direction, far } = lineOfSight(viewer.camera, x, y);
    const ray = new THREE.Raycaster(origin, direction, 0, far);
    ray.camera = viewer.camera;
    const meshes = (s ? s.parts : paintableParts(pivot.userData.views).parts || []).map((p) => p.mesh);
    return ray.intersectObjects(meshes, false)[0] || null;
  }

  /** The eyedropper: the texture's colour around the point under `e` (3 x 3 texels) -> ui.picked('#rrggbb'). */
  function pickColour(e) {
    const hit = hitAt(e);
    const s = hit?.uv && session();
    if (!s?.intact()) return;
    const at = (t) => Math.min(s.size - 3, Math.max(0, Math.floor(Math.min(1, Math.max(0, t)) * s.size) - 1));
    const hex = texelHex(s.read(s.paint, { x: at(hit.uv.x), y: at(hit.uv.y), w: 3, h: 3 }));
    if (hex) ui.picked(hex);
  }

  function onUp(e) {
    const press = eyedrop;
    if (!press || e.pointerId !== press.id) return;
    eyedrop = null;
    if (Math.hypot(e.clientX - press.x, e.clientY - press.y) < CLICK_PX) pickColour(e);
  }

  /** Where a world point is on screen, in buffer pixels (y down). */
  function toBuffer(point) {
    const v = point.clone().project(viewer.camera);
    return { x: (v.x * 0.5 + 0.5) * canvas.width, y: (0.5 - v.y * 0.5) * canvas.height, behind: v.z > 1 };
  }

  function drawDabs(dabs) {
    if (!dabs.length || !stroke) return;
    if (dabs.length * 4 > dabArray.length) {
      dabArray = new Float32Array(Math.max(dabs.length * 4, dabArray.length * 2));
      dabAttribute = new THREE.InstancedBufferAttribute(dabArray, 4);
      dabGeometry.setAttribute('dab', dabAttribute);
    }
    const h = stroke.size.y;
    dabs.forEach((d, i) => {
      dabArray[i * 4] = d.x;
      dabArray[i * 4 + 1] = h - d.y;
      dabArray[i * 4 + 2] = d.r;
      dabArray[i * 4 + 3] = d.a;
    });
    dabsBox(dabs, stroke.box);
    dabAttribute.needsUpdate = true;
    dabGeometry.instanceCount = dabs.length;
    stroke.session.pass(mask, dabScene);
    stroke.session.paintMask(stroke.tool, stroke.layer, dabs);
    stroke.painted = true;
  }

  function begin(e) {
    const s = session();
    if (s ? !s.intact() : renderer.getContext().isContextLost()) {
      tell();             // no stroke onto a texture the card no longer holds
      return true;
    }
    if (!s) return false;
    const tool = ui.tool();
    const layer = tool === OVERLAY ? OVERLAY : BASE;
    const at = bufferPoint(e);
    if (tool === 'clone') {
      if (!cloneOffset && sourceWorld) {
        const src = toBuffer(sourceWorld);
        cloneOffset = { x: src.x - at.x, y: src.y - at.y };
        sourceWorld = null;
      }
      if (!cloneOffset) {
        ui.source();      // the helper says to pick a source first
        return true;
      }
    }
    // The camera holds still for the stroke: what is left of an orbit's glide goes.
    viewer.controls._sphericalDelta?.set(0, 0, 0);
    viewer.controls._panOffset?.set(0, 0, 0);
    viewer.controls.update();
    viewer.camera.updateMatrixWorld();
    const size = screenTargets();
    if (layer === OVERLAY) s.useOverlay(ui.colour());
    s.snapshot(snap);
    s.keepBefore(layer);
    s.pass(mask, new THREE.Scene(), true);     // the snapshot left the clear colour transparent black
    const viewProj = new THREE.Matrix4().multiplyMatrices(viewer.camera.projectionMatrix, viewer.camera.matrixWorldInverse);
    s.aim(viewProj, size.x, size.y);
    const u = s.uniforms;
    u.viewProj.value.copy(viewProj);
    u.fromClip.value.copy(viewProj).invert();
    depthRow(viewer.camera, u.depthOf.value);
    u.screenSize.value.set(size.x, size.y);
    u.mask.value = mask.texture;
    u.snap.value = snap.texture;
    rawColour(u.colour.value, ui.colour());
    u.cloneOffset.value.set(cloneOffset?.x || 0, -(cloneOffset?.y || 0));
    dabMaterial.uniforms.screenSize.value.set(size.x, size.y);
    dabMaterial.uniforms.hardness.value = HARDNESS[tool];
    ui.strokeStart();
    stroke = {
      id: e.pointerId, session: s, tool, layer, viewProj, size, painted: false,
      box: { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity },
      path: createStroke({ radius: (ui.size() / 2) * at.scale, bySize: ui.bySize(), byFade: ui.byFade() }),
    };
    try { canvas.setPointerCapture(e.pointerId); } catch { /* the pointer is gone already */ }
    drawDabs(stroke.path.add(at.x, at.y, ui.pressure(e), e.timeStamp));
    return true;
  }

  function move(e) {
    if (!stroke || e.pointerId !== stroke.id) return;
    const events = e.getCoalescedEvents?.() || [];
    const dabs = [];
    for (const one of events.length ? events : [e]) {
      const at = bufferPoint(one);
      dabs.push(...stroke.path.add(at.x, at.y, ui.pressure(one), one.timeStamp));
    }
    drawDabs(dabs);
  }

  function finish(e) {
    if (!stroke || (e && e.pointerId !== stroke.id)) return;
    drawDabs(stroke.path.end());
    const done = stroke;
    stroke = null;
    ui.strokeEnd();
    if (!done.painted || done.session.disposed || !done.session.intact()) return;
    const { x, y } = done.size;
    const step = done.session.undoStep(done.viewProj, done.box, x, y, done.layer);
    if (!step) return;                       // it touched no triangle: nothing changed
    done.session.record(step);
    if (done.layer === BASE) done.session.version++;
    ui.changed(done.layer);
  }

  // Wheel zoom would move the camera under a stroke; a second button must not start an orbit in the middle of one.
  const onWheel = (e) => {
    if (!stroke) return;
    e.preventDefault();
    e.stopImmediatePropagation();
  };
  // A lost WebGL context takes the texture with it (Session): a stroke under way ends, and the artist is told when a
  // painted model is on screen or a press finds the card gone - and again once it is back.
  let told = false;
  function tell() {
    told = true;
    ui.lost?.();
  }
  const onLost = () => {
    finish(null);
    if (session(false)) tell();
  };
  const onRestored = () => {
    session(false)?.restore();     // (before the artist hears of it, whichever listener the page calls first)
    if (!told) return;
    told = false;
    ui.restored?.();
  };
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', finish);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', finish);
  canvas.addEventListener('lostpointercapture', finish);
  canvas.addEventListener('webglcontextlost', onLost);
  canvas.addEventListener('webglcontextrestored', onRestored);
  document.addEventListener('wheel', onWheel, { capture: true, passive: false });

  viewer.paintHook = {
    /** A press on the viewer: true when the painter takes it (the viewer's navigation then never sees it). */
    down(e) {
      if (stroke) return true;
      const tool = ui.tool();
      if (!tool || e.pointerType === 'touch' || e.button !== 0 || e.shiftKey) return false;
      if (e.altKey) {                  // the navigation orbits; let go without moving, it was the eyedropper
        eyedrop = tool === 'marker' ? { id: e.pointerId, x: e.clientX, y: e.clientY } : null;
        return false;
      }
      if (e.ctrlKey || e.metaKey) {
        if (tool !== 'clone') return false;              // Ctrl + drag zooms, as everywhere in the viewer
        const point = hitAt(e)?.point.clone() || null;
        if (point) {
          sourceWorld = point;
          cloneOffset = null;
        }
        ui.source(!!point);
        return true;
      }
      return begin(e);
    },
  };

  return {
    /** Why the model on screen cannot be painted ('' when it can, or when nothing was tried yet). */
    get why() { session(false); return why; },
    /** The model on screen can be painted (without making its session). */
    paintable() {
      const pivot = viewer.current;
      return !!pivot && (!!pivot.userData.paint || !!paintableParts(pivot.userData.views).map);
    },
    /** The session of the model on screen, made now if needed (null when it cannot be painted). */
    session: (make = true) => session(make),
    painting: () => !!stroke,
    hasSource: () => !!(sourceWorld || cloneOffset),
    /** Where the clone stamp copies from for a pointer at (clientX, clientY), in client pixels, or null. */
    sourceAt(clientX, clientY) {
      const rect = canvas.getBoundingClientRect();
      const sx = canvas.width / (rect.width || 1);
      if (cloneOffset) return { x: clientX + cloneOffset.x / sx, y: clientY + cloneOffset.y / sx };
      if (sourceWorld) {
        const b = toBuffer(sourceWorld);
        return b.behind ? null : { x: rect.left + b.x / sx, y: rect.top + b.y / sx };
      }
      return null;
    },
    dropSource() {
      sourceWorld = null;
      cloneOffset = null;
    },
    cancel: () => finish(null),
    detach() {
      finish(null);
      if (viewer.paintHook) viewer.paintHook = null;
      canvas.removeEventListener('pointermove', move);
      canvas.removeEventListener('pointerup', finish);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointercancel', finish);
      canvas.removeEventListener('lostpointercapture', finish);
      canvas.removeEventListener('webglcontextlost', onLost);
      canvas.removeEventListener('webglcontextrestored', onRestored);
      document.removeEventListener('wheel', onWheel, { capture: true });
      snap?.dispose();
      mask?.dispose();
      snap = mask = null;
      dabGeometry.dispose();
      dabMaterial.dispose();
    },
  };
}
