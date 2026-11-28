// One shared three.js viewer, moved into whichever stage view is showing a model (browsers limit the
// number of WebGL contexts, and views are rebuilt on every navigation). Loaded lazily with import().
//
// A GLB can be looked at in several views (setView): "all" is the lit render with every map, "none" a
// plain grey surface, and one view per texture map (base colour, normal, roughness...) that shows the
// map's own pixels on the model, unlit, exactly as the file stores them. Over any of them, setWireframe
// draws the model's triangle edges (an OBJ's own polygon edges: the clean mesh's quad flow).
//
// Navigation is the one of the Instant Meshes viewer in Retopo (see dragAction), so the hands learn it
// once for every stage. renderThumbnail() draws a model offscreen, in the studio light (the lights going
// round the model in "all" stay in the viewer), for the props' snapshots.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { isTyping } from './dom.js';
import { fullMotion } from './shell/anim.js';
import { plainEdges } from './shell/graphics.js';
import { store } from './store.js';

const CACHE_SIZE = 2;   // high-poly GLBs are big; keep only the last two in memory
const LAYERS_KEPT = 4;  // ... and of the pictures shown alone on a model (setLayer), the last few
// A model is downloaded only once its prop has stayed on screen this long (ms): going through the props
// with the arrow keys must not fetch and parse every model on the way.
const LOAD_DELAY = 200;
// How deep an inverse lens shows the model, of the way from the camera's own plane to where its lines of sight
// meet: nearer to that point a surface would fill the picture many times over. Omni-Texture's worker looks no
// deeper either (launchers/omnitexture_views.py LENS_REACH; a test holds the two together).
export const LENS_REACH = 0.875;

const { ROTATE, DOLLY, PAN } = THREE.MOUSE;
const BUTTONS = ['LEFT', 'MIDDLE', 'RIGHT'];

/**
 * What a drag does, by the button pressed and the keys held at the press: the navigation artists
 * already have in their hands from Blender, 3ds Max, Maya and Unity, identical to the Instant Meshes
 * viewer's (there the left button is the brush, and only Alt + left drag orbits).
 *   middle             orbit (Blender)          Alt + middle       orbit (3ds Max)
 *   Shift + middle     pan (Blender)            Ctrl + middle      zoom (Blender; Ctrl + Alt: 3ds Max)
 *   left               orbit, like the middle button, with the same Shift / Ctrl / Alt
 *   right              pan                      Alt + right        zoom (Maya, Unity)
 * The wheel zooms toward the cursor.
 */
export function dragAction(button, { altKey = false, shiftKey = false, ctrlKey = false, metaKey = false } = {}) {
  if (button === 2) return altKey ? DOLLY : PAN;
  if (button !== 0 && button !== 1) return null;
  if (ctrlKey || metaKey) return DOLLY;
  return shiftKey ? PAN : ROTATE;
}

/**
 * The per-map views, in button order, and where each reads a glTF material (as GLTFLoader builds it):
 * the texture property, the channel(s) shown, and what a material without that texture uses instead
 * (shown as a flat colour, so every part of the model says what the renderer uses there).
 * `opaqueToo: false`: the view exists only for materials that are not OPAQUE (glTF alphaMode).
 */
const MAP_VIEWS = {
  basecolor: { texture: 'map', swizzle: 'texel.rgb', constant: (m) => m.color.clone().convertLinearToSRGB() },
  opacity: { texture: 'map', swizzle: 'vec3( texel.a )', opaqueToo: false, constant: (m) => grey(m.opacity) },
  normal: { texture: 'normalMap', swizzle: 'texel.rgb', constant: () => new THREE.Color(0.5, 0.5, 1) },
  roughness: { texture: 'roughnessMap', swizzle: 'vec3( texel.g )', constant: (m) => grey(m.roughness) },
  metallic: { texture: 'metalnessMap', swizzle: 'vec3( texel.b )', constant: (m) => grey(m.metalness) },
  ao: { texture: 'aoMap', swizzle: 'vec3( texel.r )', constant: () => grey(1) },
  emissive: {
    texture: 'emissiveMap', swizzle: 'texel.rgb',
    constant: (m) => m.emissive.clone().multiplyScalar(m.emissiveIntensity).convertLinearToSRGB(),
  },
  // glTF has no height slot: a height map comes from outside the GLB (show's `extraMaps`).
  height: { texture: null, swizzle: 'vec3( texel.r )', constant: () => grey(0) },
};
const MAP_MODES = Object.keys(MAP_VIEWS);

const grey = (v) => new THREE.Color(v, v, v);

// A mid-grey, matte clay: a light one under the scene's lights and environment reads as a flat white
// silhouette, so the shape was hard to judge. "None" uses it a quarter lighter (with the model's own normals):
// under the studio's two lights alone the mid-grey read too dark there. The same in every stage.
// The floor of a model's saved files, drawn while its Move / Rotate handles are up (setGizmo): a grid at their
// origin, at the height 0, in the files' own units (a model is about 1 across) with cells a tenth of one, and the
// origin's three axes.
const FLOOR_SIZE = 4;
const FLOOR_CELL = 0.1;
const FLOOR_AXIS = 0x8b7dff;
const FLOOR_LINE = 0x4a5263;
const CLAY = { color: 0x7e8591, roughness: 0.92, metalness: 0, envMapIntensity: 0.4 };
const NONE_CLAY = { ...CLAY, color: 0x9ea6b5 };

// The triangle edges over a model (setWireframe): the dark lines of the retopo's wire, fainter, since a
// 3D model's triangles are small and many (and each edge is drawn by both triangles that share it).
const TRI_WIRE = { color: 0x12151c, opacity: 0.4 };

// Per-fragment MikkTSpace (Blender, Unreal, Unity HDRP, Substance): the baker encodes its normal maps
// for the decoder of mikktspace.h, n = normalize(x T + y B + z N) with B = w cross(N, T) built per
// fragment from the UNnormalised interpolated T and N. three.js builds B per vertex and normalises all
// three per fragment, which is a different basis wherever the vertex tangents differ across a
// triangle. A baked GLB says which basis it uses (material extras.spellcast.tangent_basis); for
// "mikk" this patch swaps three's tangent frame for the Mikk one. The line replaced is the USE_TANGENT
// frame of r186's normal_fragment_begin; under DOUBLE_SIDED that chunk then flips T and B on back
// faces itself, so N gets faceDirection here. FLIP_SIDED (BackSide) negated vNormal in the vertex
// shader, and three negates normalScale there: w is negated too, so B keeps its front-face direction.
const THREE_TBN = 'mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );';
const MIKK_TBN = `#ifdef DOUBLE_SIDED
		mat3 tbn = mat3( vTangent, vMikkW * cross( vNormal, vTangent ), vNormal * faceDirection );
	#else
		mat3 tbn = mat3( vTangent, vMikkW * cross( vNormal, vTangent ), vNormal );
	#endif`;
const MIKK_VARYING = '#ifdef USE_TANGENT\n\tvarying float vMikkW;\n#endif\n';
const MIKK_VERTEX = `#include <normal_vertex>
#ifdef USE_TANGENT
	vMikkW = tangent.w;
	#ifdef FLIP_SIDED
		vMikkW = - vMikkW;
	#endif
#endif`;
const MIKK_NORMAL_BEGIN = THREE.ShaderChunk.normal_fragment_begin.includes(THREE_TBN)
  ? THREE.ShaderChunk.normal_fragment_begin.replace(THREE_TBN, MIKK_TBN) : null;
if (!MIKK_NORMAL_BEGIN) console.error('viewer3d: this three.js has no tangent frame to patch; baked normal maps show in three.js\'s own basis.');

/** Makes `material` decode its tangent-space normal map in the per-fragment MikkTSpace basis. */
export function patchMikk(material) {
  if (!MIKK_NORMAL_BEGIN) return;
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = MIKK_VARYING + shader.vertexShader.replace('#include <normal_vertex>', MIKK_VERTEX);
    shader.fragmentShader = MIKK_VARYING
      + shader.fragmentShader.replace('#include <normal_fragment_begin>', MIKK_NORMAL_BEGIN);
  };
  material.customProgramCacheKey = () => 'spellcast-mikk-tbn';
  material.needsUpdate = true;
}

const MAP_VERTEX = `
varying vec2 vMapUv;
uniform mat3 mapTransform;
void main() {
	vMapUv = ( mapTransform * vec3( MAP_UV, 1.0 ) ).xy;
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`;
const MAP_FRAGMENT = `
varying vec2 vMapUv;
#ifdef USE_MAP
	uniform sampler2D map;
#else
	uniform vec3 value;
#endif
void main() {
#ifdef USE_MAP
	vec4 texel = texture2D( map, vMapUv );
	#ifdef MAP_SRGB
		texel = sRGBTransferOETF( texel );   // the GPU decoded the sRGB file to linear: back to the file's values
	#endif
	gl_FragColor = vec4( SWIZZLE, 1.0 );
#else
	gl_FragColor = vec4( value, 1.0 );
#endif
}`;

/**
 * An unlit view of one map of `source`: the texture's own values straight to the screen (the canvas is
 * sRGB, so a value of 128 in the file shows as 128), with no light, tone mapping or colour conversion.
 */
function mapMaterial(view, source, texture) {
  const params = { side: source.side, toneMapped: false, vertexShader: MAP_VERTEX, fragmentShader: MAP_FRAGMENT };
  if (!texture) {
    return new THREE.ShaderMaterial({ ...params, defines: { MAP_UV: 'uv' }, uniforms: { value: { value: view.constant(source) } } });
  }
  if (texture.matrixAutoUpdate) texture.updateMatrix();
  const uv = texture.channel ? `uv${texture.channel}` : 'uv';
  return new THREE.ShaderMaterial({
    ...params,
    // three.js declares the extra UV sets only for its own materials' maps.
    vertexShader: (texture.channel ? `attribute vec2 ${uv};\n` : '') + MAP_VERTEX,
    defines: { USE_MAP: '', MAP_UV: uv, SWIZZLE: view.swizzle, ...(texture.colorSpace === THREE.SRGBColorSpace ? { MAP_SRGB: '' } : {}) },
    uniforms: { map: { value: texture }, mapTransform: { value: texture.matrix } },
  });
}

const isOpaque = (m) => !m.transparent && !(m.alphaTest > 0);

/** The views of one loaded model: its own materials ("all") and the ones made for the other views. */
class ModelViews {
  constructor(object) {
    this.meshes = [];
    object.traverse((o) => { if (o.isMesh) this.meshes.push({ mesh: o, own: o.material }); });
    this.made = new Map();       // mode -> Map(own material -> material of that view)
    this.extraMaps = {};         // mode -> {url, colorspace, onLoad, texture?}: maps from outside the GLB
    this.wire = false;           // the triangle edges are drawn over the view (setWire)
    this.wires = null;           // one overlay per mesh, made the first time they are asked for
    // An OBJ (loadObj) brings its own lines, the file's polygon edges: they are its wireframe.
    this.edges = object.userData.edges || null;
    this.back = new WeakSet();   // the materials setBack moved back in depth
    this.layers = new Map();     // url -> a picture laid out like the base colour map (applyLayer)
    this.layerMade = new Map();  // 'all' | 'flat' | 'clay' -> Map(own material -> the material that carries a layer)
  }

  /**
   * The picture at `url`, laid out like the model's base colour map (one Omni-Texture view's share of it), as a
   * texture on the same UVs; onReady(texture) once it can be drawn (at once when it was loaded before). The
   * LAYERS_KEPT asked for last stay on the graphics card: a prop may have dozens, each as large as its texture.
   */
  layer(url, onReady) {
    let entry = this.layers.get(url);
    if (entry) {
      this.layers.delete(url);       // asked for last: the last to go
    } else {
      const like = this.meshes.flatMap(({ own }) => [own].flat()).find((m) => m.map)?.map;
      entry = { ready: false, waiting: [] };
      entry.texture = new THREE.TextureLoader().load(url, () => {
        entry.ready = true;
        entry.waiting.splice(0).forEach((done) => done(entry.texture));
      }, undefined, () => this.layers.delete(url));   // not there: the model stays as it is, a later ask tries again
      entry.texture.flipY = false;   // the GLB's own UVs are glTF's (v down)
      entry.texture.colorSpace = THREE.SRGBColorSpace;
      if (like) {                    // the base colour map's UV set, wrapping and transform
        entry.texture.channel = like.channel;
        entry.texture.wrapS = like.wrapS;
        entry.texture.wrapT = like.wrapT;
        entry.texture.offset.copy(like.offset);
        entry.texture.repeat.copy(like.repeat);
        entry.texture.center.copy(like.center);
        entry.texture.rotation = like.rotation;
      }
    }
    this.layers.set(url, entry);
    for (const [old, { texture }] of this.layers) {
      if (this.layers.size <= LAYERS_KEPT) break;
      this.layers.delete(old);
      texture.dispose();
    }
    if (entry.ready) onReady(entry.texture);
    else entry.waiting.push(onReady);
  }

  /**
   * The model with `texture` (a layer) as its colour, drawn the way the view `mode` draws the model's own base
   * colour, so a layer and the model compare like with like (lit views are tone mapped, map views are not: the same
   * colours looked richer lit than in "Base colour"): "all" - the model's own lit materials with their other maps,
   * the layer in the base colour's place; a map view - flat, the picture's own values, no light or tone mapping;
   * "none" - on the lit bare clay. apply(mode) brings the view back.
   */
  applyLayer(texture, mode) {
    const kind = mode === 'all' ? 'all' : MAP_VIEWS[mode] ? 'flat' : 'clay';
    let made = this.layerMade.get(kind);
    if (!made) this.layerMade.set(kind, made = new Map());
    const dress = (own) => {
      let m = made.get(own);
      if (!m) {
        if (kind === 'all') {
          m = own.clone();
          m.onBeforeCompile = own.onBeforeCompile;   // (a baked model's Mikk tangent frame: patchMikk)
          m.customProgramCacheKey = own.customProgramCacheKey;
        } else if (kind === 'flat') {
          m = mapMaterial(MAP_VIEWS.basecolor, own, texture);
        } else {
          m = new THREE.MeshStandardMaterial({ ...CLAY, color: 0xffffff, side: own.side, flatShading: own.flatShading });
        }
        made.set(own, m);
        this.setBack(m);
      }
      if (kind === 'flat') {
        if (texture.matrixAutoUpdate) texture.updateMatrix();
        m.uniforms.map.value = texture;
        m.uniforms.mapTransform.value = texture.matrix;
      } else {
        if (!m.map) m.needsUpdate = true;   // (a material without a colour map until now)
        m.map = texture;
        if (kind === 'all') m.aoMapIntensity = own.aoMapIntensity;   // (apply() sets it on the model's own)
      }
      return m;
    };
    for (const entry of this.meshes) entry.mesh.material = Array.isArray(entry.own) ? entry.own.map(dress) : dress(entry.own);
  }

  texture(mode, material) {
    const view = MAP_VIEWS[mode];
    if (view.opaqueToo === false && isOpaque(material)) return null;
    if (view.texture) return material[view.texture] || null;
    const extra = this.extraMaps[mode];
    if (!extra) return null;
    if (!extra.texture) {
      extra.texture = new THREE.TextureLoader().load(extra.url, extra.onLoad);
      extra.texture.flipY = false;   // it goes on the GLB's own UVs, which are glTF's (v down)
      extra.texture.colorSpace = extra.colorspace === 'srgb' ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    }
    return extra.texture;
  }

  /** ['none', 'all', ...the maps some material has], or [] for a model without texture maps. */
  available() {
    const own = this.meshes.flatMap(({ own }) => [own].flat());
    const maps = MAP_MODES.filter((mode) => (MAP_VIEWS[mode].texture ? own.some((m) => this.texture(mode, m)) : !!this.extraMaps[mode]));
    return maps.length ? ['none', 'all', ...maps] : [];
  }

  /** extra: {mode: {url, colorspace}} for maps the GLB cannot carry (height); onLoad redraws. */
  setExtraMaps(extra, onLoad) {
    for (const [mode, old] of Object.entries(this.extraMaps)) {
      if (old.url === extra[mode]?.url) continue;
      old.texture?.dispose();
      delete this.extraMaps[mode];
      this.made.get(mode)?.forEach((m) => m.dispose());
      this.made.delete(mode);
    }
    for (const [mode, map] of Object.entries(extra)) {
      if (MAP_VIEWS[mode] && !MAP_VIEWS[mode].texture && !this.extraMaps[mode]) this.extraMaps[mode] = { ...map, onLoad };
    }
  }

  apply(mode) {
    for (const entry of this.meshes) {
      entry.mesh.material = Array.isArray(entry.own) ? entry.own.map((m) => this.material(mode, m)) : this.material(mode, entry.own);
      // Deliberate: the baked AO map does not darken the "all" view. The base colour Pixal3D paints already carries
      // its own ambient occlusion, and ours on top of it was too heavy in places; the AO map is still there to look
      // at on its own (the AO button) and goes with the exported files as it is.
      for (const m of [entry.own].flat()) if (m.aoMap) m.aoMapIntensity = 0;
    }
  }

  material(mode, own) {
    if (mode === 'all') return own;
    let made = this.made.get(mode);
    if (!made) this.made.set(mode, made = new Map());
    let m = made.get(own);
    if (!m) {
      m = mode === 'none'
        ? new THREE.MeshStandardMaterial({ ...NONE_CLAY, side: own.side, flatShading: own.flatShading })
        : mapMaterial(MAP_VIEWS[mode], own, this.texture(mode, own));
      made.set(own, m);
      this.setBack(m);
    }
    return m;
  }

  /**
   * The base colour texture `from` becomes `to` on every material of the model that has it (paint3d.js: the painted
   * copy), patch(material) adapting each; the views made from the old texture are made again when next shown.
   */
  replaceMap(from, to, patch) {
    for (const { own } of this.meshes) {
      for (const m of [own].flat()) {
        if (m.map !== from) continue;
        m.map = to;
        patch?.(m);
        for (const mode of ['basecolor', 'opacity']) {
          const made = this.made.get(mode)?.get(m);
          if (!made) continue;
          made.dispose();
          this.made.get(mode).delete(m);
        }
      }
    }
  }

  /**
   * Draws the model's triangle edges over whatever view is on, or takes them away. The overlay is a
   * second mesh on the same geometry with a wireframe material: three.js builds a geometry's edge list
   * once, the first time it draws it that way, and frees it with the geometry. It is made on first use
   * and kept with the model, so switching views, or the edges off and on, builds nothing.
   */
  setWire(on) {
    this.wire = on;
    if (this.edges) {   // an OBJ: its polygon edges show or hide (its clay always sits back in depth)
      this.edges.visible = on;
      return;
    }
    if (on && !this.wires) {
      this.wireMaterial = new THREE.MeshBasicMaterial({
        ...TRI_WIRE, wireframe: true, transparent: true, depthWrite: false, toneMapped: false,
      });
      this.wires = this.meshes.map(({ mesh }) => {
        const lines = new THREE.Mesh(mesh.geometry, this.wireMaterial);
        lines.renderOrder = 1;   // after the surface it lies on, also when that one is transparent
        mesh.add(lines);
        return lines;
      });
    }
    for (const lines of this.wires || []) lines.visible = on;
    for (const { own } of this.meshes) [own].flat().forEach((m) => this.setBack(m));
    for (const made of this.made.values()) made.forEach((m) => this.setBack(m));
    for (const made of this.layerMade.values()) made.forEach((m) => this.setBack(m));
  }

  /**
   * While the edges are drawn, the surface under them sits a little back in depth (as the retopo's clay
   * always does, see loadObj), so the lines stay on top of it instead of flickering in and out of it.
   */
  setBack(material) {
    if (this.wire && !material.polygonOffset) {
      material.polygonOffset = true;
      material.polygonOffsetFactor = material.polygonOffsetUnits = 1;
      this.back.add(material);
    } else if (!this.wire && this.back.delete(material)) {
      material.polygonOffset = false;
    }
  }

  dispose() {
    this.apply('all');
    this.setWire(false);
    for (const lines of this.wires || []) lines.removeFromParent();
    this.wireMaterial?.dispose();
    for (const made of this.made.values()) made.forEach((m) => m.dispose());
    for (const extra of Object.values(this.extraMaps)) extra.texture?.dispose();
    for (const made of this.layerMade.values()) made.forEach((m) => m.dispose());
    this.layers.forEach((entry) => entry.texture.dispose());
  }
}

class Viewer {
  constructor() {
    this.container = null;
    this.scope = null;     // where the pointer or the focus must be for the F key to frame the model
    this.cache = new Map();   // url -> pivot Group (pivot.userData.views: its ModelViews)
    this.current = null;
    this.view = 'all';
    this.token = 0;
    this.loading = null;   // LoadingManager of the download in progress (aborted when superseded)
    this.dirty = true;
    this.paintHook = null; // paint3d.js while a brush is up: it takes the presses it paints with
    this.orbiting = false; // a model's "all" view is on screen: OrbitLights light it (orbitLight)
    this.holdLight = false;   // those lights hold still (a brush is up)
    this.place = null;     // where the model last on screen stood (show's keepView)
    this.lightsAt = 0;     // when the lights of "all" last moved on (ms)
    this.layerUrl = null;  // the layer shown over the view (setLayer)
    // Where the model on screen stands in its saved files (show's `placement`), and the handles that move it
    // (setGizmo): the mode on, the pane's onPlaced, three's TransformControls and the floor drawn at the origin.
    this.placed = { move: new THREE.Vector3(), turn: new THREE.Quaternion() };
    this.gizmo = { mode: null, onPlaced: null, handles: null, floor: null, origin: new THREE.Vector3() };
  }

  /** Draws the scene again at the next frame. */
  redraw() {
    this.dirty = true;
  }

  init() {
    if (this.renderer) return;
    const r = makeRenderer({ antialias: !plainEdges() });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.domElement.className = 'viewer-canvas';
    this.renderer = r;
    this.scene = studio(r);
    this.orbit = new OrbitLights(this.scene);
    this.root = new THREE.Group();
    this.scene.add(this.root);

    this.camera = new LensCamera(32, 1, 0.01, 1000);
    const controls = new OrbitControls(this.camera, r.domElement);
    this.camera.about = controls.target;
    controls.enableDamping = true;
    controls.dampingFactor = 0.09;
    controls.screenSpacePanning = true;
    controls.zoomToCursor = true;
    controls.addEventListener('change', () => { this.dirty = true; });
    this.controls = controls;
    this.resizer = new ResizeObserver(() => this.resize());
    this.installNavigation();
  }

  /**
   * The per-drag button mapping (dragAction), the F key, and no middle-click autoscroll over the canvas
   * (OrbitControls already keeps the context menu away). Listening starts in attach(), stops in detach().
   */
  installNavigation() {
    const canvas = this.renderer.domElement;
    // OrbitControls reads mouseButtons in its own pointerdown listener on the canvas, registered first:
    // a capture-phase listener on the document runs before it, whatever the order they were added in.
    this.onPointerDownCapture = (e) => {
      if (e.target !== canvas) return;
      if (this.paintHook?.down(e)) {         // a brush stroke: the navigation never sees this press
        e.stopPropagation();
        e.preventDefault();
        return;
      }
      if (e.pointerType === 'touch') return;
      const action = dragAction(e.button, e);
      if (action == null) return;
      // OrbitControls itself swaps orbit and pan while Ctrl, Cmd or Shift is held: undo that, so the
      // table in dragAction is exactly what happens.
      const swapped = e.ctrlKey || e.metaKey || e.shiftKey;
      this.controls.mouseButtons[BUTTONS[e.button]] = !swapped ? action
        : action === ROTATE ? PAN : action === PAN ? ROTATE : action;
      // Suppressing the compatibility mouse events is what stops Chrome and Firefox opening autoscroll.
      if (e.button === 1) e.preventDefault();
    };
    this.onAuxClick = (e) => { if (e.button === 1) e.preventDefault(); };
    canvas.addEventListener('auxclick', this.onAuxClick);
    // F frames the model, like Blender's "View Selected" and Unity's F: only while the pointer is over
    // the viewer or the focus is in it, and never while a field is being typed in.
    this.onKeyDown = (e) => {
      if (e.key !== 'f' && e.key !== 'F') return;
      if (e.defaultPrevented || e.repeat || e.ctrlKey || e.metaKey || e.altKey || isTyping()) return;
      if (document.querySelector('dialog[open]')) return;
      const scope = this.scope || this.container;
      if (!scope || !(scope.matches(':hover') || scope.contains(document.activeElement))) return;
      if (this.frame({ keepDirection: true })) e.preventDefault();
    };
  }

  /** Shows the viewer in `container`; scope: the element around it that owns the F key (default: container). */
  attach(container, scope = container) {
    this.init();
    if (this.container && this.container !== container) this.detach();
    this.scope = scope;
    if (this.container === container) return;
    this.container = container;
    container.prepend(this.renderer.domElement);
    this.resizer.observe(container);
    this.resize();
    document.addEventListener('pointerdown', this.onPointerDownCapture, true);
    document.addEventListener('keydown', this.onKeyDown);
    this.renderer.setAnimationLoop((now) => {
      this.controls.update();
      // The lights of "all" hold still where they are while a job works (3D, retopo...: nothing is drawn but what
      // the artist moves), while a brush is up and with "Reduced" animations, and go on from there. A hidden tab
      // draws no frames at all. Their turning alone asks for ORBIT_FPS frames a second, not for every frame the
      // screen can show: a lit frame of a large model is most of what a small graphics chip can do in that time.
      const since = now - this.lightsAt;
      if (this.orbiting && !jobWorks() && !this.holdLight && fullMotion() && since >= 1000 / ORBIT_FPS - 1) {
        this.orbit.turn((Math.min(since, ORBIT_STEP_MS) / ORBIT_MS) * Math.PI * 2);
        this.lightsAt = now;
        this.dirty = true;
      }
      if (this.dirty) {
        this.dirty = false;
        this.draw();
      }
    });
  }

  /** Draws the scene as the camera sees it now. */
  draw() {
    this.orbit.faceCamera(this.camera);
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * A model in its "all" view - the lit render with every map, or the clay of a model without maps (Retopo's clean
   * mesh) - is lit by point lights going round it from every side (OrbitLights), so the artist sees the light move
   * over its top, bottom and sides instead of baked shading; the studio's key and rim lights go dark and the
   * ambience drops. Off in the map views (unlit anyway) and in "none": the studio light, as in the thumbnails.
   * Lights are darkened, never taken out, so every view has the same lights and a switch compiles no shaders.
   */
  orbitLight(on) {
    const { key, rim, ambient } = this.scene.userData;
    this.orbiting = on;
    this.orbit.light(on);
    key.intensity = on ? 0 : KEY_LIGHT.intensity;
    rim.intensity = on ? 0 : RIM_LIGHT.intensity;
    ambient.intensity = on ? AMBIENT * ORBIT_AMBIENCE : AMBIENT;
    this.scene.environmentIntensity = on ? ORBIT_AMBIENCE : 1;
    this.dirty = true;
  }

  /** How far the lights of "all" reach, in model radii (ORBIT_REACH_WIDE for Retopo; none given: the usual). */
  setLightReach(reach = ORBIT_REACH) {
    if (this.orbit.reach === reach) return;
    this.orbit.reach = reach;
    this.orbit.power();
    this.dirty = true;
  }

  /** Detaches from `container` (only if it is still the one we are in). */
  detach(container) {
    if (!this.container || (container && container !== this.container)) return;
    this.token++;          // a show() still loading for this container resolves false
    this.stopLoading();
    this.setGizmo(null);
    this.renderer.setAnimationLoop(null);
    this.orbitLight(false);
    this.resizer.unobserve(this.container);
    document.removeEventListener('pointerdown', this.onPointerDownCapture, true);
    document.removeEventListener('keydown', this.onKeyDown);
    this.renderer.domElement.remove();
    this.container = this.scope = null;
  }

  /**
   * A still picture of `canvas` as a 2D canvas of the same size and class: a copy of the viewer's own
   * WebGL canvas (rendered once more to read it) or of any 2D canvas. A plain copy of a WebGL canvas
   * would be blank.
   */
  still(canvas) {
    if (canvas === this.renderer?.domElement && this.container) this.draw();
    const copy = document.createElement('canvas');
    copy.width = canvas.width;
    copy.height = canvas.height;
    copy.className = canvas.className;
    try {
      copy.getContext('2d').drawImage(canvas, 0, 0);
    } catch (e) {
      console.warn('No still of the preview:', e);
    }
    return copy;
  }

  resize() {
    if (!this.container) return;
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  /**
   * Shows the GLB or OBJ at `url`, in the "all" view, a GLB without its wireframe and an OBJ with its
   * polygon edges, as always (the map bar puts back what its stage shows). Resolves true once shown,
   * false if another show() superseded it. onProgress(fraction|null) reports download progress; extraMaps ({height: {url,
   * colorspace}}) adds maps a GLB cannot carry, on its own UVs. keepView: `url` is another model of the
   * object last on screen (its high or low poly, its texture changed): it takes that model's place and the
   * camera stays where the artist left it, so the two can be compared up close - and that model stays on screen
   * until this one is there, loaded at once. placement ({move, turn} | null): where the model stands in its saved
   * files (the Retopo and Export pages): it is shown turned and moved so, about the files' origin (setGizmo draws
   * it); every other page shows a model as it was made.
   */
  async show(url, kind, { onProgress, extraMaps = {}, keepView = false, placement = null } = {}) {
    this.init();
    const token = ++this.token;
    this.stopLoading();         // nobody waits for the previous model any more
    this.setGizmo(null);        // the pane that shows a model puts its handles back
    // Where frame() put the model last on screen: one shown with keepView goes to the same place, also when a
    // model still loading had taken that one off the screen meanwhile.
    if (this.current) this.place = this.current.children[0].position.clone();
    const place = keepView ? this.place : null;
    let pivot = this.cache.get(url);
    if (pivot) {
      this.cache.delete(url);   // refresh its LRU position
    } else {
      if (!keepView) {
        this.root.clear();      // don't keep showing the previous prop's model while this one loads
        this.current = null;
        this.dirty = true;
        await new Promise((resolve) => setTimeout(resolve, LOAD_DELAY));
        if (token !== this.token) return false;
      }
      const manager = new THREE.LoadingManager();
      this.loading = manager;
      let object;
      try {
        object = kind === 'obj' ? await loadObj(url, manager.abortController.signal)
          : await loadGlb(url, manager, onProgress);
      } catch (e) {
        if (token !== this.token) return false;   // aborted: a newer show() or a detach took over
        throw e;
      } finally {
        if (this.loading === manager) this.loading = null;
      }
      pivot = new THREE.Group();
      pivot.add(object);
      pivot.userData.views = new ModelViews(object);
    }
    this.cache.set(url, pivot);
    for (const [oldUrl, old] of this.cache) {
      if (this.cache.size <= CACHE_SIZE) break;
      if (old === pivot || old === this.current) continue;
      this.cache.delete(oldUrl);
      old.userData.paint?.dispose();         // a painted model's texture and passes (paint3d.js)
      old.userData.views.dispose();
      dispose(old);
    }
    if (token !== this.token) return false;
    const views = pivot.userData.views;
    views.setExtraMaps(extraMaps, () => { this.dirty = true; });
    views.apply('all');
    views.setWire(!!views.edges);
    this.view = 'all';
    this.layerUrl = null;
    this.root.clear();
    this.root.add(pivot);
    this.current = pivot;
    this.orbitLight(true);
    // Turned as placed before it is centred; a model that takes another's place stands where that one stood, which
    // holds the move already (the two are one prop's).
    this.placed.move.fromArray(placement?.move || [0, 0, 0]);
    this.placed.turn.fromArray(placement?.turn || [0, 0, 0, 1]);
    pivot.children[0].quaternion.copy(this.placed.turn);
    if (place) {
      pivot.children[0].position.copy(place);
      this.orbit.fit(pivot);
      this.dirty = true;
    } else {
      this.frame();
    }
    return true;
  }

  // ---- where the model stands in its saved files, and the handles that put it there ---------------------------

  /** The placement of the model on screen as the hub keeps it: {move: [x, y, z], turn: [x, y, z, w]}. */
  placement() {
    return { move: this.placed.move.toArray(), turn: this.placed.turn.toArray() };
  }

  /**
   * Where the saved files' origin is on screen: the model's own origin, less its move. frame() centres the model in
   * the view wherever it was placed, so the origin is worked out from the model, never kept.
   */
  fileOrigin() {
    const object = this.current?.children[0];
    return object ? object.position.clone().sub(this.placed.move) : new THREE.Vector3();
  }

  /**
   * The Move ('move') or Rotate ('turn') handles on the model on screen, with the floor of its saved files drawn
   * under it (a grid at their origin, the height 0); null takes both away. onPlaced(placement) is told after every
   * drag, and after placeOnFloor() and placeBack(). The handles are three.js's TransformControls, loaded at their
   * first use; a drag of theirs holds the camera still.
   */
  async setGizmo(mode, onPlaced = null) {
    const gizmo = this.gizmo;
    gizmo.mode = mode || null;
    gizmo.onPlaced = mode ? onPlaced : null;
    if (mode && !gizmo.handles) {
      const { TransformControls } = await import('three/addons/controls/TransformControls.js');
      if (!gizmo.handles) this.makeHandles(TransformControls);
    }
    this.syncGizmo();
  }

  makeHandles(TransformControls) {
    const gizmo = this.gizmo;
    const handles = new TransformControls(this.camera, this.renderer.domElement);
    handles.setSize(0.85);
    handles.addEventListener('change', () => { this.dirty = true; });
    handles.addEventListener('dragging-changed', (e) => {
      this.controls.enabled = !e.value;
      if (e.value) gizmo.origin.copy(this.fileOrigin());   // the origin stays where it is while the model moves
      else gizmo.onPlaced?.(this.readPlaced(gizmo.origin));
    });
    gizmo.handles = handles;
    this.scene.add(handles.getHelper());
    const floor = new THREE.Group();
    const grid = new THREE.GridHelper(FLOOR_SIZE, FLOOR_SIZE / FLOOR_CELL, FLOOR_AXIS, FLOOR_LINE);
    const axes = new THREE.AxesHelper(FLOOR_CELL * 2);
    for (const lines of [grid, axes]) {
      lines.material.transparent = true;
      lines.material.opacity = lines === grid ? 0.55 : 0.9;
      lines.material.depthWrite = false;
      lines.raycast = () => {};
    }
    floor.add(grid, axes);
    floor.visible = false;
    gizmo.floor = floor;
    this.scene.add(floor);
  }

  /** The handles and the floor as the mode and the model on screen ask (after a show, a frame, a placement). */
  syncGizmo() {
    const { mode, handles, floor } = this.gizmo;
    if (!handles) return;
    const object = mode && this.current ? this.current.children[0] : null;
    if (object) {
      handles.setMode(mode === 'turn' ? 'rotate' : 'translate');
      handles.setSpace(mode === 'turn' ? 'local' : 'world');
      if (handles.object !== object) handles.attach(object);
      floor.position.copy(this.fileOrigin());
    } else if (handles.object) {
      handles.detach();
      this.controls.enabled = true;     // a drag cut short by a new model must not leave the camera held
    }
    floor.visible = !!object;
    this.dirty = true;
  }

  /** The model was moved or turned on screen, the files' origin staying at `origin` -> its placement, read back. */
  readPlaced(origin) {
    const object = this.current.children[0];
    this.placed.move.copy(object.position).sub(origin);
    this.placed.turn.copy(object.quaternion).normalize();
    this.orbit.fit(this.current);       // the lights go round the model where it stands now
    this.syncGizmo();
    return this.placement();
  }

  /**
   * Stands the model on the floor of its saved files, their origin under the middle of it (what a game engine
   * expects of a prop or a character: dropped into a level, it stands on the ground). Its turn stays. -> the
   * placement it has now, or null without a model.
   */
  placeOnFloor() {
    const object = this.current?.children[0];
    if (!object) return null;
    const origin = this.fileOrigin();
    this.current.updateMatrixWorld(true);
    // From every vertex (precise): the box of a turned model's own box is larger than the model, and would leave it
    // hovering over the floor.
    const box = new THREE.Box3().setFromObject(object, true);
    if (box.isEmpty()) return null;
    const middle = box.getCenter(new THREE.Vector3());
    object.position.add(new THREE.Vector3(origin.x - middle.x, origin.y - box.min.y, origin.z - middle.z));
    return this.readPlaced(origin);
  }

  /** The model back as it was made, no move and no turn -> that placement, or null without a model. */
  placeBack() {
    const object = this.current?.children[0];
    if (!object) return null;
    const origin = this.fileOrigin();
    object.position.copy(origin);
    object.quaternion.identity();
    return this.readPlaced(origin);
  }

  /** The views the model on screen offers: ['none', 'all', 'basecolor', ...], or [] (no texture maps). */
  availableViews() {
    return this.current?.userData.views.available() || [];
  }

  /** Switches the model on screen to one of availableViews(). Returns false when it has no such view. */
  setView(mode) {
    if (!this.availableViews().includes(mode)) return false;
    this.current.userData.views.apply(mode);
    this.view = mode;
    this.layerUrl = null;
    this.orbitLight(mode === 'all');
    this.dirty = true;
    return true;
  }

  /**
   * Shows the picture at `url` alone on the model on screen, as its colour, drawn as the view chosen draws the
   * model's own (applyLayer: lit with the lights turning in "all", flat in a map view, on the lit clay in "none"): a
   * picture laid out like the model's base colour map, e.g. what one Omni-Texture view painted.
   * null takes it off: the model is back in the view it was in. Until the next show() or setView().
   */
  setLayer(url) {
    const views = this.current?.userData.views;
    this.layerUrl = url || null;
    if (!views) return;
    if (!url) {
      views.apply(this.view);
      this.orbitLight(this.view === 'all');
      this.dirty = true;
      return;
    }
    views.layer(url, (texture) => {
      if (this.layerUrl !== url || this.current?.userData.views !== views) return;   // the pointer moved on meanwhile
      views.applyLayer(texture, this.view);
      this.orbitLight(this.view === 'all');
      this.dirty = true;
    });
  }

  /**
   * Draws the triangle edges of the model on screen over it (or stops), whatever view it is in: they
   * stay through setView() and frame(), until the next show(). An OBJ shows or hides its polygon edges.
   */
  setWireframe(on) {
    this.current?.userData.views.setWire(!!on);
    this.dirty = true;
  }

  stopLoading() {
    this.loading?.abort();   // cancels the GLB / OBJ download and its textures
    this.loading = null;
  }

  /**
   * Centres the current model and places the camera so all of it is in view: from the front, a little
   * from the right and above, or (keepDirection, the F key) from where the artist looks at it now.
   * Returns false when there is no model to frame.
   */
  frame({ keepDirection = false } = {}) {
    const pivot = this.current;
    if (!pivot) return false;
    const direction = keepDirection ? this.camera.position.clone().sub(this.controls.target) : null;
    const radius = fitCamera(this.camera, pivot, direction);
    if (!radius) return false;
    this.orbit.fit(pivot);
    this.stopGlide();
    this.controls.target.set(0, 0, 0);
    this.controls.minDistance = radius * 0.2;
    this.controls.maxDistance = radius * 8;
    this.controls.update();
    this.syncGizmo();           // the files' floor follows the model to the middle of the view
    this.dirty = true;
    return true;
  }

  /** What is left of a drag's damped glide goes: it would carry the camera off the place it has. */
  stopGlide() {
    this.controls._sphericalDelta?.set(0, 0, 0);
    this.controls._panOffset?.set(0, 0, 0);
  }

  /**
   * How much perspective the view has (LensCamera.perspective): 1, a camera's, is what every stage shows; the stage
   * that turns it down or inside out sets it back when it goes.
   */
  setPerspective(amount) {
    if (!this.camera || this.camera.perspective === amount) return;
    this.camera.perspective = amount;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }
}

/**
 * The viewer's camera: a perspective camera whose perspective can be turned down, or inside out, about the point the
 * artist orbits (`about`), what lies at that point's depth keeping its size. `perspective`: 1 as a camera sees, 0
 * none (orthographic), below 0 inverse - the lines of sight meet behind the model, so what is farther shows larger
 * and the sides of a round shape turn towards the artist.
 */
export class LensCamera extends THREE.PerspectiveCamera {
  constructor(fov, aspect, near, far) {
    super(fov, aspect, near, far);
    this.perspective = 1;
    this.about = new THREE.Vector3();
  }

  /**
   * {tan, wide, far}: half the picture's height at depth z, from the camera along its view, is wide + tan * z, and
   * nothing shows deeper than `far`. A camera's own: its angle's tangent, wide 0.
   */
  lens() {
    const half = Math.tan(THREE.MathUtils.degToRad(this.fov) / 2) / this.zoom;
    const tan = half * this.perspective;
    const wide = half * this.position.distanceTo(this.about) * (1 - this.perspective);
    return { tan, wide, far: tan < 0 ? Math.min(this.far, (LENS_REACH * wide) / -tan) : this.far };
  }

  updateProjectionMatrix() {
    super.updateProjectionMatrix();
    if ((this.perspective ?? 1) === 1) return;         // (also while the camera is being made)
    const { tan, wide, far } = this.lens();
    // A point's clip w is the picture's half height at its depth; its clip z runs from -w at `near` to w at `far`.
    const slope = (2 * wide + tan * (this.near + far)) / (far - this.near);
    const offset = -(wide + tan * this.near) - slope * this.near;
    this.projectionMatrix.set(
      1 / this.aspect, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, -slope, offset,
      0, 0, -tan, wide,
    );
    this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
  }

  /** A bent lens follows the distance to the point it turns about: whoever reads the camera's place reads its lens. */
  updateMatrixWorld(force) {
    super.updateMatrixWorld(force);
    if (this.perspective !== 1) this.updateProjectionMatrix();
  }
}

function makeRenderer(options = {}) {
  const r = new THREE.WebGLRenderer({ antialias: true, alpha: true, ...options });
  r.outputColorSpace = THREE.SRGBColorSpace;
  r.toneMapping = THREE.NeutralToneMapping;   // keeps the generated albedo colours faithful
  r.setClearColor(0x000000, 0);
  return r;
}

// The studio's lights, from where they shine: a warm key from the front right above, a cool rim from behind.
const KEY_LIGHT = { color: 0xfff4e8, intensity: 1.4, from: [3, 5, 4] };
const RIM_LIGHT = { color: 0xc8d4ff, intensity: 0.9, from: [-4, 2.5, -4] };

/**
 * The scene every model is shown in: a studio environment for the reflections, a key and a rim light (userData:
 * key, rim, ambient). It lights the thumbnails, and the viewer in every view but "all" (Viewer.orbitLight).
 */
function studio(renderer) {
  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();
  const [key, rim] = [KEY_LIGHT, RIM_LIGHT].map(({ color, intensity, from }) => {
    const light = new THREE.DirectionalLight(color, intensity);
    light.position.fromArray(from);
    return light;
  });
  const ambient = new THREE.HemisphereLight(0xffffff, 0x2a2d38, AMBIENT);
  scene.add(key, rim, ambient);
  Object.assign(scene.userData, { key, rim, ambient });
  return scene;
}

// The ambience: the hemisphere light and the environment's reflections. In "all" (OrbitLights) both drop to a
// fifth: a little fill, so no side is black, low enough for the moving lights to read.
const AMBIENT = 0.35;
const ORBIT_AMBIENCE = 0.2;

/** A job is at work (not only queued): the viewer then draws only what the artist moves. */
const jobWorks = () => (store.jobs || []).some((j) => j.status === 'running' || j.status === 'starting_tool');

// ---- the lights of "all" ------------------------------------------------------------------------

// Six point lights start on the corners of an octahedron round the model, each a quarter turn from its four
// neighbours, and go round it in pairs: the two of a pair opposite each other on a great circle of their own, the
// three circles at right angles. So no two ever come closer than 60 degrees (lights on circles drawn at random soon
// pass through each other, and a side goes dark), and in a turn a light passes within 35 degrees of every side of
// the model. One circle goes over the top and under the bottom, the other two lean at 45 degrees (ORBIT_FRAME, the
// octahedron's axes); the way the circles face and the way each pair goes are drawn from ORBIT_SEED, the same every
// time.
const ORBIT_FRAME = [[1, 0, 0], [0, Math.SQRT1_2, Math.SQRT1_2], [0, -Math.SQRT1_2, Math.SQRT1_2]];
const ORBIT_SEED = 1;
const ORBIT_MS = 8700;       // one turn of a light round the model
const ORBIT_FPS = 30;        // the frames a second their turning asks for (the artist's own moves get every frame)
const ORBIT_STEP_MS = 100;   // the most one frame moves them: after a hidden tab they go on from where they were
const ORBIT_RADIUS = 1.25;   // their distance from the model's centre, in model radii (half its box's diagonal)
// Six lights of one strength that do not fade light a model from every side alike: its shading hardly changes as
// they go round. So each fades as a lamp does (ORBIT_DECAY) and reaches ORBIT_REACH model radii and no further: it
// lights the side it passes and leaves the far side to the others. And the two of a pair differ (ORBIT_PAIR, the
// first of a pair and the one opposite it): what one lights the other does not light as much, so the model has a
// lit side and a shaded one, which go round with them. ORBIT_INTENSITY is a light's strength at one model radius,
// the pair's mean - told in model radii, so every prop is lit alike whatever its size: about as bright where it is
// lit as under far lights of 1.4 + 1.4 + 0.9 (a key, a second key and the rim), half as bright in its shade.
// A stage may have them reach further (Viewer.setLightReach): Retopo's bare clay, ORBIT_REACH_WIDE.
const ORBIT_REACH = 1.725;
export const ORBIT_REACH_WIDE = 1.95;
const ORBIT_DECAY = 2;
const ORBIT_INTENSITY = 2.0;
const ORBIT_PAIR = [4 / 3, 2 / 3];
// Each light shows as a small glow, PARTICLE_SIZE model radii across, but never more than PARTICLE_ANGLE radians of
// the view: one passing close by the camera stays a dot.
const PARTICLE_SIZE = 0.1;
const PARTICLE_ANGLE = 0.05;

/** A seeded random number generator (mulberry32): () -> [0, 1). */
function seededRandom(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The lights' paths for `seed`: [{start, travel}] in pairs, two unit vectors at right angles each, the light's place
 * at the start and the way it sets off (orbitPlace).
 */
export function orbitPaths(seed = ORBIT_SEED) {
  const random = seededRandom(seed);
  const yaw = random() * Math.PI * 2;
  const [cos, sin] = [Math.cos(yaw), Math.sin(yaw)];
  const frame = ORBIT_FRAME.map(([x, y, z]) => [x * cos + z * sin, y, z * cos - x * sin]);
  return frame.flatMap((start, k) => {
    const way = random() < 0.5 ? -1 : 1;
    const travel = frame[(k + 1) % 3].map((v) => v * way);
    return [{ start, travel }, { start: start.map((v) => -v), travel: travel.map((v) => -v) }];
  });
}

/** Where a light on `path` is, as a unit vector from the model's centre, once it has gone round by `angle` (radians). */
export function orbitPlace({ start, travel }, angle) {
  const [cos, sin] = [Math.cos(angle), Math.sin(angle)];
  return start.map((v, i) => v * cos + travel[i] * sin);
}

/** A light's glow: white, its alpha a bright core in a soft halo, nothing at the edge. */
function glowTexture(size = 64) {
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const d = Math.hypot(x + 0.5 - size / 2, y + 0.5 - size / 2) / (size / 2);   // 0 at the centre, 1 at the edge
      const glow = d < 1 ? Math.min(1, Math.exp(-((d / 0.15) ** 2)) + 0.35 * Math.exp(-((d / 0.45) ** 2))) * (1 - d) : 0;
      data.set([255, 255, 255, Math.round(glow * 255)], (y * size + x) * 4);
    }
  }
  const texture = new THREE.DataTexture(data, size, size);
  texture.generateMipmaps = true;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

/**
 * The point lights of "all" on their paths (orbitPaths) round the model fitted, each with its glow. Made once with
 * the viewer's scene and kept in it, dark and without glows until lit (Viewer.orbitLight); the glows are never
 * picked by a ray, and the thumbnails' scene has none.
 */
export class OrbitLights {
  constructor(scene, seed = ORBIT_SEED) {
    this.paths = orbitPaths(seed);
    this.angle = 0;                     // how far they have gone round (radians)
    this.centre = new THREE.Vector3();  // the model's, and its radius
    this.radius = 1;
    this.reach = ORBIT_REACH;           // in model radii
    this.lit = false;                   // they shine (light)
    const glow = new THREE.SpriteMaterial({
      map: glowTexture(), color: KEY_LIGHT.color, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false,
    });
    this.lights = this.paths.map(() => new THREE.PointLight(KEY_LIGHT.color, 0, 0, ORBIT_DECAY));
    this.glows = this.lights.map((light) => {
      const sprite = new THREE.Sprite(glow);
      sprite.visible = false;
      sprite.raycast = () => {};
      light.add(sprite);
      return sprite;
    });
    scene.add(...this.lights);
    this.place();
  }

  /** Puts the lights round `object`: about the centre of its box, ORBIT_RADIUS times its radius away. */
  fit(object) {
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) return;
    box.getCenter(this.centre);
    this.radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-3);
    this.place();
    this.power();
  }

  /** The lights shine and their glows show (`on`), or both go out. */
  light(on) {
    this.lit = !!on;
    this.power();
    for (const sprite of this.glows) sprite.visible = this.lit;
  }

  /** Each light's reach and strength for the model fitted: as strong at one of its radii whatever its size. */
  power() {
    this.lights.forEach((light, i) => {
      light.distance = this.radius * this.reach;
      light.intensity = this.lit ? ORBIT_INTENSITY * ORBIT_PAIR[i % 2] * this.radius ** ORBIT_DECAY : 0;
    });
  }

  /** Every light goes on along its path by `angle` (radians). */
  turn(angle) {
    this.angle = (this.angle + angle) % (Math.PI * 2);
    this.place();
  }

  place() {
    const distance = this.radius * ORBIT_RADIUS;
    this.lights.forEach((light, i) => {
      light.position.fromArray(orbitPlace(this.paths[i], this.angle)).multiplyScalar(distance).add(this.centre);
    });
  }

  /** Sizes the glows for `camera`: PARTICLE_SIZE model radii across, at most PARTICLE_ANGLE of its view. */
  faceCamera(camera) {
    this.lights.forEach((light, i) => {
      const size = Math.min(this.radius * PARTICLE_SIZE, camera.position.distanceTo(light.position) * PARTICLE_ANGLE);
      this.glows[i].scale.setScalar(size);
    });
  }
}

/**
 * Centres the model in `pivot` (its only child) on the origin and moves `camera` back until all of it
 * is in view, looking from the front, a little from the right and above (or along `direction`, from the
 * model toward the camera). Returns the model's radius (0 for an empty model).
 */
function fitCamera(camera, pivot, direction = null) {
  const object = pivot.children[0];
  object.position.set(0, 0, 0);
  pivot.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) return 0;
  object.position.sub(box.getCenter(new THREE.Vector3()));
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-3);
  const vfov = THREE.MathUtils.degToRad(camera.fov);
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
  const dist = (radius / Math.sin(Math.min(vfov, hfov) / 2)) * 0.85;   // the bounding sphere is loose; this fills the view
  camera.near = radius / 100;
  camera.far = radius * 100;
  if (direction?.lengthSq() > 1e-12) camera.position.copy(direction);
  else camera.position.set(0.55, 0.35, 1);
  camera.position.normalize().multiplyScalar(dist);
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();
  return radius;
}

// ---- thumbnails ---------------------------------------------------------------------------------

// Each prop keeps a snapshot of its latest 3D result (thumbs.js). They are drawn by a second, small
// renderer that never goes on screen (one at a time), released again when the thumbnails are done.
// The retopo's quads, blue over the clay: faint, since a few thousand polygons fill a small tile densely.
const THUMB_WIRE = { color: 0x2f6fd8, opacity: 0.45 };
let thumbRig = null;
let thumbChain = Promise.resolve();

/**
 * Draws the GLB or OBJ at `url` into a `size` px square with a transparent background: a GLB with its
 * own materials, an OBJ as clay with its polygon edges in blue. Resolves with a WebP (else PNG) Blob.
 */
export function renderThumbnail(url, kind, { size = 256 } = {}) {
  const run = thumbChain.then(async () => {
    if (!thumbRig) {
      const renderer = makeRenderer({ preserveDrawingBuffer: true });
      renderer.setPixelRatio(1);
      thumbRig = { renderer, scene: studio(renderer), camera: new THREE.PerspectiveCamera(30, 1, 0.01, 1000) };
    }
    const { renderer, scene, camera } = thumbRig;
    renderer.setSize(size, size, false);
    const object = kind === 'obj' ? await loadObj(url, undefined, THUMB_WIRE) : await loadGlb(url, new THREE.LoadingManager());
    const pivot = new THREE.Group();
    pivot.add(object);
    scene.add(pivot);
    try {
      fitCamera(camera, pivot);
      renderer.render(scene, camera);
      return await canvasBlob(renderer.domElement);
    } finally {
      scene.remove(pivot);
      dispose(pivot);
    }
  });
  thumbChain = run.catch(() => {});
  return run;
}

/** Frees the thumbnail renderer's WebGL context (it is made again for the next thumbnail). */
export function releaseThumbnails() {
  thumbChain = thumbChain.then(() => {
    if (!thumbRig) return;
    thumbRig.scene.environment?.dispose();
    thumbRig.renderer.dispose();
    thumbRig.renderer.forceContextLoss();
    thumbRig = null;
  });
  return thumbChain;
}

function canvasBlob(canvas) {
  return new Promise((resolve, reject) => canvas.toBlob((webp) => {
    if (webp?.type === 'image/webp') resolve(webp);
    else canvas.toBlob((png) => (png ? resolve(png) : reject(new Error('The picture could not be encoded.'))), 'image/png');
  }, 'image/webp', 0.9));
}

async function loadGlb(url, manager, onProgress) {
  // GLTFLoader handles Pixal3D's EXT_texture_webp and the core KHR_* extensions natively. It uses the
  // file's TANGENT attribute when there is one (a bake always has it); tangents are never computed here.
  const gltf = await new GLTFLoader(manager).loadAsync(url, (e) => onProgress?.(e.total ? e.loaded / e.total : null));
  gltf.scene.traverse((o) => {
    for (const m of o.isMesh ? [o.material].flat() : []) {
      if (m.userData.spellcast?.tangent_basis === 'mikk' && m.normalMap) patchMikk(m);
    }
  });
  return gltf.scene;
}

const WIRE = { color: 0x12151c, opacity: 0.85 };

async function loadObj(url, signal, wire = WIRE) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`The mesh file could not be loaded (error ${res.status}).`);
  const text = await res.text();
  const group = new OBJLoader().parse(text);
  const clay = new THREE.MeshStandardMaterial({
    ...CLAY, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,   // keeps the wire on top
  });
  group.traverse((m) => {
    if (!m.isMesh) return;
    let g = m.geometry;
    for (const name of Object.keys(g.attributes)) if (name !== 'position') g.deleteAttribute(name);
    g = mergeVertices(g);
    g.computeVertexNormals();
    m.geometry.dispose();
    m.geometry = g;
    m.material = clay;
  });
  // OBJLoader triangulates; draw the file's real polygon edges so the quad flow is visible (the viewer's
  // wireframe: ModelViews.setWire shows or hides them).
  const edges = new THREE.LineSegments(polygonEdges(text),
    new THREE.LineBasicMaterial({ color: wire.color, transparent: true, opacity: wire.opacity }));
  group.add(edges);
  group.userData.edges = edges;
  return group;
}

function polygonEdges(text) {
  const v = [];
  const out = [];
  const seen = new Set();
  for (const raw of text.split('\n')) {
    const line = raw.trim();
    if (line.startsWith('v ')) {
      const [, x, y, z] = line.split(/\s+/);
      v.push(+x, +y, +z);
    } else if (line.startsWith('f ')) {
      const idx = line.split(/\s+/).slice(1).map((t) => {
        const i = parseInt(t, 10);
        return i < 0 ? v.length / 3 + i : i - 1;
      });
      for (let k = 0; k < idx.length; k++) {
        const a = idx[k];
        const b = idx[(k + 1) % idx.length];
        const key = a < b ? a * 2 ** 26 + b : b * 2 ** 26 + a;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(v[3 * a], v[3 * a + 1], v[3 * a + 2], v[3 * b], v[3 * b + 1], v[3 * b + 2]);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(out, 3));
  return g;
}

function dispose(object) {
  object.traverse((o) => {
    o.geometry?.dispose();
    for (const m of [o.material].flat()) {
      if (!m) continue;
      for (const value of Object.values(m)) if (value?.isTexture) value.dispose();
      m.dispose();
    }
  });
}

export const viewer = new Viewer();
