// A graphics card for the page tests, as far as paint3d.js uses one (no browser, no GPU): three.js's WebGLRenderer
// keeping each render target's RGBA bytes (StandInRenderer), a model on it as the viewer shows one (standInModel), a
// viewer showing it (standInViewer) and a stroke of the tool that is up (strokeAt). Importing it first lets the page's
// modules import 'three'.
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

const STATIC = new URL('../../spellcast/static/js/', import.meta.url);
const THREE_URL = new URL('../vendor/three/build/three.module.js', STATIC).href;
registerHooks({ resolve: (specifier, context, next) => (specifier === 'three' ? { url: THREE_URL, shortCircuit: true } : next(specifier, context)) });
export const THREE = await import(THREE_URL);

/**
 * three.js's WebGLRenderer as far as paint3d.js uses it, keeping each render target's RGBA bytes: a picture copied in
 * (a fill), tiles written (undo, a kept painting put on), reads. Like three.js on a real card it draws nothing while
 * the context is lost; once it is back every render target is empty, and a read that spanned the loss hands back
 * zeros without a word. An async read takes the pixels as they are when it is asked for (three.js reads them into a
 * buffer on the card at once) and hands them over later (`slowRead`: a promise it waits for). Of the passes it counts
 * them all, and draws three as the card would: a stroke's (it paints the texels of `brush` in its layer - the marker
 * its colour, the clone stamp what is 4 texels to the left, the overlay full coverage), the composite of the base with
 * the overlay, and the model in sight (sight).
 */
export class StandInRenderer {
  constructor() {
    this.domElement = Object.assign(new EventTarget(), {
      width: 100, height: 100, style: {}, getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }),
    });
    this.lost = false;
    this.era = 0;                      // a new one with every restored context
    this.memory = new Map();           // a render target's texture -> its bytes
    this.target = null;
    this.autoClear = true;
    this.extensions = { has: () => true };
    this.slowRead = null;              // a promise an async read waits for before it hands its pixels over
    this.renders = 0;                  // passes drawn
    this.reads = 0;                    // async reads asked for
    this.brush = { x: 0, y: 0, w: 0, h: 0 };   // the texels a stroke covers
  }

  getContext() { return { isContextLost: () => this.lost }; }
  getDrawingBufferSize(size) { return size.set(this.domElement.width, this.domElement.height); }
  bytes(texture, width = texture.image.width) {
    if (!this.memory.has(texture)) this.memory.set(texture, new Uint8Array(width * width * 4));
    return this.memory.get(texture);
  }

  initRenderTarget(rt) { if (!this.lost) this.bytes(rt.texture); }
  getRenderTarget() { return this.target; }
  setRenderTarget(rt) { this.target = rt; }
  getClearColor(colour) { return colour; }
  getClearAlpha() { return 0; }
  setClearColor() {}
  clear() { if (!this.lost && this.target) this.bytes(this.target.texture).fill(0); }
  render(scene) {
    this.renders++;
    if (this.lost || !this.target) return;
    const material = (scene.children.find((m) => m.userData.inside) || scene.children[0])?.material;
    const u = material?.uniforms || {};
    if (u.src) this.bytes(this.target.texture).set(u.src.value.image.data || u.src.value.image.pixels);   // a fill's copy
    else if (u.base && u.overlay) this.composite(this.bytes(this.target.texture), u);
    else if (u.overlay) this.sight(this.bytes(this.target.texture), u);
    else if (u.before && u.islands) this.stroke(this.bytes(this.target.texture), u, material.defines);
  }

  /**
   * The model in sight, small (paint3d.js Session.sight): the middle of the picture, from 0.3 to 0.7 across and 0.1 to
   * 0.9 down; masked at its very middle while the overlay covers anything.
   */
  sight(into, u) {
    const { width, height } = this.target;
    const cover = u.overlay.value && this.memory.get(u.overlay.value);
    const masked = !!cover?.some((v, i) => i % 4 === 0 && v > 0);
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) {
        const [fx, fy] = [(col + 0.5) / width, (row + 0.5) / height];
        if (fx < 0.3 || fx > 0.7 || fy < 0.1 || fy > 0.9) continue;
        const i = (row * width + col) * 4;
        into[i + 1] = into[i + 3] = 255;
        if (masked && Math.abs(fx - 0.5) < 0.05 && Math.abs(fy - 0.5) < 0.05) into[i] = 255;
      }
    }
  }

  stroke(into, u, defines) {
    const before = this.memory.get(u.before.value);
    const size = this.target.width;
    const rgb = [u.colour.value.r, u.colour.value.g, u.colour.value.b].map((c) => Math.round(c * 255));
    const { x, y, w, h } = this.brush;
    for (let row = y; row < y + h; row++) {
      for (let col = x; col < x + w; col++) {
        const i = (row * size + col) * 4;
        if ('OVERLAY' in defines) into.fill(255, i, i + 4);
        else if ('CLONE' in defines) into.set(before.subarray(i - 16, i - 13), i);
        else into.set(rgb, i);
      }
    }
  }

  composite(into, u) {
    const base = this.memory.get(u.base.value);
    const cover = this.memory.get(u.overlay.value);
    const rgb = [u.colour.value.r, u.colour.value.g, u.colour.value.b].map((c) => c * 255);
    for (let i = 0; i < into.length; i += 4) {
      for (let k = 0; k < 3; k++) into[i + k] = Math.round(base[i + k] + (rgb[k] - base[i + k]) * (cover[i] / 255));
      into[i + 3] = base[i + 3];
    }
  }

  copyTextureToTexture(tile, texture, region, at) {
    if (this.lost) return;
    const into = this.bytes(texture);
    if (this.memory.has(tile)) {       // a render target's texture, whole (a stroke's "before")
      into.set(this.memory.get(tile));
      return;
    }
    const { data, width, height } = tile.image;
    const size = Math.sqrt(into.length / 4);
    for (let row = 0; row < height; row++) into.set(data.subarray(row * width * 4, (row + 1) * width * 4), ((at.y + row) * size + at.x) * 4);
  }

  readRenderTargetPixels(rt, x, y, w, h, out) {
    const from = !this.lost && this.memory.get(rt.texture);
    if (!from) return;
    for (let row = 0; row < h; row++) out.set(from.subarray(((y + row) * rt.width + x) * 4, ((y + row) * rt.width + x + w) * 4), row * w * 4);
  }

  async readRenderTargetPixelsAsync(rt, x, y, w, h, out) {
    this.reads++;
    const era = this.era;
    const now = new Uint8Array(out.length);
    this.readRenderTargetPixels(rt, x, y, w, h, now);
    await this.slowRead;
    if (this.lost) throw undefined;     // eslint-disable-line no-throw-literal -- three.js rejects with nothing
    if (era === this.era) out.set(now);
    return out;
  }

  loseContext() {
    this.lost = true;
    this.domElement.dispatchEvent(new Event('webglcontextlost'));
  }

  restoreContext() {
    this.lost = false;
    this.era++;
    this.memory.clear();               // three.js forgets every render target (onContextRestore runs first)
    this.domElement.dispatchEvent(new Event('webglcontextrestored'));
  }
}

/**
 * A model as the viewer shows one (its pivot, with viewer3d.js's ModelViews as far as paint3d.js uses them): a square
 * of `size` texels with a base colour picture (opaque, every texel its own colour) -> {pivot, own (its material),
 * picture (the picture's RGBA bytes)}.
 */
export function standInModel(size) {
  const picture = { width: size, height: size, pixels: Uint8Array.from({ length: size * size * 4 }, (_, i) => (i % 4 === 3 ? 255 : (i * 7) % 256)) };
  const map = new THREE.Texture(picture);
  map.flipY = false;
  const own = new THREE.MeshStandardMaterial({ map });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), own);
  const pivot = new THREE.Group().add(mesh);
  pivot.userData.views = {
    meshes: [{ mesh, own }],
    replaceMap(from, to, patch) {
      if (own.map !== from) return;
      own.map = to;
      patch?.(own);
    },
    apply() {},
  };
  return { pivot, own, picture: picture.pixels };
}

/**
 * A camera 2 units in front of a standInModel: the model fills most of the stand-in's 100 x 100 px viewport. Its
 * lens as viewer3d.js's LensCamera tells it, turned about the model's middle.
 */
export function standInCamera() {
  const camera = new THREE.PerspectiveCamera();
  camera.position.set(0, 0, 2);
  camera.updateMatrixWorld();
  camera.perspective = 1;
  camera.lens = () => {
    const half = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    return { tan: half * camera.perspective, wide: half * camera.position.length() * (1 - camera.perspective), far: camera.far };
  };
  return camera;
}

/** A viewer showing a standInModel of `size` texels -> {viewer, renderer, picture, own}. */
export function standInViewer(size) {
  const renderer = new StandInRenderer();
  const { pivot, own, picture } = standInModel(size);
  const viewer = { renderer, camera: standInCamera(), controls: { update() {} }, current: pivot, view: 'all', redraw() {} };
  return { viewer, renderer, picture, own };
}

/** The left button pressed with a mouse (a pointer event as far as the painter reads one). */
export const press = { pointerType: 'mouse', button: 0, pointerId: 1, shiftKey: false, altKey: false, ctrlKey: false, metaKey: false };

const pointer = (type, x, y) => Object.assign(new Event(type), { pointerId: 1, clientX: x, clientY: y });

/** The tool that is up pressed in the viewport's middle: a stroke begins, which covers `rect` (texels) as it goes. */
export function pressAt(viewer, rect) {
  viewer.renderer.brush = rect;
  assert.equal(viewer.paintHook.down({ ...press, clientX: 50, clientY: 50 }), true, 'the painter takes the press');
}

/** The stroke under way goes on to (x, y) on the viewport, painting `rect` there. */
export function moveTo(viewer, x, y, rect) {
  viewer.renderer.brush = rect;
  viewer.renderer.domElement.dispatchEvent(pointer('pointermove', x, y));
}

/** The pointer let go: the stroke under way ends. */
export const letGo = (viewer) => viewer.renderer.domElement.dispatchEvent(pointer('pointerup', 50, 50));

/** A stroke of the tool that is up, pressed and let go in the viewport's middle: it covers `rect` (texels). */
export function strokeAt(viewer, rect) {
  pressAt(viewer, rect);
  letGo(viewer);
}
