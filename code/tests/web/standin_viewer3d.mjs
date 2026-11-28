// viewer3d.js's shared viewer in the stage tests (a resolve hook puts this module in its place): it shows each model
// asked for at once - a stand-in model of 64 x 64 texels on the stand-in card (standin_card.mjs), one per URL - and
// does what the views ask of it. Its camera is moved by the tests alone, which then tell its controls' listeners so
// (a "change", as OrbitControls does).
import { StandInRenderer, standInCamera, standInModel } from './standin_card.mjs';

export const TEX = 64;
export const ORBIT_REACH_WIDE = 1.95;
const models = new Map();       // url -> standInModel

export const viewer = {
  renderer: new StandInRenderer(),
  camera: standInCamera(),
  controls: Object.assign(new EventTarget(), { update() {} }),
  current: null,
  view: 'all',
  paintHook: null,
  holdLight: false,
  setPerspective(amount) { this.camera.perspective = amount; },
  lightReach: undefined,        // as the last stage asked (setLightReach: none given, the usual)
  setLightReach(reach) { this.lightReach = reach; },
  shown: [],                    // the URLs shown, in order
  attach() {},
  detach() {},
  frame() {},
  redraw() {},
  placements: [],               // the placement each show was given ({move, turn} | null), in order
  gizmo: null,                  // the handles up ('move' | 'turn' | null) and whom they tell: setGizmo
  onPlaced: null,
  placed: { move: [0, 0, 0], turn: [0, 0, 0, 1] },
  async show(url, _kind, { placement = null } = {}) {
    if (!models.has(url)) models.set(url, standInModel(TEX));
    this.current = models.get(url).pivot;
    this.view = 'all';
    this.shown.push(url);
    this.placements.push(placement);
    this.placed = placement || { move: [0, 0, 0], turn: [0, 0, 0, 1] };
    this.gizmo = this.onPlaced = null;
    return true;
  },
  async setGizmo(mode, onPlaced = null) {
    this.gizmo = mode || null;
    this.onPlaced = mode ? onPlaced : null;
  },
  /** A test drags the handles: the model stands at `placement`, and the pane is told as after a real drag. */
  drag(placement) {
    this.placed = placement;
    this.onPlaced?.(placement);
  },
  placeOnFloor() {
    this.placed = { move: [0, 0.5, 0], turn: this.placed.turn };
    return this.placed;
  },
  placeBack() {
    this.placed = { move: [0, 0, 0], turn: [0, 0, 0, 1] };
    return this.placed;
  },
  /** The viewer lets go of the model of `url`, its painting with it (as its cache does): shown again, it is loaded anew. */
  forget(url) {
    models.get(url)?.pivot.userData.paint?.dispose();
    models.delete(url);
  },
  availableViews: () => ['none', 'all', 'basecolor'],
  setView(mode) {
    this.view = mode;
    return true;
  },
  setWireframe() {},
  setLayer() {},
};
