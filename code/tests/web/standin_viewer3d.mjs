// viewer3d.js's shared viewer in the stage tests (a resolve hook puts this module in its place): it shows each model
// asked for at once - a stand-in model of 64 x 64 texels on the stand-in card (standin_card.mjs), one per URL - and
// does what the views ask of it. Its camera is moved by the tests alone: `held` says whether the artist could.
import { StandInRenderer, standInCamera, standInModel } from './standin_card.mjs';

export const TEX = 64;
const models = new Map();       // url -> standInModel

export const viewer = {
  renderer: new StandInRenderer(),
  camera: standInCamera(),
  controls: { update() {} },
  current: null,
  view: 'all',
  paintHook: null,
  holdLight: false,
  held: false,
  holdView(on) { this.held = !!on; },
  shown: [],                    // the URLs shown, in order
  attach() {},
  detach() {},
  frame() {},
  redraw() {},
  async show(url) {
    if (!models.has(url)) models.set(url, standInModel(TEX));
    this.current = models.get(url).pivot;
    this.view = 'all';
    this.shown.push(url);
    return true;
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
