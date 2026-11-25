// Stage 6, "Export All" (still 'rig' in the code, the project files and the hub; rigging itself is to
// come): where Projection's "Export all" goes down to. The workbench shows the prop's finished model (the
// same viewer and navigation as 3D and Projection) and says where the exported props are.

import { api } from '../api.js';
import { h } from '../dom.js';
import { icon } from '../icons.js';
import { projectId } from '../store.js';
import { attempt } from '../ui.js';
import {
  afterMove, boxFile, boxHeading, discordCorner, donateCorner, emptyState, fadeIn, focusBox, thumbUrl,
} from './common.js';
import { modelPane } from './preview.js';
import { omnitextureUsed } from './rules.js';

const EXPORTED = '“Export all” in Projection saves every finished prop into its own folder in the project’s exports '
  + 'folder, ready to drag into your game engine. Press it again after a change. Rigging (skeletons for '
  + 'characters) is coming later.';

/** Opens the project's exports folder in File Explorer again (Export all opened it once; a closed window is no loss). */
const openExports = () => attempt(() => api('POST', '/api/open-folder', { which: 'exports', project: projectId() }));
const exportsButton = () => h('button', { class: 'btn ghost', onclick: openExports }, icon('folder'), 'Open the exports folder');

let host = null;
let ctx = null;
let view = null;         // the prop view on screen: {key, el, update(prop), destroy()}
let shownId = null;      // the prop shown last, kept while nobody picks another

/**
 * The prop's most finished model, as Export all saves it: the baked low poly, else its clean mesh, else its
 * Omni-Texture while used, else its 3D model.
 */
function bestModel(p) {
  if (p.projection?.glb) return { url: boxFile(p, p.projection.glb, 'projection'), kind: 'glb', snapshot: thumbUrl(p, 'projection') };
  if (p.retopo?.obj) return { url: boxFile(p, p.retopo.obj, 'retopo'), kind: 'obj', snapshot: thumbUrl(p, 'retopo') };
  if (omnitextureUsed(p)) {
    return { url: boxFile(p, p.omnitexture.glb, 'omnitexture'), kind: 'glb', snapshot: thumbUrl(p, 'omnitexture') };
  }
  if (p.mesh?.glb) return { url: boxFile(p, p.mesh.glb, 'mesh'), kind: 'glb', snapshot: thumbUrl(p, 'mesh') };
  return null;
}

function buildProp(prop) {
  const pane = modelPane({ discord: true });
  const heading = boxHeading();
  const side = h('div', { class: 'sv-side' },
    h('div', { class: 'sv-side-scroll' }, heading,
      emptyState('folder', 'Exported props', EXPORTED, exportsButton(),
        h('button', { class: 'btn ghost', onclick: () => ctx?.goto('projection') }, icon('sun'), 'Go to Projection'))));
  const el = h('div', { class: 'sv sv-rig' }, h('div', { class: 'sv-media' }, pane.el), side);

  let shown = null;
  const update = (p) => {
    heading.set(p, '', thumbUrl(p, 'projection') || thumbUrl(p, 'mesh')
      || (p.images?.chosen ? thumbUrl(p, 'images') || boxFile(p, p.images.chosen, 'images') : null));
    const model = bestModel(p);
    const key = model ? model.url : 'none';
    if (key === shown) return;
    shown = key;
    if (!model) {
      pane.message(emptyState('cube', 'No 3D model yet', 'This prop becomes 3D in the 3D stage first.'));
      return;
    }
    pane.show(model.url, model.kind, afterMove(ctx, model.snapshot));
  };
  return { key: prop.id, el, update, destroy: () => pane.destroy() };
}

function buildEmpty() {
  const el = h('div', { class: 'sv sv-rig sv-single' },
    h('div', { class: 'sv-stagebox' }, emptyState('folder', 'Exported props', EXPORTED, exportsButton(),
      h('button', { class: 'btn ghost', onclick: () => ctx?.goto('projection') }, icon('sun'), 'Go to Projection')),
    donateCorner(), discordCorner()));
  return { key: 'empty', el, update() {}, destroy() {} };
}

function render() {
  if (!host) return;
  const { box: prop } = focusBox(ctx, 'rig', { previousId: shownId });
  shownId = prop?.id ?? null;
  if (view?.key !== (prop ? prop.id : 'empty')) {
    view?.destroy();
    view = prop ? buildProp(prop) : buildEmpty();
    host.replaceChildren(view.el);
    fadeIn(view.el);
  }
  if (prop) view.update(prop);
}

export default {
  id: 'rig',
  title: 'Export',
  ctaLabel: null,
  plus: null,

  mount(el, c) {
    host = el;
    ctx = c;
    render();
  },

  update(c) {
    ctx = c;
    render();
  },

  unmount() {
    view?.destroy();
    view = null;
    host = null;
  },

  // No marks on the tiles here: nothing is made in this stage.
  status: () => null,

  /** The prop the workbench shows. */
  shownId: () => shownId,

  ctaEnabled: () => ({ enabled: false, hint: 'This is the last stage.' }),

  async onCta() {},
};
