// The six stages a prop goes through, as the shell words them, and the stage views (js/stages/*.js)
// that fill the workbench. A view that cannot be loaded is replaced by one that says so, so one broken
// stage never takes the others down.

import { h } from '../dom.js';
import { icon } from '../icons.js';
import { STAGES } from '../store.js';

// title: the stage's name (the "↑" button back up to it); heading: what it does to every prop, the props
// column's big heading, on one line; tip: what happens in it (the heading's tooltip). Omni-Texture is
// optional: only the props the artist chooses there. The last stage (still 'rig' in the project files and the hub)
// is where the exported props end up.
export const STAGE_INFO = {
  images: { title: 'Images', heading: 'All Images', tip: 'Give every prop its picture: import one, or generate it from your words' },
  mesh: { title: '3D', heading: 'All 3D', tip: 'Every picture becomes a 3D model' },
  omnitexture: {
    title: 'Omni-Texture', heading: 'Omni-Texture All',
    tip: 'A new, sharper colour texture for the props you choose, painted by the image generator from a few sides',
  },
  retopo: { title: 'Retopo', heading: 'All Retopo', tip: 'Clean, game-ready polygons and UVs for every 3D model' },
  projection: { title: 'Projection', heading: 'Project All', tip: 'The detail and colours are baked onto the clean model' },
  rig: { title: 'Export', heading: 'Export All', tip: 'Every finished prop, saved into its own folder in the project folder' },
};

/** The stage after `stage`, where its call to action goes down to (null for the last one). */
export function nextStage(stage) {
  const i = STAGES.indexOf(stage);
  return i >= 0 && i < STAGES.length - 1 ? STAGES[i + 1] : null;
}

/** The stage before `stage` (the "↑" button over the heading), or null for the first one. */
export function prevStage(stage) {
  const i = STAGES.indexOf(stage);
  return i > 0 ? STAGES[i - 1] : null;
}

/** A stand-in for a stage view whose module failed to load: it explains, and does nothing else. */
function brokenView(id, err) {
  return {
    id, title: STAGE_INFO[id].title, ctaLabel: null, plus: null,
    mount(host) {
      host.replaceChildren(h('div', { class: 'centre-note' },
        icon('alert', 'big'),
        h('h3', null, 'This part of Spellcast3D could not be loaded'),
        h('p', null, `${err?.message || err}. Reload the page; if it stays like this, update Spellcast3D.`)));
    },
    update() {},
    unmount() {},
    status: () => null,
    ctaEnabled: () => ({ enabled: false }),
    async onCta() {},
    async onPlus() {},
  };
}

/** Loads every stage view: {images: view, mesh: view, ...}. */
export async function loadViews() {
  const views = await Promise.all(STAGES.map((id) => import(`../stages/${id}.js`)
    .then((m) => m.default)
    .catch((e) => {
      console.error(`The ${id} stage could not be loaded:`, e);
      return brokenView(id, e);
    })));
  return Object.fromEntries(STAGES.map((id, i) => [id, views[i]]));
}
