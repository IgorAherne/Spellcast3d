// The Move / Rotate handles of a model pane, on the Retopo and Export pages: where the prop stands in the files of
// the export folder. A model is made centred on its middle, so a character dropped into a game engine sinks half
// into the floor; here the artist moves and turns it over a grid that is the files' floor (its centre their origin),
// or stands it there with one press. The hub keeps the placement on the prop and every saved file holds it in its
// vertices (exports.py); the bake, the paintings and the other pages' views are of the model as it was made.

import { h } from '../dom.js';
import { icon } from '../icons.js';

const GRID = 'The grid is the floor of the saved files, its centre their origin.';
const TOOLS = [
  { mode: 'move', icon: 'move', label: 'Move',
    title: `Move the model in its saved files: drag an arrow, or a square between two. ${GRID}` },
  { mode: 'turn', icon: 'turn', label: 'Rotate',
    title: `Turn the model in its saved files: drag a ring. ${GRID}` },
];
const FLOOR = 'Stand it on the floor: the origin of the saved files goes under the middle of the model, so it stands '
  + 'on the ground when dropped into a game engine.';
const BACK = 'Back as it was made: centred on its middle, not turned.';
const TINY = 1e-6;

// The handles the artist left up on each page, kept until the page closes (as the Wireframe toggle is).
const chosen = {};

/** A placement that moves and turns nothing is none: null. */
export function placementOrNone(placement) {
  if (!placement) return null;
  const { move, turn } = placement;
  const still = move.every((n) => Math.abs(n) < TINY) && turn.slice(0, 3).every((n) => Math.abs(n) < TINY);
  return still ? null : placement;
}

/**
 * The bar of a model pane: [Move] [Rotate] [On the floor] [Back]. stage: the page (its handles are remembered);
 * get(): the prop's placement, or null; set(placement | null): the artist placed it; start: the handles up when
 * the page is first seen ('move' on Export, none on Retopo, where the topology is what is looked at).
 * -> {el, bind(viewer) once a model shows in it, unbind()}.
 */
export function placeBar({ stage, get, set, start = null }) {
  let viewer = null;
  const mode = () => (stage in chosen ? chosen[stage] : start);
  const placed = (placement) => {
    if (!placement) return;
    set(placementOrNone(placement));
    sync();
  };
  const tools = TOOLS.map((tool) => {
    const button = h('button', {
      type: 'button', class: 'icon-btn glass', title: tool.title, 'aria-label': tool.label, 'aria-pressed': 'false',
      onclick: () => pick(mode() === tool.mode ? null : tool.mode),
    }, icon(tool.icon));
    button.dataset.mode = tool.mode;
    return button;
  });
  const floor = h('button', {
    type: 'button', class: 'icon-btn glass', title: FLOOR, 'aria-label': 'Stand it on the floor',
    onclick: async () => {
      if (!viewer) return;
      if (!mode()) await pick('move');     // the floor it stands on shows with the handles
      placed(viewer?.placeOnFloor());
    },
  }, icon('floor'));
  const back = h('button', {
    type: 'button', class: 'icon-btn glass', title: BACK, 'aria-label': 'Back as it was made',
    onclick: () => placed(viewer?.placeBack()),
  }, icon('reset'));
  const el = h('div', { class: 'sv-place', role: 'toolbar', 'aria-label': 'Place the model in its saved files', hidden: true },
    tools, h('span', { class: 'sv-place-gap' }), floor, back);

  function sync() {
    for (const button of tools) button.setAttribute('aria-pressed', String(button.dataset.mode === mode()));
    back.disabled = !placementOrNone(get());
  }
  function pick(next) {
    chosen[stage] = next;
    sync();
    return viewer?.setGizmo(next, placed);
  }

  return {
    el,
    /** A model shows in `shown` (the pane's viewer): the bar is up, the handles as the artist left them. */
    bind(shown) {
      viewer = shown;
      el.hidden = false;
      sync();
      viewer.setGizmo(mode(), placed);
    },
    unbind() {
      viewer = null;
      el.hidden = true;
    },
  };
}
