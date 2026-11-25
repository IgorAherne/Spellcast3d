// The None | All | <maps> bar under a 3D preview: switches the shared viewer between the lit render,
// the bare shape and each texture map of the model on screen (viewer3d.js setView). The labels match
// the map buttons of the Instant Meshes viewer. A stage can have a Wireframe toggle at the bar's right end
// (offerWireframe): the model's triangle edges (a clean mesh's polygon edges) over whichever of those views
// is on. A stage without map buttons (buttons: false) can still have the toggle, alone.

import { h } from './dom.js';
import { beckon } from './shell/anim.js';

export const VIEW_LABELS = {
  none: 'None', all: 'All', basecolor: 'Base colour', opacity: 'Opacity', normal: 'Normal',
  roughness: 'Roughness', metallic: 'Metallic', ao: 'AO', emissive: 'Emissive', height: 'Height',
};

const MAP_HINT = 'Shows this one map on the model, unlit, exactly as the file stores it. Parts without '
  + 'this map show the flat value their material uses instead.';
const HINTS = {
  none: 'Takes the maps off: a plain grey surface that shows the shape alone.',
  all: 'The model as a game engine renders it: every map together, lit by a studio environment.',
};
const WIRE_HINT = 'Draws the model’s edges as lines over it, whichever map it shows.';

/**
 * onChange(mode) is called when the artist picks a view (not when the page picks one: select()). buttons:
 * false leaves the map buttons out. bind(viewer) shows the views of the model the
 * viewer shows now (call it after each show()); hold() keeps the bar in place, dimmed and inert, while
 * the next model loads (the viewport does not jump); unbind() hides the bar. offerWireframe(on, onChange)
 * adds the Wireframe toggle, set to `on` (call it again to set it anew): every model bound from then on is
 * shown that way, and onChange(on) is called when the artist switches it (the stage remembers the choice).
 */
export function mapBar({ onChange, buttons = true } = {}) {
  const bar = h('div', { class: 'map-bar', role: 'radiogroup', 'aria-label': 'What the model shows' });
  const wireBtn = h('button', {
    type: 'button', class: 'chip toggle', 'aria-pressed': 'false', title: WIRE_HINT, hidden: true,
    onclick: () => setWire(!wire.on),
  }, 'Wireframe');
  const el = h('div', { class: 'map-row', hidden: true }, bar, wireBtn);
  let viewer = null;
  let wire = null;   // {on, onChange} once the stage offers the toggle

  // byArtist: a button was pressed (onChange hears of it); else the page picked the view for the next model.
  const pick = (mode, byArtist = true) => {
    if (!viewer?.setView(mode)) return false;
    sync();
    if (byArtist) onChange?.(mode);
    return true;
  };
  const sync = () => {
    for (const b of bar.children) {
      const on = b.dataset.mode === viewer?.view;
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', String(on));
    }
    wireBtn.classList.toggle('on', !!wire?.on);
    wireBtn.setAttribute('aria-pressed', String(!!wire?.on));
  };
  const setWire = (on) => {
    wire.on = on;
    viewer?.setWireframe(on);
    sync();
    wire.onChange?.(on);
  };

  const hold = (on) => {
    el.classList.toggle('waiting', on);
    el.inert = on;
  };
  el.bind = (v) => {
    viewer = v;
    hold(false);
    const modes = buttons ? v.availableViews() : [];
    bar.hidden = !modes.length;
    el.hidden = bar.hidden && !wire;   // a model without texture maps still has its triangles to show
    bar.replaceChildren(...modes.map((mode) => h('button', {
      type: 'button', class: `map-btn${mode === 'none' || mode === 'all' ? ' view' : ''}`, role: 'radio',
      'data-mode': mode, title: HINTS[mode] || MAP_HINT, onclick: () => pick(mode),
    }, VIEW_LABELS[mode] || mode)));
    if (wire) v.setWireframe(wire.on);   // show() starts every model without it
    sync();
  };
  el.hold = () => {
    viewer = null;
    hold(true);
  };
  el.unbind = () => {
    viewer = null;
    hold(false);
    el.hidden = true;
    bar.replaceChildren();
  };
  el.offerWireframe = (on, onWire) => {
    wire = { on: !!on, onChange: onWire };
    wireBtn.hidden = false;
    sync();
  };
  /** The stage offered the Wireframe toggle: the bar shows for every model, with or without maps. */
  el.hasWireframe = () => !!wire;
  /** "Look here": the toggle rings and swells once (a stage points at it from a control it goes with). */
  el.ringWireframe = () => { if (wire) beckon(wireBtn); };
  /**
   * Picks `mode` when the model has it (e.g. to keep the same map while switching models); onChange does not
   * hear of it, the artist did not pick it. Returns whether the model is in that view now.
   */
  el.select = (mode) => !!viewer?.availableViews().includes(mode) && pick(mode, false);
  return el;
}
