// The 3D preview of a stage view: the shared three.js viewer (viewer3d.js, loaded on first use) in a
// dark stage, with a loading layer, a message layer, badges, "reset the view", the navigation hint in
// its bottom-right corner, Donate in its bottom-left one and the map bar under it, on the right (None |
// All | the model's maps). While the stages swipe, a model waits for them to settle, with the prop's
// snapshot standing in: loading one takes the page's full attention for a moment.
//
// Also the [.glb] [.fbx] [.obj] buttons that save a prop's model into the project's export folder.

import { api, propApi } from '../api.js';
import { h, plural } from '../dom.js';
import { icon } from '../icons.js';
import { progressBar, spinner } from '../controls.js';
import { mapBar } from '../mapbar.js';
import { projectId } from '../store.js';
import { toast } from '../ui.js';
import { discordCorner, donateCorner, emptyState } from './common.js';

// The main moves (viewer3d.js dragAction has them all: Blender's and 3ds Max's modifiers work too).
const NAV_HINT = 'Orbit: middle or left drag · Pan: right drag · Zoom: wheel · F: frame';
// What the artist reads when the browser gives the page no 3D context (it blocks a page after the graphics driver was
// reset under it, until the browser starts again).
export const NO_WEBGL = 'Your models are fine. The browser refuses 3D views on this page, which it does after the '
  + 'graphics driver was reset or with hardware acceleration switched off. Close the browser completely, open it '
  + 'again and come back to this address.';

// The view a model is shown in first, in 3D, Projection and Export: "All", lit with every map (its baked AO left
// out: viewer3d.js apply). A stage keeps the view the artist picks after that (Projection's lastView).
export const DEFAULT_VIEW = 'all';

// The Wireframe toggle's choice, per stage, kept until the page closes: Retopo starts with the edges on (both its
// High and Low sides: the topology is what the artist judges there), Projection and 3D with them off; a stage keeps
// what the artist last chose in it, on every model - never switched off behind the artist's back.
const wireChoice = {};
export const wireframeOn = (stage, stageDefault) => (wireChoice[stage] ?? stageDefault);
export const rememberWireframe = (stage, on) => { wireChoice[stage] = !!on; };

/**
 * tools: nodes placed over the model's top right corner, next to "reset the view" (e.g. the High | Low
 * switch). onView(mode): the artist picked a map view. maps: false leaves the map buttons out (the model
 * is shown in the view show() asks for); the Wireframe toggle (maps.offerWireframe) still shows under it,
 * alone, for every model, so the view keeps its size whatever the model has.
 * Returns {el, stage (the dark area, for layers of the view on top), maps, show(), message(), badges(),
 * destroy()}.
 */
export function modelPane({ tools = [], onView, maps: withMaps = true, discord = false } = {}) {
  const host = h('div', { class: 'sv-viewer-host' });
  const bar = progressBar(null);
  const loading = h('div', { class: 'sv-viewer-loading', hidden: true }, spinner(), h('span', null, 'Loading the 3D model…'), bar);
  const still = h('img', { class: 'sv-viewer-still', alt: '', draggable: 'false', hidden: true });
  const msg = h('div', { class: 'sv-viewer-msg', hidden: true });
  const badges = h('div', { class: 'sv-badges' });
  const reset = h('button', {
    class: 'icon-btn glass', title: 'Reset the view (F frames the model from where you look)', hidden: true,
    onclick: () => viewer?.frame(),
  }, icon('reset'));
  const help = h('div', { class: 'sv-viewer-help', hidden: true }, NAV_HINT);
  const maps = mapBar({ onChange: onView, buttons: withMaps });
  const stage = h('div', { class: 'sv-stagebox dark' }, host, still, loading, msg, badges,
    h('div', { class: 'sv-viewer-tools' }, tools, reset), help, donateCorner(), discord && discordCorner());
  // While a model loads, the bar keeps its place (or takes it), so the viewport does not jump.
  const mediabar = h('div', { class: 'sv-mediabar' }, maps);
  const el = h('div', { class: 'sv-model' }, stage, mediabar);   // the bar hides itself when it has nothing to show

  let viewer = null;
  let token = 0;
  let destroyed = false;
  let shown = false;      // a model is on screen (show resolved true and nothing replaced it since)

  const idle = () => {
    shown = false;
    viewer?.detach(host);
    maps.unbind();
    mediabar.classList.remove('reserve');
    reset.hidden = help.hidden = loading.hidden = still.hidden = true;
  };
  const showStill = (src) => {
    still.hidden = !src;
    if (src) still.src = src;
    else still.removeAttribute('src');
  };

  return {
    el,
    stage,
    maps,
    /**
     * Shows the GLB / OBJ at `url` (see viewer3d.js show()); view: the map view to pick (the one the
     * artist picked last, where the stage keeps it), else DEFAULT_VIEW, else the lit render; after: a
     * promise to wait for first (the stages swiping: stages/common.js afterMove), with the picture
     * `snapshot` meanwhile. Resolves true once shown, false when superseded; a failure shows a message.
     */
    async show(url, kind, { extraMaps, keepView = false, view = DEFAULT_VIEW, after = null, snapshot = null } = {}) {
      const mine = ++token;
      shown = false;
      msg.hidden = true;
      maps.hold();
      mediabar.classList.toggle('reserve', withMaps || maps.hasWireframe());
      bar.set(null);
      if (after) {
        showStill(snapshot);
        loading.hidden = !!snapshot;
        await after;
        if (destroyed || mine !== token) return false;
      }
      loading.hidden = false;
      try {
        viewer ||= (await import('../viewer3d.js')).viewer;
        if (destroyed || mine !== token) return false;
        viewer.attach(host, stage);
        const ok = await viewer.show(url, kind, { onProgress: (f) => bar.set(f), extraMaps, keepView });
        if (!ok || destroyed || mine !== token) return false;
        showStill(null);
        reset.hidden = help.hidden = false;
        maps.bind(viewer);
        mediabar.classList.toggle('reserve', !maps.hidden);
        // show() starts every model lit ("all"): the view asked for when the model has it, else the default.
        if (view !== 'all' && !maps.select(view)) maps.select(DEFAULT_VIEW);
        shown = true;
        return true;
      } catch (e) {
        if (destroyed || mine !== token) return false;
        console.warn(e);
        idle();
        msg.hidden = false;
        // No 3D view at all (three.js: "Error creating WebGL context") is the browser's refusal, not the model's fault.
        msg.replaceChildren(/WebGL/i.test(String(e.message || e))
          ? emptyState('alert', 'The browser could not start 3D graphics', NO_WEBGL)
          : emptyState('alert', "Couldn't show this model",
            `${e.message || e}. The file itself may still be fine: save it to the project folder and open it in another program.`));
        return false;
      } finally {
        if (mine === token) loading.hidden = true;
      }
    },
    /** Takes the model away and shows `node` (an empty state, a picture...) instead. */
    message(node) {
      token++;
      idle();
      msg.hidden = false;
      msg.replaceChildren(node);
    },
    /** The shared viewer while this pane shows a model in it, else null. */
    get viewer() {
      return shown && !destroyed ? viewer : null;
    },
    badges(...nodes) {
      badges.replaceChildren(...nodes.filter(Boolean));
    },
    destroy() {
      destroyed = true;
      shown = false;
      token++;
      viewer?.detach(host);
    },
  };
}

// ---- saving a prop's model into the export folder ----------------------------------------------------

const FORMATS = [
  { format: 'glb', title: 'One file with the textures inside, for Blender, Godot and the web.' },
  { format: 'fbx', title: 'For Unity, Unreal, 3ds Max and Maya, with the textures inside.' },
  { format: 'obj', title: 'With its .mtl and the texture maps as PNG next to it, for any other program.' },
];

/**
 * The saved model of this prop is TriFlow's clean mesh, or was baked onto it. what: 'projection' (its
 * bake), 'all' (its best result: the bake, else the clean mesh, else the 3D model) or 'mesh' (the 3D
 * model: never).
 */
export function madeWithTriflow(prop, what) {
  if (!prop || what === 'mesh') return false;
  const retopo = prop.retopo || {};
  const bake = prop.projection || {};
  if (bake.glb) return bake.source_retopo ? /(^|\/)triflow\.obj$/.test(bake.source_retopo) : retopo.method === 'triflow';
  return what === 'all' && !!retopo.obj && retopo.method === 'triflow';
}

/**
 * The note TriFlow's terms add to a save ('' for none): "Barrel was made with TriFlow — non-commercial
 * use only." props: the props saved; what: as madeWithTriflow.
 */
export function triflowNote(props, what) {
  const made = (props || []).filter((p) => madeWithTriflow(p, what));
  if (!made.length) return '';
  const who = made.length === 1 ? made[0].name || 'This prop' : plural(made.length, 'prop');
  return `${who} ${made.length === 1 ? 'was' : 'were'} made with TriFlow — non-commercial use only.`;
}

/**
 * The toast after a save into the export folder: "Saved <what> to exports\Tavern props" when File
 * Explorer opened there, else with the folder's whole path (folder_path) so the artist can find it,
 * and what the hub warned about (e.g. a map it could not convert). note: the page's own note (TriFlow's
 * terms: triflowNote), unless the hub's warnings say it already.
 */
export function toastSaved(what, answer, note = '') {
  const { folder = '', folder_path: path = '', opened = false, warnings = [] } = answer || {};
  const where = String((!opened && path) || folder).replace(/\//g, '\\');
  const notes = (Array.isArray(warnings) ? warnings : []).filter((w) => typeof w === 'string' && w);
  if (note && !notes.some((w) => /triflow/i.test(w))) notes.push(note);
  const saved = `Saved ${what} to ${where}`;
  if (notes.length) toast(`${saved}. ${notes.join(' ')}`, 'warn', 12000);
  else toast(saved, 'success');
}

/**
 * The [.glb] [.fbx] [.obj] buttons of a prop's model under a small caption: a press asks the hub to
 * write that file into the prop's folder in the project's export folder (POST .../export {stage, format};
 * stage 'mesh': the 3D model, 'projection': the baked clean mesh), which then shows it in File Explorer,
 * ready to drag into a game engine. The browser downloads nothing. sync(prop, enabled) shows them for `prop` (null hides
 * them); enabled: false while its result is being made again. before(): awaited first (Projection: the painting
 * still on its way to the bake's files gets there).
 */
export function exportButtons(stage, { before = null } = {}) {
  let prop = null;
  let enabled = true;
  let saving = null;   // the format the hub is writing now
  const buttons = FORMATS.map(({ format, title }) => {
    const b = h('button', { type: 'button', class: 'btn', title, onclick: () => save(format) },
      icon('download'), spinner(), `.${format}`);
    b.dataset.format = format;
    return b;
  });
  const el = h('div', { class: 'sv-export', hidden: true },
    h('div', { class: 'sv-export-caption', title: 'Every prop gets its own folder in the project folder, and File Explorer '
      + 'opens there with the file selected: drag it into your game engine from there.' }, 'Save to the project folder'),
    h('div', { class: 'sv-export-btns' }, buttons));

  const paint = () => {
    for (const b of buttons) {
      b.disabled = !!saving || !enabled;
      b.classList.toggle('working', b.dataset.format === saving);
    }
  };

  async function save(format) {
    if (saving || !prop) return;
    const pid = projectId();
    const { id, name } = prop;
    saving = format;
    paint();
    try {
      await before?.();
      const answer = await api('POST', `${propApi(pid, id)}/export`, { stage, format });
      toastSaved(answer?.shown || answer?.files?.[0] || `${name}.${format}`, answer, triflowNote([prop], stage));
    } catch (e) {
      toast(e.message || String(e), 'error');
    } finally {
      saving = null;
      paint();
    }
  }

  el.sync = (p, on = true) => {
    prop = p;
    enabled = on;
    el.hidden = !p;
    paint();
  };
  return el;
}
