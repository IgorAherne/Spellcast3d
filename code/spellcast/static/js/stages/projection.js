// Stage 5, Projection: the high poly's textures are baked onto each prop's clean mesh (our CPU baker,
// run by Instant Meshes' process). The high poly is the 3D model, or its Omni-Texture while the prop is chosen for
// one and has a fresh one (rules.highPoly). The workbench shows the baked low poly - or the high poly, to compare
// up close - with the map bar, the bake settings, the bake's numbers and the [.glb] [.fbx] [.obj] buttons
// that save it into the project folder. Once a prop's bake is fresh, those lead and its settings fold
// away. "Export all" saves every prop's best result into that folder and opens it in File Explorer.
// The baked low poly can be painted in the viewport (paintbar.js): a marker and a clone stamp, saved into the
// bake's base colour; baking again starts from new textures, so it asks first when the prop was painted.

import { api, projectApi } from '../api.js';
import { h, plural } from '../dom.js';
import { icon } from '../icons.js';
import { boxesOf, getProp, jobFor, projectId } from '../store.js';
import { queueSettings, startJob } from '../actions.js';
import { field, segmented, sliderField, toggle } from '../controls.js';
import { attempt, confirmDialog } from '../ui.js';
import {
  afterMove, boxFile, boxHeading, busyLayer, discordCorner, donateCorner, emptyState, errorLine, fadeIn, focusBox, gateLine, gateOpen,
  jobState, rowAction, statsBlock, thumbUrl, toolGate,
} from './common.js';
import { bakeFresh, highPoly, omniChosen, omniOwn, omniTextured, omnitextureUsed } from './rules.js';
import { createPaintBar } from './paintbar.js';
import { DEFAULT_VIEW, exportButtons, modelPane, rememberWireframe, toastSaved, triflowNote, wireframeOn } from './preview.js';

// The ray distance slider, in % of the model's size (the hub takes 0.05-20: typed values stay valid).
const RAY_MIN = 0.1;
const RAY_MAX = 10;
const RAY_DEFAULT = 1;
const HARD_EDGE_DEFAULT = 60;

let host = null;
let ctx = null;
let view = null;          // the prop view on screen: {key, el, update(prop, auto), destroy()}
let shownId = null;       // the prop shown last, kept while nobody picks another
// Kept while going from prop to prop: which model is shown, and the map picked last (the base colour
// until the artist picks another).
let side = 'low';
let lastView = DEFAULT_VIEW;

/** Rounds to 2 significant digits, so the slider lands on friendly numbers (1.2 %, not 1.1873 %). */
const friendly = (v) => {
  const p = 10 ** (Math.floor(Math.log10(v)) - 1);
  return +(Math.round(v / p) * p).toPrecision(2);
};
const percent = (v, digits = 1) => `${(v * 100).toFixed(digits)} %`;

/**
 * What the panel says of a prop's bake: {model, projection: [label, value] rows, warnings}. The model first - the
 * vertices the saved files hold (split at the UV seams and hard edges) and its texture -, then how the projection went.
 */
export function bakeFacts(p) {
  const st = p.projection?.glb ? p.projection.stats || {} : {};
  return {
    model: [
      ['Vertices', typeof st.render_vertices === 'number' ? st.render_vertices.toLocaleString('en-US') : null],
      ['Texture', st.resolution ? `${st.resolution} × ${st.resolution} px` : null],
    ],
    projection: [
      ['Covered', typeof st.coverage === 'number' ? percent(st.coverage, 0) : null],
      ['Missed', typeof st.miss_fraction === 'number' ? percent(st.miss_fraction, 2) : null],
      ['Ray distance', typeof st.ray_distance_percent === 'number'
        ? `${st.ray_distance_percent.toFixed(2)} % of the size${p.projection.settings?.ray_distance == null ? ' (automatic)' : ''}` : null],
      ['Time', typeof st.seconds === 'number' ? `${st.seconds.toFixed(1)} s` : null],
    ],
    warnings: Array.isArray(st.warnings) ? st.warnings : [],
  };
}

const STALE = 'Retopo, 3D model or Omni-Texture changed: bake again';
/** A chosen prop whose Omni-Texture is not made: the row action and the call to action bake it once it is. */
const awaitsOmniTexture = (p) => omniChosen(p) && !omniTextured(p);

/** Why this prop can't be baked yet, or ''. */
function notReady(p) {
  if (!p.mesh?.glb) return 'This prop has no 3D model yet.';
  if (!p.retopo?.obj) return 'This prop has no clean mesh yet: retopologize it first.';
  if (p.retopo.stats?.has_uv === false) return 'The clean mesh has no UVs. Retopologize it again, then bake.';
  return '';
}

// ---- one prop -------------------------------------------------------------------------------------

function buildProp(prop) {
  const id = prop.id;
  const cur = () => getProp(id) || prop;
  const settings = () => cur().projection?.settings || {};
  const set = (patch) => queueSettings(id, 'projection', patch);
  const s = settings();

  // ---- left: preview -------------------------------------------------------------------------
  const sides = segmented([
    { value: 'high', label: 'High', title: 'The detailed 3D model (or its Omni-Texture), which the bake reads' },
    { value: 'low', label: 'Low', title: 'The clean mesh with the baked textures' },
  ], side, (v) => {
    side = v;
    renderMedia(cur());
  });
  const pane = modelPane({ tools: [sides], onView: (mode) => { lastView = mode; }, discord: true });
  const busy = busyLayer('projection');
  const paint = createPaintBar();
  pane.stage.append(busy, paint.el);
  const media = h('div', { class: 'sv-media' }, pane.el);

  /** What the preview shows now: {url, kind, extraMaps, snapshot, note} or {empty: <node>}. */
  function target(p) {
    if (side === 'high') {
      if (!p.mesh?.glb) return { empty: emptyState('cube', 'No 3D model yet', 'This prop becomes 3D in the 3D stage first.') };
      if (omnitextureUsed(p)) {
        return { url: boxFile(p, highPoly(p), 'omnitexture'), kind: 'glb', snapshot: thumbUrl(p, 'omnitexture'),
          note: omniOwn(p) ? 'Its own texture with what you painted on it: the colours the bake reads'
            : 'Its Omni-Texture: the colours the bake reads' };
      }
      return { url: boxFile(p, p.mesh.glb, 'mesh'), kind: 'glb', snapshot: thumbUrl(p, 'mesh') };
    }
    const baked = p.projection?.glb;
    if (baked) {
      const height = (p.projection.maps || []).find((m) => m.id === 'height');
      return {
        url: boxFile(p, baked, 'projection'), kind: 'glb', baked: true, snapshot: thumbUrl(p, 'projection'),
        extraMaps: height ? { height: { url: boxFile(p, height.path, 'projection'), colorspace: height.colorspace } } : {},
      };
    }
    if (!p.retopo?.obj) {
      return { empty: emptyState('grid', 'No clean mesh yet', 'The textures are baked onto the clean mesh: retopologize this prop first.',
        h('button', { class: 'btn ghost', onclick: () => ctx?.goto('retopo') }, icon('grid'), 'Go to Retopo')) };
    }
    return {
      url: boxFile(p, p.retopo.obj, 'retopo'), kind: 'obj', snapshot: thumbUrl(p, 'retopo'), note: 'Bake to see the textured low poly',
    };
  }

  let shown = null;
  let shownBake = null;     // the bake's GLB while the baked low poly is on screen (it can be painted then)
  // After the first model, High / Low (and a new bake) keep the camera: the artist compares up close.
  let framed = false;
  async function renderMedia(p) {
    const t = target(p);
    const stale = side === 'low' && !!t.baked && !bakeFresh(p);
    pane.badges(
      t.note ? h('span', { class: 'chip' }, icon('info'), t.note) : null,
      stale ? h('span', { class: 'chip warn', title: 'The 3D model, its Omni-Texture or the clean mesh changed after this bake. Bake again to update the textures.' },
        icon('alert'), STALE) : null);
    const key = t.url ? `${t.kind}|${t.url}` : `empty|${side}|${!p.mesh?.glb}`;
    if (key === shown) return;
    shown = key;
    shownBake = null;
    syncPaint(p);
    if (!t.url) {
      pane.message(t.empty);
      return;
    }
    pane.maps.offerWireframe(wireframeOn('projection', false), (on) => rememberWireframe('projection', on));   // off until asked
    const opts = { extraMaps: t.extraMaps, keepView: framed, view: lastView, ...afterMove(ctx, t.snapshot) };
    if (await pane.show(t.url, t.kind, opts)) {
      framed = true;
      if (shown === key && t.baked) shownBake = p.projection.glb;
      syncPaint(cur());
    }
  }

  /** The brushes are there while the baked low poly is on screen and no bake of this prop is queued or running. */
  function syncPaint(p) {
    const viewer = pane.viewer;
    if (!viewer || !shownBake || shownBake !== p.projection?.glb || jobFor('projection', id)) {
      paint.unbind();
      return;
    }
    paint.bind({ viewer, project: projectId(), propId: id, glb: shownBake, select: (mode) => pane.maps.select(mode) });
  }

  // ---- right: settings -----------------------------------------------------------------------
  const heading = boxHeading();
  const size = segmented([
    { value: 512, label: '512' }, { value: 1024, label: '1K' }, { value: 2048, label: '2K' }, { value: 4096, label: '4K' },
  ], s.texture_size ?? 2048, (v) => set({ texture_size: v }));
  const quality = segmented([
    { value: 1, label: 'Fast', title: '1 sample per texel' },
    { value: 2, label: 'Good', title: '4 samples per texel: smooth edges' },
    { value: 3, label: 'Best', title: '9 samples per texel: slowest' },
  ], s.quality ?? 2, (v) => set({ quality: v }));

  let manualRay = s.ray_distance ?? RAY_DEFAULT;
  const raySlider = sliderField('Distance', {
    value: manualRay,
    toPos: (v) => Math.round((Math.log(v / RAY_MIN) / Math.log(RAY_MAX / RAY_MIN)) * 1000),
    fromPos: (pos) => friendly(RAY_MIN * (RAY_MAX / RAY_MIN) ** (pos / 1000)),
    format: (v) => `${v} % of the model size`,
    onInput: (v) => { manualRay = v; set({ ray_distance: v }); },
  });
  raySlider.hidden = s.ray_distance == null;
  const rayMode = segmented([
    { value: 'auto', label: 'Automatic', title: 'Measured from how far the clean mesh lies from the detailed one' },
    { value: 'manual', label: 'Manual' },
  ], s.ray_distance == null ? 'auto' : 'manual', (v) => {
    raySlider.hidden = v === 'auto';
    if (v === 'manual') {
      // Start from what the last automatic bake measured, when there is one.
      const measured = cur().projection?.stats?.ray_distance_percent;
      if (settings().ray_distance == null && measured > 0) manualRay = friendly(Math.min(RAY_MAX, Math.max(RAY_MIN, measured)));
      raySlider.set(manualRay);
    }
    set({ ray_distance: v === 'auto' ? null : manualRay });
  });

  const ao = toggle(s.ao ?? true, (v) => set({ ao: v }), 'Ambient occlusion',
    'Bakes soft contact shadows into the AO map. Skipped when the model brings its own AO map.');
  const normalY = segmented([
    { value: 'opengl', label: 'OpenGL', title: 'Blender, Unity, Godot, glTF' },
    { value: 'directx', label: 'DirectX', title: 'Unreal, 3ds Max' },
  ], s.normal_y ?? 'opengl', (v) => set({ normal_y: v }));

  const basis = segmented([
    { value: 'mikk', label: 'MikkTSpace', title: 'Blender, Unreal, Unity HDRP, Substance' },
    { value: 'per_vertex', label: 'Per vertex', title: 'three.js, Unity URP, xNormal' },
  ], s.tangent_basis ?? 'mikk', (v) => set({ tangent_basis: v }));
  let hardAngle = s.hard_edge_angle >= 0 ? s.hard_edge_angle : HARD_EDGE_DEFAULT;
  const hardSlider = sliderField('Sharper than', {
    min: 0, max: 180, step: 5, value: hardAngle, format: (v) => `${v}°`,
    onInput: (v) => { hardAngle = v; set({ hard_edge_angle: v }); },
  });
  hardSlider.hidden = s.hard_edge_angle === -1;
  const hard = toggle(s.hard_edge_angle !== -1, (v) => {
    hardSlider.hidden = !v;
    set({ hard_edge_angle: v ? hardAngle : -1 });
  }, 'Hard edges at UV seams', 'Where a UV seam runs along a sharp bend, the shading breaks there, so the '
    + 'normal map has less to correct. Off keeps every edge smooth.');
  const repack = toggle(s.repack_uvs ?? true, (v) => set({ repack_uvs: v }), 'Repack UVs',
    'Lays the UV islands out again with room between them, so colours never bleed into a neighbour. The '
    + 'baked mesh then has its own UVs; the Retopo result keeps its own.');
  const height = toggle(!!s.height, (v) => set({ height: v }), 'Height map',
    'Also bakes how far the detailed surface stands out from the clean mesh (16-bit PNG).');
  const writeFbx = toggle(s.export_fbx ?? true, (v) => set({ export_fbx: v }), 'Also write FBX',
    'For Unity, Unreal, 3ds Max and Maya.');
  const writeObj = toggle(s.export_obj ?? true, (v) => set({ export_obj: v }), 'Also write OBJ',
    'An OBJ with its MTL and the maps, for programs that read neither GLB nor FBX.');

  const stats = h('div', { class: 'sv-group tight' });
  const exports = exportButtons('projection', { before: () => paint.flush() });

  let starting = false;   // until the job is queued: a double click must not queue it twice
  const bakeLabel = h('span');
  const bakeBtn = h('button', {
    class: 'btn primary big sv-wide',
    onclick: async () => {
      if (starting) return;
      const painted = cur().projection?.painted || paint.painted() || paint.pending();
      if (painted && cur().projection?.glb && !(await confirmDialog({
        title: 'Bake again?', ok: 'Bake again', danger: true,
        message: 'Baking again makes new textures from the 3D model, so the painting on this prop is replaced.',
      }))) return;
      if (starting) return;
      starting = bakeBtn.disabled = true;
      await attempt(() => startJob('projection', [id]));
      starting = false;
      sync(cur());
    },
  }, icon('sun'), bakeLabel);
  const why = h('div', { class: 'sv-why', hidden: true });
  const gate = gateLine('projection');
  const error = errorLine('projection');
  const foot = h('div', { class: 'sv-foot' }, error, gate, why, exports, bakeBtn);
  // While the prop has a fresh bake, its settings fold away (open them to bake again differently).
  const settingsTitle = h('summary', null, 'Bake settings');
  const bakeSettings = h('details', { class: 'fold sv-settings', open: true },
    settingsTitle,
    h('div', { class: 'sv-group' },
      field('Texture size', size, 'Width and height of every baked map, in pixels.'),
      field('Quality', quality, 'More samples per texel give cleaner edges and take longer.'),
      field('Ray distance', h('div', { class: 'sv-group tight' }, rayMode, raySlider),
        'How far from the clean mesh the bake looks for the detailed surface. Raise it when parts come out '
        + 'empty or smeared; lower it when a neighbouring part shows through.'),
      ao,
      field('Normal map for', normalY, 'OpenGL: Blender, Unity, Godot, glTF. DirectX: Unreal, 3ds Max.'),
      h('details', { class: 'fold' },
        h('summary', null, 'Advanced'),
        h('div', { class: 'sv-group' },
          field('Tangent basis', basis, 'MikkTSpace matches Blender, Unreal, Unity HDRP and Substance; per vertex '
            + 'matches three.js, Unity URP and xNormal. The preview shows either one correctly.'),
          hard, hardSlider, repack, height, writeFbx, writeObj))));
  const sidePanel = h('div', { class: 'sv-side' },
    h('div', { class: 'sv-side-scroll' }, heading, stats, bakeSettings),
    foot);
  const el = h('div', { class: 'sv sv-projection' }, media, sidePanel);

  /** The bake's numbers under their headings: the model it made, then how the projection went. */
  function renderStats(p) {
    const facts = bakeFacts(p);
    const group = (title, rows, warnings = []) => {
      const block = statsBlock(rows, warnings);
      return block ? [h('div', { class: 'field-label' }, h('span', null, title)), block] : [];
    };
    stats.replaceChildren(...group('Model', facts.model), ...group('Projection', facts.projection, facts.warnings));
  }

  let wasFresh = null;
  function sync(p, auto = false) {
    const j = jobFor('projection', id);
    const reason = notReady(p);
    bakeBtn.disabled = starting || !!j || !!reason || !gateOpen(toolGate('projection'));
    bakeBtn.classList.toggle('working', !!j);
    bakeLabel.textContent = j ? (j.phase === 'running' ? 'Baking…' : 'Waiting…')
      : p.projection?.glb ? 'Bake again' : 'Bake textures';
    // A fresh bake: its export buttons lead, baking again is there quietly, the settings fold away.
    const fresh = !j && bakeFresh(p);
    exports.sync(p.projection?.glb ? p : null, !j);
    bakeBtn.classList.toggle('primary', !fresh);
    bakeBtn.classList.toggle('big', !fresh);
    bakeBtn.classList.toggle('ghost', fresh);
    settingsTitle.hidden = !fresh;
    if (fresh !== wasFresh) bakeSettings.open = !fresh;
    wasFresh = fresh;
    why.hidden = !reason || !!j;
    why.textContent = reason;
    gate.sync();
    error.sync(p);
    busy.sync(p);
    heading.set(p, auto && j?.phase === 'running' ? 'Following the prop being baked' : '',
      thumbUrl(p, 'projection') || thumbUrl(p, 'mesh') || (p.images?.chosen ? thumbUrl(p, 'images') || boxFile(p, p.images.chosen, 'images') : null));
  }

  const update = (p, auto) => {
    renderMedia(p);
    renderStats(p);
    sync(p, auto);
    syncPaint(p);
  };

  return {
    key: id, el, update,
    flush: () => paint.flush(),
    destroy: () => {
      paint.destroy();
      pane.destroy();
    },
  };
}

function buildEmpty() {
  const el = h('div', { class: 'sv sv-projection sv-single' },
    h('div', { class: 'sv-stagebox' }, emptyState('sun', 'No props here yet',
      'Props arrive here once they have a clean mesh: the detailed model’s textures are baked onto it, ready '
      + 'for your game engine.',
      h('button', { class: 'btn ghost', onclick: () => ctx?.goto('images') }, icon('image'), 'Go to Images')),
    donateCorner(), discordCorner()));
  return { key: 'empty', el, update() {}, destroy() {} };
}

// ---- per prop, for the props list -----------------------------------------------------------------

function status(prop) {
  const job = jobState('projection', prop);
  if (job) return job;
  if (prop.projection?.glb) {
    return bakeFresh(prop) ? { state: 'done', message: 'Has its baked textures' } : { state: 'stale', message: STALE };
  }
  if (!prop.retopo?.obj || !prop.mesh?.glb) {
    return ['images', 'mesh', 'retopo'].some((s) => jobFor(s, prop.id)) ? { state: 'waiting', message: 'Waiting for its clean mesh' }
      : { state: 'unreachable', message: notReady(prop) };
  }
  if (prop.retopo.stats?.has_uv === false) return { state: 'unreachable', message: notReady(prop) };
  if (jobFor('omnitexture', prop.id)) return { state: 'waiting', message: 'Waiting for its Omni-Texture' };
  if (awaitsOmniTexture(prop) && gateOpen(toolGate('omnitexture'))) {
    return { state: 'idle', message: 'Not baked yet: chosen for Omni-Texture, which is not made yet' };
  }
  return { state: 'idle', message: 'Not baked yet' };
}

/** Some prop has something to export: a bake, a clean mesh or a 3D model. */
const exportable = () => boxesOf().some((b) => b.projection?.glb || b.retopo?.obj || b.mesh?.glb);

// ---- the stage view -------------------------------------------------------------------------------

function render() {
  if (!host) return;
  const { box: prop, auto } = focusBox(ctx, 'projection', { needsWork: (b) => !notReady(b) && !bakeFresh(b), previousId: shownId });
  shownId = prop?.id ?? null;
  if (view?.key !== (prop ? prop.id : 'empty')) {
    view?.destroy();
    view = prop ? buildProp(prop) : buildEmpty();
    host.replaceChildren(view.el);
    fadeIn(view.el);
  }
  if (prop) view.update(prop, auto);
}

export default {
  id: 'projection',
  title: 'Projection',
  ctaLabel: 'Export all',
  ctaIcon: 'folder',
  ctaHover: 'Saves every prop’s finished model, each prop in its own folder, and opens the project folder, ready to drag into your game engine.',
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

  status,

  /** The prop the workbench shows. */
  shownId: () => shownId,

  ctaEnabled() {
    return exportable() ? { enabled: true } : { enabled: false, hint: 'Nothing to export yet: bake a prop first.' };
  },

  /**
   * Saves every prop's best result (the bake, else the clean mesh, else the 3D model) into the project's
   * export folder, each prop in its own folder there; the hub opens the project's folder in File Explorer
   * and the toast says which were made with TriFlow (non-commercial use only). A failure reaches the
   * shell, which toasts it.
   */
  async onCta() {
    await view?.flush?.();   // the painting still on its way into the bake's files gets there first
    const answer = await api('POST', `${projectApi(projectId())}/export-all`, {});
    toastSaved(plural(answer?.props ?? boxesOf().length, 'prop'), answer, triflowNote(boxesOf(), 'all'));
  },

  /** "Bake 4 props": every clean mesh without a fresh bake, baked right here. */
  rowAction: () => rowAction('projection', shownId, 'Bake', 'sun'),
};
