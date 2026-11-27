// Stage 4, Retopo: each prop's 3D model is rebuilt as clean, light meshes with UVs, one per size the prop asks for
// (the boxes on its tile on the left: thousands of vertices, 0.25 for 250, up to three; tileSizes / sizeTile), and the
// artist picks the one the prop keeps. Each prop picks its method at the top of the workbench (every new prop starts
// with TriFlow, whatever the prop before it picked): TriFlow or LATO.2 (AI tools on the graphics card) or Instant
// Meshes (the processor), whose interactive viewer, a click away, also shapes a mesh by hand with guide strokes - its
// Accept keeps that one. TriFlow is free for non-commercial use only, and says so wherever it is picked. A size
// outside the vertices a method places is made with the nearest count it places, and the page says so.
// Nothing is retopologized by itself: "Retopo All", in the viewport's lower-left corner, makes the clean meshes the
// props lack, one job per method (rules.retopoPlan), and the tool of its first job loads as the stage opens
// (warmWish). The viewport shows a clean mesh with its edges drawn (the topology is what the artist judges; the
// Wireframe toggle under it takes them away); the boxes at its bottom, one per size, show another one and pick it -
// the one picked has a green check. A prop with one size has it picked as soon as it is made. The call to action
// goes on to Projection once every prop has a picked, up-to-date clean mesh. A method's clean meshes stay with the
// prop while another method is picked: back at its method, they are there again.

import { api, localUrl, propApi } from '../api.js';
import { h, plural } from '../dom.js';
import { icon } from '../icons.js';
import { ACTIVE_JOB, boxesOf, bump, getProp, jobFor, projectId, store, toolName, toolStatus } from '../store.js';
import { pollStatusSoon, queueSettings, replaceProp, startJob } from '../actions.js';
import { field, segmented, seedInput, sliderField, spinner } from '../controls.js';
import { attempt, confirmDialog, toast } from '../ui.js';
import { licenseChip, licenseLine } from '../licenses.js';
import {
  afterMove, boxFile, boxHeading, busyLayer, carryDown, donateCorner, emptyState, errorLine, fadeIn, focusBox, gateLine,
  gateOpen, jobState, liveDeps, thumbUrl, toolGate,
} from './common.js';
import { modelPane, rememberWireframe, wireframeOn } from './preview.js';
import {
  RETOPO_SIZES, planCount, planSummary, retopoFresh, retopoMethod, retopoMissing, retopoPlan, retopoReady,
  retopoSizeName, retopoSizes, retopoSwitched, retopoUsed, retopoVariant, retopoVertices, tileable,
} from './rules.js';

// The methods, in the picker's order: TriFlow, the one a prop starts with, first. Their hints say what each
// is like (on hover).
const METHODS = [
  { value: 'triflow', label: 'TriFlow',
    title: 'AI retopology that lays the surface out in quad-like pairs of triangles. Runs on the graphics card, about a '
      + 'minute per clean mesh, and needs about 10 GB of video memory. Free for non-commercial use.' },
  { value: 'lato2', label: 'LATO.2',
    title: 'AI retopology: an even, clean triangle mesh. Runs on the graphics card, about a minute per clean mesh.' },
  { value: 'imesh', label: 'Instant Meshes',
    title: 'Quads, which you can also shape by hand with guide strokes in its interactive viewer. Runs on the processor.' },
];
const TRIFLOW_QUADS = 0.95;
// TriFlow's peak video memory was about 10 GB on a 12 GB card; below this (an 8 GB card) the page says it may not fit.
const TRIFLOW_VRAM_MB = 9000;
const SIZES_TITLE = 'How many thousand vertices its clean meshes get (0.25 is 250): one for each number, then you '
  + 'pick one. Leave a box empty for fewer.';
const CTA_HOVER = 'Bakes the detail and colours of every 3D model (or its Omni-Texture) onto its picked clean mesh, in '
  + 'one pass.';

let host = null;
let ctx = null;
let view = null;          // the prop view on screen: {key, id, el, update(prop, auto), destroy()}
let shownId = null;       // the prop shown last, kept while nobody picks another
let viewerFor = null;     // the prop whose Instant Meshes viewer is open in the workbench (its "Shape it by hand")

/** The method that made the prop's clean meshes (Instant Meshes for results older than the choice). */
const madeWith = (prop) => prop.retopo?.method || 'imesh';

/** A retopo job waits or runs. */
const working = () => (store.jobs || []).some((j) => j.stage === 'retopo' && ACTIVE_JOB.has(j.status));

/** 45 -> "45 s", 635 -> "10 min 35 s". */
const duration = (seconds) => {
  const s = Math.round(seconds);
  return s < 120 ? `${s} s` : `${Math.floor(s / 60)} min${s % 60 ? ` ${s % 60} s` : ''}`;
};

/** "4k", "4k and 22k", "0.25k, 22k and 6k". */
const sizeList = (sizes) => {
  const names = sizes.map(retopoSizeName);
  return names.length > 1 ? `${names.slice(0, -1).join(', ')} and ${names.at(-1)}` : names.join('');
};

/**
 * How `method` makes a clean mesh of `size` that is outside the vertices it places (rules.retopoVertices): {count:
 * the vertices it is asked for instead, side: 'at least' | 'at most'}, or null when it is asked for the size's own.
 */
function outside(method, size) {
  const count = retopoVertices(method, size);
  const own = Math.round(1000 * size);
  return count === own ? null : { count, side: count > own ? 'at least' : 'at most' };
}

/**
 * What the panel says quietly of the sizes `method` makes with another count than their own, '' when there are none:
 * "LATO.2 places 10,000 vertices at most, so 22k is made with that many."
 */
function boundsLine(method, sizes) {
  return ['at least', 'at most'].map((side) => {
    const off = sizes.filter((size) => outside(method, size)?.side === side);
    if (!off.length) return '';
    return `${toolName(method)} places ${outside(method, off[0]).count.toLocaleString()} vertices ${side}, so `
      + `${sizeList(off)} ${off.length === 1 ? 'is' : 'are'} made with that many.`;
  }).filter(Boolean).join(' ');
}

/** A field whose explanation shows on hover (the panel stays quiet). */
const quietField = (label, control, title) => {
  const el = field(label, control);
  el.title = title;
  return el;
};

/**
 * The viewer's address as this page reaches it (localUrl). Its Accept button talks only to the page
 * named in host_origin (the hub fills it in): that is this page, whatever address it was opened at.
 */
function viewerUrl(url) {
  const u = new URL(localUrl(url), location.href);
  if (u.searchParams.has('host_origin')) u.searchParams.set('host_origin', location.origin);
  return u.href;
}

/**
 * The props the stage is about: objects with a 3D model, or one being made. One without has nothing to make clean
 * (a tileable texture never has): it holds nobody back.
 */
const inStage = (p) => !tileable(p) && (!!p.mesh?.glb || !!jobFor('mesh', p.id));

/**
 * Why the call to action holds the props back: how many still need a pick, their clean meshes, or wait for the ones
 * being made ('' when none does).
 */
export function waitingHint(props) {
  const busy = props.filter((p) => jobFor('retopo', p.id)).length;
  const pick = props.filter((p) => !jobFor('retopo', p.id) && retopoFresh(p) && !retopoReady(p)).length;
  const make = props.filter((p) => !jobFor('retopo', p.id) && !retopoFresh(p)).length;
  const parts = [
    pick ? `${plural(pick, 'prop')} ${pick === 1 ? 'needs' : 'need'} a pick (click a size under the model)` : '',
    make ? `${plural(make, 'prop')} ${make === 1 ? 'needs its' : 'need their'} clean meshes (Retopo All)` : '',
    busy ? `${plural(busy, 'prop')} ${busy === 1 ? 'is' : 'are'} being retopologized` : '',
  ].filter(Boolean);
  return parts.length ? `Every prop needs a picked clean mesh first: ${parts.join(', ')}.` : '';
}

// ---- the method picker over both kinds of view ------------------------------------------------------

/**
 * "Retopologize with [TriFlow | LATO.2 | Instant Meshes]" at the top of the workbench, saved per prop
 * (retopo.settings.method), then `extra` (the viewer's way back). Each method keeps the clean meshes it made for the
 * prop (rules.retopoSwitched: those under 50,000 vertices), so picking a method again shows them, their numbers and
 * their pick again without making them again. It can't change while the prop's retopo is queued or running, and a
 * method whose tool can't run here (too old a graphics card, a failed install) can't be picked: its button says why
 * on hover. sync(prop).
 */
function methodBar(id, ...extra) {
  const seg = segmented(METHODS.map((m) => ({
    value: m.value, title: m.title, label: [m.label, licenseChip(m.value)],
  })), retopoMethod(getProp(id)), (method) => {
    const p = getProp(id);
    if (!p || retopoMethod(p) === method) return;
    // The clean meshes this method made before are the prop's again, on screen at once (the hub does the same when
    // the method is saved); the thumbnail shows the other method's, so it is drawn again.
    const retopo = retopoSwitched(p.retopo, method);
    if (retopo !== p.retopo) {
      p.retopo = retopo;
      if (p.thumbs) delete p.thumbs.retopo;
    }
    queueSettings(id, 'retopo', { method });
    ctx?.refresh?.();
  });
  seg.classList.add('rt-methods');
  seg.setAttribute('aria-label', 'Retopology method');
  const el = h('div', { class: 'rt-bar' }, h('span', { class: 'rt-bar-label' }, 'Retopologize with'), seg, extra);
  el.sync = (p) => {
    const method = retopoMethod(p);
    seg.set(method);
    const busy = !!jobFor('retopo', p.id);
    seg.buttons.forEach((b, i) => {
      const m = METHODS[i];
      const gate = toolGate('retopo', m.value);
      const on = m.value === method;
      b.disabled = !on && (busy || !gateOpen(gate));
      b.title = !on && busy ? 'This prop is being retopologized: stop it first to pick another method.'
        : !gateOpen(gate) ? gate.text : gate.kind === 'installing' ? `${m.title}\n${gate.text}` : m.title;
    });
  };
  return el;
}

// ---- "Retopo All", in the viewport's lower-left corner ------------------------------------------------

/**
 * The button that starts the jobs for every prop that lacks a clean mesh of one of its sizes (rules.retopoPlan, one
 * job per method); "… again" once none does (asked first: every clean mesh is made anew).
 */
function goButton() {
  const words = h('span');
  const count = h('span', { class: 'go-count' });
  const glyph = h('span', { class: 'rt-glyph' });
  let starting = false;
  const btn = h('button', { type: 'button', class: 'btn primary rt-go', onclick: () => go() }, glyph, words, count);

  async function go() {
    if (starting) return;
    const deps = liveDeps();
    let plan = retopoPlan(boxesOf(), deps);
    if (!plan.length) {
      plan = retopoPlan(boxesOf(), deps, { again: true });
      const n = planCount(plan, 'retopo');
      if (!n || !(await confirmDialog({
        title: `Retopologize ${plural(n, 'prop')} again?`, ok: 'Make them again',
        message: 'Every prop has its clean meshes already. Making them again replaces them; a prop with several '
          + 'sizes then needs a pick again.',
      }))) return;
    }
    starting = true;
    sync();
    await attempt(async () => {
      for (const [stage, ids, tool] of plan) await startJob(stage, ids, { tool });
    });
    starting = false;
    sync();
  }

  function sync() {
    const deps = liveDeps();
    const todo = retopoPlan(boxesOf(), deps);
    const again = todo.length ? [] : retopoPlan(boxesOf(), deps, { again: true });
    const busy = starting || (!todo.length && working());
    const n = planCount(todo.length ? todo : again, 'retopo');
    btn.disabled = busy || !n;
    btn.classList.toggle('working', busy);
    words.textContent = busy ? 'Retopologizing…' : todo.length || !again.length ? 'Retopo All' : 'Retopo All again';
    count.textContent = n && !busy ? String(n) : '';
    count.hidden = !count.textContent;
    const shown = busy ? 'working' : 'grid';
    if (glyph.dataset.icon !== shown) {
      glyph.dataset.icon = shown;
      glyph.replaceChildren(busy ? spinner() : icon('grid'));
    }
    btn.title = busy ? 'The props are being retopologized. Their tiles say how far each one is.'
      : todo.length ? `${planSummary(todo, toolName)} Each gets a clean mesh of every size in the boxes on its `
        + 'tile that it lacks.'
        : n ? `${planSummary(again, toolName)} Every clean mesh of theirs is made anew.`
          : boxesOf().some((p) => p.mesh?.glb) ? 'Type a size in the boxes on the props on the left first.'
            : 'The props need their 3D models first: make them in 3D.';
  }
  btn.sync = sync;
  return btn;
}

// ---- the clean meshes of the prop shown, at the bottom of the viewport ----------------------------------

/**
 * One box per size the prop asks for (and the one it uses, when its size is no longer asked for): "4k" over its
 * vertex count. A click on one that is made shows it in the viewport (the camera stays) and picks it (show(size),
 * then the hub's pick); the one picked has a green check, one not made yet (or made from an older 3D model) is shown
 * as such and can't be picked. A size the method makes with another count says so on hover. sync(prop, shownSize).
 */
function variantSwitch(id, show) {
  const el = h('div', { class: 'rt-variants', role: 'radiogroup', 'aria-label': 'This prop’s clean meshes' });
  let picking = null;     // the size being picked, until the hub has answered
  let drawn = null;

  async function pick(size) {
    const p = getProp(id);
    show(size);
    if (!p || picking !== null || (retopoReady(p) && retopoUsed(p) === size)) return;
    picking = size;
    sync(p, size);
    try {
      replaceProp(await api('POST', `${propApi(projectId(), id)}/retopo/pick`, { size }));
      delete store.jobErrors[`retopo:${id}`];
      if (boxesOf().every(retopoReady)) ctx?.nudgeCta?.();
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      picking = null;
      sync(getProp(id) || p, size);
    }
  }

  function sync(p, shownSize) {
    const fresh = retopoFresh(p);
    const used = retopoUsed(p);
    const sizes = retopoSizes(p);
    const boxes = used !== null && !sizes.includes(used) ? [...sizes, used] : sizes;
    const method = retopoMethod(p);
    const key = JSON.stringify([boxes, boxes.map((size) => retopoVariant(p, size)), fresh, used, !!p.retopo?.accepted,
      shownSize, picking, method]);
    el.hidden = !boxes.length || !p.mesh?.glb;
    if (key === drawn) return;
    drawn = key;
    el.replaceChildren(...boxes.map((size) => {
      const made = fresh ? retopoVariant(p, size) : null;
      const picked = made && size === used && !!p.retopo.accepted;
      const vertices = made?.stats?.vertices;
      const name = retopoSizeName(size);
      const off = outside(method, size);
      const capped = off ? ` ${toolName(method)} makes ${retopoSizeName(off.count / 1000)} vertices ${off.side}: this `
        + `size is made with ${retopoSizeName(off.count / 1000)}.` : '';
      const btn = h('button', {
        type: 'button', class: `rt-variant${size === shownSize && made ? ' shown' : ''}${picked ? ' picked' : ''}`,
        role: 'radio', 'aria-checked': String(!!picked), disabled: !made || picking !== null,
        title: !made ? `${name} vertices: not made yet. Retopo All makes it.${capped}`
          : `${name} vertices${typeof vertices === 'number' ? ` (${vertices.toLocaleString()} made)` : ''}. `
            + `${picked ? 'Picked: the next stages use this one.' : 'Click to see it and pick it.'}${capped}`,
        onclick: () => pick(size),
      }, h('span', { class: 'rt-variant-size' }, name),
      h('small', null, !made ? 'not made' : typeof vertices === 'number' ? vertices.toLocaleString() : 'made'),
      picked ? h('span', { class: 'rt-variant-check', 'aria-hidden': 'true' }, icon('check'))
        : picking === size ? h('span', { class: 'rt-variant-check waiting', 'aria-hidden': 'true' }, spinner()) : null);
      return btn;
    }));
  }

  el.sync = sync;
  return el;
}

// ---- one prop: its clean meshes ---------------------------------------------------------------------

function buildProp(prop) {
  const id = prop.id;
  const cur = () => getProp(id) || prop;
  const set = (patch) => queueSettings(id, 'retopo', patch);
  const s = cur().retopo?.settings || {};
  let destroyed = false;

  const bar = methodBar(id);
  let side = 'low';         // the clean mesh or the 3D model (each prop and each new clean mesh: the clean mesh)
  let shownSize = retopoUsed(prop);   // the clean mesh on screen, by its size (the one the prop uses at first)

  // ---- left: a clean mesh (or the 3D model), the sizes under it, "Retopo All" -------------------------
  const sides = segmented([
    { value: 'high', label: 'High', title: 'The detailed 3D model the clean meshes are made from' },
    { value: 'low', label: 'Low', title: 'The clean mesh, with its edges drawn' },
  ], side, (v) => {
    side = v;
    renderMedia(cur());
  });
  // No map buttons: this stage is about the edges, and buttons only the 3D model has would resize the view at
  // each High / Low (the camera stays put for the comparison). The Wireframe toggle is there for both sides, far
  // from them, under the viewport: the pointer over High or Low rings it, as Images' buttons ring the Resolution.
  const pane = modelPane({ tools: [sides], maps: false, wideLights: true });
  for (const button of sides.buttons) button.addEventListener('pointerenter', () => pane.maps.ringWireframe());
  const busy = busyLayer('retopo');
  const variants = variantSwitch(id, (size) => {
    shownSize = size;
    side = 'low';
    sides.set('low');
    renderMedia(cur());
  });
  const go = goButton();
  pane.stage.append(variants, busy, go);
  const media = h('div', { class: 'sv-media' }, pane.el);

  /** The clean mesh on screen: the one picked in the switch, while it is there; else the one the prop uses. */
  const shownMesh = (p) => (retopoFresh(p) ? retopoVariant(p, shownSize) : null)
    || (p.retopo?.obj ? { size: retopoUsed(p), obj: p.retopo.obj, stats: p.retopo.stats } : null);

  let shown = null;
  let framed = false;   // after the first model, High / Low and the sizes keep the camera: the artist compares up close
  let wireAgain = true; // the prop just picked: its Low side shows its edges again, whatever was chosen before
  let result = null;    // the clean meshes on file: a new one is shown first (the artist judges its edges)
  async function renderMedia(p) {
    const made = p.retopo?.obj ? (p.retopo.variants || []).map((v) => boxFile(p, v.obj, 'retopo')).join() : null;
    if (made !== result) {
      if (result !== null && made) {
        side = 'low';
        sides.set('low');
        shownSize = retopoUsed(p);
      }
      result = made;
    }
    const mesh = shownMesh(p);
    const low = !!mesh && side === 'low';
    sides.hidden = !mesh || !p.mesh?.glb;
    variants.sync(p, low ? mesh.size : null);
    const method = retopoMethod(p);
    const fresh = retopoFresh(p);
    pane.badges(
      !mesh && p.mesh?.glb ? h('span', { class: 'chip', title: 'The detailed 3D model. Press “Retopo All” to make its clean meshes.' },
        icon('info'), 'Not retopologized yet') : null,
      fresh && !p.retopo.accepted && !jobFor('retopo', p.id)
        ? h('span', { class: 'chip rt-pick', title: 'The next stages use the clean mesh picked for each prop.' },
          icon('info'), 'Pick one: click a size under the model') : null,
      low && !fresh && madeWith(p) !== method
        ? h('span', { class: 'chip warn', title: `“Retopo All” replaces it with ${toolName(method)}’s.` },
          icon('alert'), `Made with ${toolName(madeWith(p))}`)
        : low && !fresh ? h('span', { class: 'chip warn', title: 'The 3D model changed after this clean mesh was made. Make it again to update it.' },
          icon('alert'), 'Made from an older 3D model') : null);
    const url = low ? boxFile(p, mesh.obj, 'retopo') : p.mesh?.glb ? boxFile(p, p.mesh.glb, 'mesh') : null;
    const key = url ? `${low ? 'obj' : 'glb'}|${url}` : 'none';
    if (key === shown) return;
    shown = key;
    if (!url) {
      pane.message(emptyState('cube', 'No 3D model yet', 'This prop gets its 3D model in the 3D stage first.',
        h('button', { class: 'btn ghost', onclick: () => ctx?.goto('mesh') }, icon('cube'), 'Go to 3D')));
      return;
    }
    // One Wireframe choice for both sides; both shown lit ("all"): the clean mesh is clay, and the 3D model in its
    // lit render reads the same way beside it (an unlit map view looked darker than the clay next to it).
    if (low && wireAgain) {
      rememberWireframe('retopo', true);
      wireAgain = false;
    }
    pane.maps.offerWireframe(wireframeOn('retopo', true), (on) => rememberWireframe('retopo', on));
    // While the page moves, the snapshot of what is shown stands in: the retopo one is of the clean mesh the prop uses.
    const snapshot = !low ? thumbUrl(p, 'mesh') : mesh.obj === p.retopo?.obj ? thumbUrl(p, 'retopo') : null;
    const opts = { keepView: framed, view: 'all', ...afterMove(ctx, snapshot) };
    if (await pane.show(url, low ? 'obj' : 'glb', opts) && !destroyed) framed = true;
  }

  // ---- right: the prop, its method's settings, the shown clean mesh's numbers --------------------------
  const heading = boxHeading();
  const bounds = h('p', { class: 'hint rt-cap', hidden: true });   // the sizes the method makes with another count
  const lato2 = h('div', { class: 'sv-group' },
    quietField('Seed', seedInput(s.lato2_seed ?? 0, (v) => set({ lato2_seed: v }), { randomMeansMinusOne: false }),
      'Change it for a different take on the same model.'));
  // TriFlow's terms, where it is picked (licenses.js), and on a card with little video memory a word about that.
  const terms = h('div', { class: 'rt-terms' });
  const vramNote = h('div', { class: 'rt-vram', hidden: true }, icon('info'), h('span'));
  const triflow = h('div', { class: 'sv-group' }, terms, vramNote,
    Object.assign(sliderField('Quad pairing', {
      min: 0.05, max: 1, step: 0.05, value: Math.max(0.05, s.triflow_quad_ratio ?? TRIFLOW_QUADS),
      format: (v) => `${Math.round(v * 100)} %`,
      onInput: (v) => set({ triflow_quad_ratio: +v.toFixed(2) }),
    }), { title: 'How strongly TriFlow lines its triangles up in pairs that read as quads. The mesh stays triangles.' }),
    quietField('Seed', seedInput(s.triflow_seed ?? 0, (v) => set({ triflow_seed: v }), { randomMeansMinusOne: false }),
      'Change it for a different take on the same model.'));
  // Instant Meshes: its own settings are in its viewer, which also shapes a mesh by hand.
  const handBtn = h('button', {
    class: 'btn ghost sv-wide', onclick: () => {
      viewerFor = id;
      render();
    },
  }, icon('pencil'), 'Shape it by hand in the viewer');
  const imesh = h('div', { class: 'sv-group' },
    h('p', { class: 'hint' }, 'Its viewer sets the polygon count and the kind of mesh, and takes guide strokes '
      + 'painted on the model. Its Accept keeps the mesh shaped there as one of this prop’s clean meshes, picked.'),
    handBtn);
  const madeNote = h('div', { class: 'sv-warning', hidden: true }, icon('alert'), h('span'));
  const infoNote = h('p', { class: 'hint rt-info', hidden: true });   // what the tool says about a result, no alarm
  const changedNote = h('p', { class: 'hint rt-changed', hidden: true }, 'Settings changed: make it again to use them.');
  const sizesNote = h('p', { class: 'hint rt-sizes', hidden: true },
    'Type how many thousand vertices its clean meshes get in the boxes on its tile on the left.');

  let starting = false;    // until the job is queued: a double click must not queue it twice
  const makeLabel = h('span');
  const makeBtn = h('button', { class: 'btn primary big sv-wide', onclick: () => make() }, icon('grid'), makeLabel);
  const gate = gateLine('retopo');
  const error = errorLine('retopo');
  const foot = h('div', { class: 'sv-foot' }, error, gate, changedNote, makeBtn);
  const sidePanel = h('div', { class: 'sv-side' },
    h('div', { class: 'sv-side-scroll' }, heading, madeNote, infoNote, sizesNote, bounds, lato2, triflow, imesh),
    foot);
  const el = h('div', { class: 'sv sv-retopo' }, bar, h('div', { class: 'rt-body' }, media, sidePanel));

  async function make() {
    if (starting) return;
    starting = makeBtn.disabled = true;
    await attempt(() => startJob('retopo', [id], { tool: retopoMethod(cur()) }));
    starting = false;
    sync(cur());
  }

  /**
   * "2,004 vertices · 3,998 triangles · Made in 12 s": a clean mesh's numbers on one line, under the prop's name.
   * Each part keeps together (no-break spaces): a narrow panel breaks the line only between them.
   */
  function statsLine(p, mesh) {
    const st = mesh?.stats || {};
    const faces = madeWith(p) === 'imesh' ? 'polygons' : 'triangles';   // Instant Meshes' are mostly quads
    return [
      typeof st.vertices === 'number' ? `${st.vertices.toLocaleString()} vertices` : '',
      typeof st.faces === 'number' ? `${st.faces.toLocaleString()} ${faces}` : '',
      typeof st.seconds === 'number' ? `Made in ${duration(st.seconds)}` : '',
    ].filter(Boolean).map((part) => part.replace(/ /g, ' ')).join(' · ');
  }

  function sync(p, auto = false) {
    const method = retopoMethod(p);
    const j = jobFor('retopo', id);
    const sizes = retopoSizes(p);
    const missing = retopoMissing(p);
    const mesh = shownMesh(p);
    bar.sync(p);
    lato2.hidden = method !== 'lato2';
    triflow.hidden = method !== 'triflow';
    imesh.hidden = method !== 'imesh';
    handBtn.disabled = !p.mesh?.glb || !!j || !['ok', 'unknown'].includes(toolGate('retopo', 'imesh').kind);
    bounds.textContent = boundsLine(method, sizes);
    bounds.hidden = !bounds.textContent;
    sizesNote.hidden = sizes.length > 0;
    makeBtn.hidden = !p.mesh?.glb;
    makeBtn.disabled = starting || !!j || !sizes.length || !gateOpen(toolGate('retopo', method));
    makeBtn.classList.toggle('working', !!j);
    makeLabel.textContent = j ? (j.phase === 'running' ? 'Making the clean meshes…' : 'Waiting…')
      : !missing.length ? 'Make it again'
        : missing.length === sizes.length ? (sizes.length === 1 ? 'Make clean mesh' : 'Make clean meshes')
          : `Make ${sizeList(missing)}`;
    makeBtn.title = !missing.length ? 'Makes every clean mesh of this prop anew (with its settings as they are now).'
      : `Makes this prop’s clean mesh${missing.length === 1 ? '' : 'es'} of ${sizeList(missing)} vertices.`;
    // Once the settings changed since these clean meshes were made, making them again leads, and says why.
    const changed = !j && !missing.length && settingsChanged(p);
    changedNote.hidden = !changed;
    const quiet = !j && !missing.length && !changed;
    makeBtn.classList.toggle('primary', !quiet);
    makeBtn.classList.toggle('big', !quiet);
    makeBtn.classList.toggle('ghost', quiet);
    gate.sync(method);
    // Its tool loading for this prop: the busy layer over the model says so already.
    gate.hidden ||= !p.mesh?.glb || (!!j && gate.classList.contains('loading'));
    if (method === 'triflow' && !terms.firstChild) terms.replaceChildren(licenseLine('triflow') || '');
    const vram = store.status?.gpu?.vram_total_mb;
    vramNote.hidden = method !== 'triflow' || !vram || vram >= TRIFLOW_VRAM_MB;
    if (!vramNote.hidden) {
      vramNote.lastChild.textContent = 'TriFlow can use about 10 GB of video memory on big props, and this graphics '
        + `card has ${Math.round(vram / 1024)} GB: it may be slow or run out. LATO.2 and Instant Meshes need less.`;
    }
    error.sync(p);
    busy.sync(p);
    go.sync();
    // What the tool said about the clean mesh on screen, or that it has no UVs (it can't be baked then).
    const st = mesh?.stats || {};
    const warning = [st.has_uv === false ? 'The clean mesh has no UVs, so it can’t be baked. Make it again.' : '',
      typeof st.warning === 'string' ? st.warning : ''].filter(Boolean).join(' ');
    madeNote.hidden = !warning;
    madeNote.lastChild.textContent = warning;
    infoNote.hidden = typeof st.note !== 'string' || !st.note;
    infoNote.textContent = infoNote.hidden ? '' : st.note;
    heading.set(p, auto && j?.phase === 'running' ? 'Following the prop being retopologized'
      : mesh ? [mesh.size ? retopoSizeName(mesh.size) : '', statsLine(p, mesh)].filter(Boolean).join(' · ') : '',
    thumbUrl(p, 'retopo') || thumbUrl(p, 'mesh') || (p.images?.chosen ? thumbUrl(p, 'images') || boxFile(p, p.images.chosen, 'images') : null));
  }

  return {
    key: id,
    id,
    el,
    update(p, auto) {
      renderMedia(p);
      sync(p, auto);
    },
    destroy: () => {
      destroyed = true;
      pane.destroy();
    },
  };
}

// ---- one prop, shaped by hand in Instant Meshes' viewer --------------------------------------------

function buildViewer(prop) {
  const id = prop.id;
  const cur = () => getProp(id) || prop;
  let destroyed = false;
  // idle | loading | ready {id, url, origin} | lost (the tool stopped) | error {message}
  let session = { state: 'idle' };
  let seenRunning = false;   // the status showed Instant Meshes running since the session opened
  let accepting = false;     // until the hub has kept the viewer's result

  const toolReady = () => ['ok', 'unknown'].includes(toolGate('retopo', 'imesh').kind);
  const back = h('button', {
    type: 'button', class: 'btn ghost rt-back', title: 'Back to this prop’s clean meshes, without keeping what the '
      + 'viewer shows (its guide strokes stay in the viewer for later)',
    onclick: () => close(),
  }, icon('left'), 'Back to the clean meshes');
  const bar = methodBar(id, back);
  const viewerMsg = h('div', { class: 'sv-viewer-msg' });
  const busy = busyLayer('retopo');
  const error = errorLine('retopo');
  const stage = h('div', { class: 'sv-imesh' }, viewerMsg, busy, error, donateCorner());
  const el = h('div', { class: 'sv sv-retopo' }, bar, stage);

  /** Back to the prop's clean meshes (after an Accept: the one shaped here, picked). */
  function close() {
    if (viewerFor === id) viewerFor = null;
    render();
  }

  // One iframe per viewer page: a new session gets a fresh one (no history entries, and a message can
  // only come from the page of the session on screen).
  let frame = null;
  function showFrame(url) {
    if (frame?.dataset.src !== url) {
      const next = h('iframe', { class: 'sv-imesh-frame', src: url, allow: 'clipboard-write', title: 'Instant Meshes viewer' });
      next.dataset.src = url;
      if (frame) frame.replaceWith(next);
      else stage.prepend(next);
      frame = next;
    }
    frame.hidden = false;
  }
  function dropFrame() {
    frame?.remove();
    frame = null;
  }

  // ---- Accept: the viewer asks, the hub keeps its result --------------------------------------
  function onMessage(e) {
    if (!frame || e.source !== frame.contentWindow || session.state !== 'ready' || e.origin !== session.origin) return;
    const msg = e.data;
    if (msg?.source === 'instant-meshes' && msg.type === 'export-request') accept(msg, session, e.source);
  }
  window.addEventListener('message', onMessage);

  async function accept(msg, s, viewerWindow) {
    const tell = (state, message) => viewerWindow.postMessage(
      { source: 'spellcast', type: 'export-state', state, ...(message ? { message } : {}) }, s.origin);
    if (accepting) {
      tell('busy');
      return;
    }
    if (msg.session && msg.session !== s.id) {
      tell('error', 'This viewer belongs to another prop. Select the prop again.');
      return;
    }
    accepting = true;
    tell('busy');
    try {
      // The hub keeps the viewer's mesh, picked, and the settings that made it, as the session itself reports them.
      const updated = await api('POST', `/api/imesh/session/${encodeURIComponent(s.id)}/save`,
        { project: projectId(), prop_id: id });
      delete store.jobErrors[`retopo:${id}`];   // an earlier job's error: this result answers it
      bump('retopo', id);
      replaceProp(updated);
      tell('done');
      toast(`Kept the mesh shaped by hand for ${updated.name || cur().name || 'this prop'}`, 'success');
      if (!destroyed) close();
    } catch (e) {
      tell('error', e.message);
      toast(e.message, 'error');
    } finally {
      accepting = false;
    }
  }

  // ---- the viewer session -----------------------------------------------------------------------
  async function openSession() {
    await ctx?.settled;   // the viewer's page loads once the workbench has stopped moving
    if (destroyed || session.state === 'loading') return;
    if (!cur().mesh?.glb || !toolReady()) {
      renderViewer();   // says why (a stale "Try again" would do nothing)
      return;
    }
    session = { state: 'loading' };
    renderViewer();
    try {
      const r = await api('POST', '/api/imesh/session', { project: projectId(), prop_id: id });
      if (destroyed) return;
      const url = viewerUrl(r.viewer_url);
      session = { state: 'ready', id: r.session_id, url, origin: new URL(url).origin };
      // The open may have started Instant Meshes: the status the page has can still say "stopped".
      seenRunning = false;
      pollStatusSoon(0);
      if (r.lost_strokes) {
        toast(`The retopology tool had to close the earlier viewer of ${cur().name || 'this prop'} to make room for `
          + 'other props, so guide strokes that were not accepted are gone.', 'warn', 12000);
      }
    } catch (e) {
      if (destroyed) return;
      session = { state: 'error', message: e.message };
    }
    renderViewer();
    sync(cur());
  }

  let shownMsg = null;
  function renderViewer() {
    const p = cur();
    const ready = session.state === 'ready';
    viewerMsg.hidden = ready;
    if (ready) {
      shownMsg = null;
      showFrame(session.url);
      return;
    }
    if (frame) frame.hidden = true;
    const gate = toolGate('retopo', 'imesh');
    const opening = session.state === 'loading' || session.state === 'idle';
    const why = !p.mesh?.glb ? 'no model' : !toolReady() ? `${gate.text}|${gate.detail}`
      : jobFor('retopo', id) && session.state === 'idle' ? 'busy' : opening ? 'opening' : `${session.state}|${session.message}`;
    if (why === shownMsg) return;   // re-drawn only when it says something else (status polls come often)
    shownMsg = why;
    let msg;
    if (why === 'no model') {
      msg = emptyState('cube', 'No 3D model yet', 'This prop gets its 3D model in the 3D stage first.',
        h('button', { class: 'btn ghost', onclick: () => ctx?.goto('mesh') }, icon('cube'), 'Go to 3D'));
    } else if (!toolReady()) {
      msg = emptyState(gate.kind === 'installing' ? 'clock' : 'alert', gate.text, gate.detail || '');
    } else if (why === 'busy') {
      msg = '';   // the busy layer over it says what happens; the viewer opens when it is done
    } else if (opening) {
      msg = h('div', { class: 'sv-empty' }, spinner(), h('h3', null, 'Opening the interactive viewer…'),
        h('p', null, 'The retopology tool starts by itself and loads this model. This takes a few seconds.'));
    } else if (session.state === 'lost') {
      msg = emptyState('power', 'The retopology tool is not running', 'Open the viewer again to keep working.',
        h('button', { class: 'btn primary', onclick: openSession }, icon('refresh'), 'Reopen the viewer'));
    } else {
      msg = emptyState('alert', "Couldn't open the viewer", session.message,
        h('button', { class: 'btn ghost', onclick: openSession }, icon('refresh'), 'Try again'));
    }
    viewerMsg.replaceChildren(msg);
  }

  function sync(p) {
    bar.sync(p);
    busy.sync(p);
    error.sync(p);
    // Lost only once the page has seen the tool running since the session opened: an older status is stale.
    const state = toolStatus('imesh')?.state;
    if (['ready', 'busy'].includes(state)) seenRunning = true;
    if (session.state === 'ready' && seenRunning && state && !['ready', 'busy', 'starting'].includes(state)) {
      session = { state: 'lost' };
      // The page inside is dead: unload it (it would keep trying to reconnect to a session that no
      // longer exists), so reopening loads it afresh.
      dropFrame();
    }
    if (session.state !== 'ready') renderViewer();
  }

  // A new 3D model, or the end of this prop's job: the session opens (again) for the model the prop has now.
  let glbKey = prop.mesh?.glb ? boxFile(prop, prop.mesh.glb, 'mesh') : '';
  let wasBusy = !!jobFor('retopo', id);
  const update = (p) => {
    const newGlbKey = p.mesh?.glb ? boxFile(p, p.mesh.glb, 'mesh') : '';
    const busyNow = !!jobFor('retopo', id);
    if (newGlbKey !== glbKey || (wasBusy && !busyNow && session.state === 'idle')) {
      if (newGlbKey !== glbKey) session = { state: 'idle' };   // that session holds the old model
      if (!busyNow) openSession();
    }
    glbKey = newGlbKey;
    wasBusy = busyNow;
    sync(p);
  };

  renderViewer();
  if (!wasBusy) openSession();
  return {
    key: `${id}|viewer`,
    id,
    el,
    update,
    destroy: () => {
      destroyed = true;
      window.removeEventListener('message', onMessage);
      dropFrame();   // the viewer page stops at once (its session stays in the tool for this prop)
    },
  };
}

function buildEmpty() {
  const el = h('div', { class: 'sv sv-retopo sv-single' },
    h('div', { class: 'sv-stagebox' }, emptyState('grid', 'No props here yet',
      'Props arrive here once they are 3D models: each is rebuilt as a clean, light mesh with UVs, ready for games.',
      h('button', { class: 'btn ghost', onclick: () => ctx?.goto('images') }, icon('image'), 'Go to Images')),
    donateCorner()));
  return { key: 'empty', id: null, el, update() {}, destroy() {} };
}

// ---- per prop, for the props column ---------------------------------------------------------------

function status(prop) {
  const job = jobState('retopo', prop);
  if (job) return job;
  const method = retopoMethod(prop);
  if (prop.retopo?.obj && !retopoFresh(prop)) {
    if (madeWith(prop) !== method && !prop.retopo.stale && (!prop.retopo.source || prop.retopo.source === prop.mesh?.glb)) {
      return { state: 'stale', message: `Made with ${toolName(madeWith(prop))}, not ${toolName(method)}: make it again` };
    }
    return { state: 'stale', message: 'Made from an older 3D model: make it again' };
  }
  if (retopoReady(prop)) {
    const missing = retopoMissing(prop);
    return { state: 'done', message: `Clean mesh of ${retopoSizeName(retopoUsed(prop))} picked`
      + `${missing.length ? ` · ${sizeList(missing)} not made yet` : ''}` };
  }
  if (retopoFresh(prop)) return { state: 'review', message: 'Ready to pick: click a size under the model' };
  if (!prop.mesh?.glb) {
    return jobFor('mesh', prop.id) || jobFor('images', prop.id) ? { state: 'waiting', message: 'Waiting for its 3D model' }
      : { state: 'unreachable', message: 'No 3D model yet' };
  }
  const gate = toolGate('retopo', method);
  if (!gateOpen(gate)) return { state: 'unreachable', message: gate.text };
  if (!retopoSizes(prop).length) return { state: 'idle', message: 'Type a size in the boxes on its picture' };
  return { state: 'idle', message: 'Not retopologized yet' };
}

// ---- the stage view -------------------------------------------------------------------------------

function render() {
  if (!host) return;
  const { box: prop, auto } = focusBox(ctx, 'retopo', { needsWork: (p) => !!p.mesh?.glb && !retopoReady(p), previousId: shownId });
  shownId = prop?.id ?? null;
  // The viewer belongs to the prop it was opened for, while that prop is on screen and set to Instant Meshes.
  if (!prop || viewerFor !== prop.id || retopoMethod(prop) !== 'imesh') viewerFor = null;
  const key = !prop ? 'empty' : viewerFor ? `${prop.id}|viewer` : prop.id;
  if (view?.key !== key) {
    const sameProp = !!prop && view?.id === prop.id;   // the viewer opened or closed: no fade, the picker stays put
    view?.destroy();
    view = !prop ? buildEmpty() : viewerFor ? buildViewer(prop) : buildProp(prop);
    host.replaceChildren(view.el);
    if (!sameProp) fadeIn(view.el);
  }
  if (prop) view.update(prop, auto);
}

/**
 * The prop's LATO.2 / TriFlow settings differ from those its clean mesh was made with (its stats say them; an
 * older result without them: false).
 */
export function settingsChanged(prop) {
  const st = prop.retopo?.stats || {};
  const set = prop.retopo?.settings || {};
  const method = prop.retopo?.method;
  if (!prop.retopo?.obj || method !== retopoMethod(prop)) return false;
  const differs = (made, now) => typeof made === 'number' && Math.abs(made - now) > 1e-6;
  if (method === 'lato2') return differs(st.seed, set.lato2_seed ?? 0);
  if (method === 'triflow') {
    return differs(st.quad_ratio, set.triflow_quad_ratio ?? TRIFLOW_QUADS) || differs(st.seed, set.triflow_seed ?? 0);
  }
  return false;
}

export default {
  id: 'retopo',
  title: 'Retopo',
  ctaLabel: 'Project textures',
  ctaHover: () => waitingHint(boxesOf()) || CTA_HOVER,
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
    viewerFor = null;
  },

  status,

  /**
   * The boxes on each prop's tile: the sizes of its clean meshes, in thousands of vertices with up to two decimals.
   * The unit is short: three boxes as wide as "12.25" and "k vertices" did not fit the tile.
   */
  tileSizes: (prop) => (prop.draft || tileable(prop) ? null
    : { values: retopoSizes(prop), ...RETOPO_SIZES, unit: 'k verts', title: SIZES_TITLE }),

  /** What was typed in them: the prop's sizes, each once, saved like any setting. */
  sizeTile(id, numbers) {
    queueSettings(id, 'retopo', { sizes: [...new Set(numbers)].slice(0, RETOPO_SIZES.count) });
  },

  /** The tool of "Retopo All"'s first job starts as the stage opens, taking the card from a tool idling on it. */
  warmWish: () => {
    const [first] = retopoPlan(boxesOf(), liveDeps());
    return first ? { tool: first[2], displace: true } : null;
  },

  /** The prop the workbench shows. */
  shownId: () => shownId,

  /** On to Projection once every prop has a picked, up-to-date clean mesh and none is being made again. */
  ctaEnabled() {
    const hint = waitingHint(boxesOf().filter(inStage));
    return hint ? { enabled: false, blocked: true, hint } : { enabled: true };
  },

  /** Bakes the textures of every prop that lacks a fresh bake. */
  async onCta() {
    await carryDown('projection');
  },
};
