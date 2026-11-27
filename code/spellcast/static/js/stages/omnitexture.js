// Stage 3, Omni-Texture (optional): for the props the artist chooses, the 3D model's colour texture is painted anew -
// Qwen-Image 2.1 paints the model seen from a few sides (the hub's omnitexture.VIEW_COUNT), together in one picture
// where they fit (a sheet: omnitexture.sheets), and each view is projected back onto it - into a copy of the model
// (omnitexture.glb). The box beside a prop's tile chooses it or leaves it out again (tileCheck / checkTile, saved as
// omnitexture.settings.chosen); a click on the tile only opens the prop. "Omni-Texture the chosen", in the viewport's
// lower-left corner, starts one job for the chosen props that lack a fresh one. The viewport shows the result, or
// the 3D model (the switch at its top right); on the right, the pictures that went onto the model: with the pointer
// over one the model shows that view's share alone, drawn as the view chosen under the viewport draws the model
// (viewer3d.js setLayer); Alt + click on one shows what Qwen-Image was given for it and what it answered (showInputs);
// its × takes it off the model again (takeOff: Ctrl+Z brings it back), and a click anywhere on it hides it there or
// shows it again - the eye in its middle says which (setHidden: the painted views all together, a repaint alone).
// The hub is asked to ready its worker for that click as a result with repaints comes on screen, and for a click
// on any of them as the pointer comes to the pictures (readyWorker), so that the first click is as quick as the
// next - where the hub sees fit: not beside a job. Retopo reads the 3D model's shape
// either way; Projection bakes the result's colours, as shown, while it is used (rules.omnitextureUsed).
// The artist may paint with the marker and the clone stamp of the viewport's bottom-right corner (saved into the step
// shown as hand paint), and mask what came out wrong with the mask marker and have it painted again as seen from
// there ("Render Masked": stages/maskbar.js) - on an Omni-Texture, and as well on any prop's 3D model, chosen or not:
// the first tool taken up makes a copy of that model the first step of a result of its own texture (begin: the hub's
// .../omnitexture/own; rules.omniOwn), and a job that paints the prop later keeps all of it on top of its views.
// Before the page has the hub make the steps again - a job, a fade, a picture taken off - the hand paint is in the
// files: the hub takes none while that is under way.
// The call to action goes on to Retopo and paints nothing: Omni-Texture is only ever started by its own button.

import { api, propApi } from '../api.js';
import { h, plural } from '../dom.js';
import { icon } from '../icons.js';
import { ACTIVE_JOB, boxesOf, getProp, jobFor, projectId, store } from '../store.js';
import { queueSettings, replaceProp, startJob } from '../actions.js';
import { field, segmented, spinner } from '../controls.js';
import { attempt, confirmDialog, modal, toast, toastError } from '../ui.js';
import {
  INPUTS_KEY, afterMove, boxFile, boxHeading, busyLayer, canCarry, carryDown, donateCorner, emptyState, errorLine, fadeIn,
  figure, focusBox, gateLine, gateOpen, jobState, liveDeps, showConfigFile, statsBlock, thumbUrl, toolGate,
} from './common.js';
import {
  TAKEN_OFF, omniBatch, omniChosen, omniOwn, omniRepaints, omniStatus, omniTakeOff, omniTakenOff, omniTextured,
  omniViews, omniViewsEye, omniViewsHidden, omnitextureFresh, tileable,
} from './rules.js';
import { ring } from '../shell/anim.js';
import { DEFAULT_VIEW, modelPane, rememberWireframe, wireframeOn } from './preview.js';
import { livePreview } from './images.js';
import { createFadeSlider } from './fadeslider.js';
import { fasterSwitch } from './faster.js';
import { createMaskBar, renameModels, tookOff } from './maskbar.js';

const INTRO = 'Tick the box beside each prop on the left that needs Omni-Texture. Click a prop to look at it: you can '
  + 'also paint on it, or mask a part and have only that painted again, without choosing it.';
const TICK = 'Tick the box beside its tile on the left to choose it for Omni-Texture.';
const WHOLE = 'Take the whole Omni-Texture off this model';
// The eye in the middle of a picture of the list, by what a click on the picture does.
const VIEWS_EYE = {
  hide: 'Hide the painted views - all of them, they go together: the model shows its own texture in their place',
  show: 'Show the painted views on the model again',
};
const REPAINT_EYE = { hide: 'Hide this repaint: the model shows what is under it', show: 'Show this repaint on the model again' };
// The side panel's button: the whole Omni-Texture anew, or - while something is masked - the mask alone.
const AGAIN = { words: 'Paint it again', title: 'Paints this prop’s Omni-Texture anew (the one it has now is replaced)' };
const MASKED = {
  words: 'Render Masked',
  title: 'Qwen-Image repaints only what you masked, as you see it from here. (Put the mask marker down and discard '
    + 'the mask to paint the whole Omni-Texture again.)',
};
const WHAT = 'Qwen-Image 2.1 paints the 3D model from a few sides and the pictures are projected back onto it: a new, sharper '
  + 'colour texture. The shape stays the same, and Project bakes these colours.';
const LIVE_LINGER_MS = 600;   // the last rough picture stays this long after its prop: the model arrives meanwhile

let host = null;
let ctx = null;
let view = null;          // what is on screen: {key, el, update(prop), destroy()}
let shownId = null;       // the prop shown last, kept while nobody picks another
// The material the artist picked last (None, Base colour...), kept across the Original | Omni-Texture switch and
// the props, as Projection keeps its own.
let lastView = DEFAULT_VIEW;

/** Omni-Texture can run here (Qwen-Image 2.1 is the image generator, and it can run or is installing). */
const canRun = () => gateOpen(toolGate('omnitexture'));

/** An Omni-Texture job waits or runs. */
const working = () => (store.jobs || []).some((j) => j.stage === 'omnitexture' && ACTIVE_JOB.has(j.status));

/** 45 -> "45 s", 635 -> "10 min 35 s". */
const duration = (seconds) => {
  const s = Math.round(seconds);
  return s < 120 ? `${s} s` : `${Math.floor(s / 60)} min${s % 60 ? ` ${s % 60} s` : ''}`;
};

// ---- one view's inputs ------------------------------------------------------------------------------------------

// What a view's picture 1 showed (request.json "shown"; records older than it showed the depth map).
const SHOWN_AS = {
  clay: 'the model in plain grey, lit from the side',
  depth: 'the depth map',
  normal: 'the model’s normal map',
  texture: 'the model with its own texture',
};

// What a repaint's pictures are, by their number (adapters/omnitexture.py RETOUCH_PICTURES and the example). The
// example is the prop's own picture (request.json's retouch.example names its side; retouch.delit: its lighting was
// taken off first; retouch.mirrored: given flipped left to right, the camera looking from behind that side); before
// that it was the model from the nearest camera around it (retouch.example_view), and at first your view without
// the mask. retouch.smooth: the depth map went without the model's facets ("No Facets"). `held`: the painting was
// drawn anew where the view is masked and held to the view elsewhere (request.json's redrawn); before that it
// started from it.
const repaintGiven = ({ example, mirrored, delit, smooth, example_view: from } = {}, held = false) => ({
  1: `Picture 1, also what it ${held ? 'is held to outside the mask' : 'starts from'}: your view of the model, what you masked in magenta`,
  2: `Picture 2: the depth map from there${smooth ? ', its facets blurred away (No Facets)' : ''}`,
  3: `Picture 3, the example: ${example ? `the prop’s own picture of its ${example}${delit ? ', its lighting taken off' : ''}${
    mirrored ? ', mirrored: you look from behind it' : ''}`
    : from ? `the model from the ${from}, the nearest camera around it` : 'the model without the mask'}`,
});

/**
 * What Qwen-Image was given for one picture of the list and what came back (Alt + click on it), from the files
 * kept in its folder (request.json names the words and the pictures in the order it got them).
 * shown: a painted view {folder, side, picture, albedo, sheet: the sheet's folder or null, together: its views'
 * sides} - one painted together with others in one picture has that sheet's files, the whole sheet's pictures -,
 * or a repaint {folder: the one its pictures are kept in, repaint: its number in the list}.
 */
async function showInputs(prop, shown) {
  const file = (rel) => boxFile(prop, rel, 'omnitexture');
  const base = shown.sheet || shown.folder;
  const whole = (name) => file(`${base}/${name}`);
  let record;
  let lightOff = !!shown.albedo;
  try {
    const res = await fetch(whole('request.json'));
    if (!res.ok) throw new Error(`The record of this picture could not be read (error ${res.status}).`);
    record = await res.json();
    // A repaint made when a repaint's lighting was still taken off kept that picture: its file says so.
    if (shown.repaint) lightOff = (await fetch(whole('albedo.png'))).ok;
  } catch (e) {
    toastError(e);
    return;
  }
  const lines = (Array.isArray(record.pictures) ? record.pictures : [])
    .map((line) => /^picture (\d+): (prop: )?(.+)$/.exec(line)).filter(Boolean);
  // Captions by file: picture 2 was the mask at first, then the model with its own texture, and is now the prop's
  // own picture (the example; request.json's "example" names its side).
  const start = record.start_image || '1-render.png';
  const viewCaption = (n, own, name) => `Picture ${n}${!own && name === start ? ', also what it starts from' : ''}`
    + (n === 1 ? `: ${SHOWN_AS[record.shown] || SHOWN_AS.depth}`
      + (shown.sheet ? ', every view in its place' : '') + (record.painted_where ? '' : ', earlier views over it')
      : !own && name === 'model.png' ? ': the model with the texture the 3D stage gave it'
        : !own && name === record.mask_image ? ', the mask: white is painted'
          : own && n === 2 && record.example ? `, the example: the prop’s own picture of its ${record.example}`
            : `: the prop’s own picture${!own && record.pictures_size
              ? `, blurred (shrunk to ${record.pictures_size} px and enlarged back)` : ''}`);
  // How large Qwen-Image's own copy of a picture is, when the request said (request.json's ref_sides).
  const seenAt = (n) => (record.ref_sides?.[n - 1] ? ` (Qwen-Image sees it at ${record.ref_sides[n - 1]} px)` : '');
  const given = lines.map(([, number, own, name]) => figure(file(own ? name : `${base}/${name}`),
    shown.repaint ? repaintGiven(record.retouch, !!record.redrawn)[Number(number)] || `Picture ${number}`
      : viewCaption(Number(number), own, name) + seenAt(Number(number))));
  // A sheet starts from the model with its own texture, which Qwen-Image is no longer shown as a picture.
  if (!shown.repaint && !lines.some(([, , own, name]) => !own && name === start)) {
    given.push(figure(whole(start), 'Not a picture it sees: what the painting starts from, one step in - the model '
      + 'with the texture the 3D stage gave it'));
  }
  // Since Qwen-Image is given no mask: where its painting went onto the model, faded out towards the edge.
  if (record.painted_where) {
    given.push(figure(whole(record.painted_where),
      'Not given to Qwen-Image: where its painting went onto the model (white), fading out at the edge'));
  } else if (record.mask_image && !lines.some(([, , own, name]) => !own && name === record.mask_image)) {
    given.push(figure(whole(record.mask_image), 'Not a picture it sees: the area it paints (white), the engine’s mask'));
  }
  if (record.redrawn) {
    given.push(figure(whole(record.redrawn), 'Not a picture it sees: what it draws anew (white); everywhere else its '
      + 'painting is held to picture 1'));
  }
  // A sheet's own pictures (all its views); a view painted alone and a repaint have them in their folder.
  const landed = shown.sheet || shown.repaint ? whole('projected.png') : shown.picture;
  const back = lightOff ? [figure(whole('answer.png'), 'Qwen-Image’s painting'),
    figure(whole('albedo.png'), 'Its lighting taken off'), figure(landed, 'What went onto the model')]
    : [figure(shown.sheet || shown.repaint ? whole('answer.png') : shown.picture,
      'Qwen-Image’s painting, as it went onto the model')];
  const others = (shown.together || []).filter((side) => side !== shown.side);
  const tries = record.retouch?.tries?.length || 1;
  modal({
    title: shown.repaint ? `Repaint ${shown.repaint}: what Qwen-Image was given`
      : shown.sheet ? `The ${shown.side} view, painted in one picture with ${others.length === 1 ? `the ${others[0]}`
        : `${others.length} others`}: what Qwen-Image was given`
        : `The ${shown.side} view: what Qwen-Image was given`,
    wide: true,
    className: 'ot-inputs',
    body: h('div', { class: 'stack' },
      field('Words', h('p', { class: 'ot-words' }, record.prompt || '')),
      h('div', { class: 'hint' }, `Qwen-Image 2.1${record.model === 'full' ? '' : ' Turbo'} · seed ${record.seed} · `
        + `${record.steps} steps · strength ${record.strength}`
        + (tries > 1 ? ` · the ${tries === 2 ? 'second' : 'third'} picture (the ones before kept the mask’s shape)` : '')
        + (record.retouch?.wide ? ` · seen ${record.retouch.tan < 0 ? 'in inverse perspective' : 'with less perspective'}` : '')
        + (shown.sheet ? ` · ${record.width} × ${record.height} px, the views: ${(shown.together || []).join(', ')}` : '')),
      field('Pictures given, in this order', h('div', { class: 'ot-inputs-grid' }, given)),
      field('What came back', h('div', { class: 'ot-inputs-grid' }, back)),
      h('div', { class: 'hint' }, `The files: projects/${projectId()}/props/${prop.id}/${base}`)),
    actions: [{ label: 'Close', value: true }],
  });
}

// ---- "Omni-Texture the chosen", in the viewport's lower-left corner -----------------------------------------------

/**
 * The button that starts the job for the chosen props (rules.omniBatch); "… again" once every chosen prop has a fresh
 * Omni-Texture (asked first: they are painted anew). Hidden while Omni-Texture can't run here (the viewport says why).
 * before(): waited for just before the job is asked for (the hand paint of the prop shown goes into its files).
 */
function goButton({ before = null } = {}) {
  const words = h('span');
  const count = h('span', { class: 'go-count' });
  const glyph = h('span', { class: 'ot-glyph' });
  let starting = false;
  const btn = h('button', { type: 'button', class: 'btn primary ot-go', onclick: () => go() }, glyph, words, count);

  async function go() {
    if (starting) return;
    const deps = liveDeps();
    let ids = omniBatch(boxesOf(), deps).map((b) => b.id);
    if (!ids.length) {
      ids = omniBatch(boxesOf(), deps, { again: true }).map((b) => b.id);
      if (!ids.length || !(await confirmDialog({
        title: `Omni-Texture ${plural(ids.length, 'prop')} again?`, ok: 'Paint them again',
        message: 'Every chosen prop has its Omni-Texture already. Painting them again replaces it with a new one; '
          + 'what you painted or had painted again on them stays on top.',
      }))) return;
    }
    starting = true;
    sync();
    await before?.();
    await attempt(() => startJob('omnitexture', ids));
    starting = false;
    sync();
  }

  function sync() {
    const run = canRun();
    btn.hidden = !run;
    if (!run) return;
    const deps = liveDeps();
    const todo = omniBatch(boxesOf(), deps);
    const again = todo.length ? [] : omniBatch(boxesOf(), deps, { again: true });
    const busy = starting || (!todo.length && working());
    const n = todo.length || again.length;
    btn.disabled = busy || !n;
    btn.classList.toggle('working', busy);
    words.textContent = busy ? 'OmniTexturing…' : todo.length || !again.length ? 'Omni-Texture the chosen'
      : 'Omni-Texture the chosen again';
    count.textContent = n && !busy ? String(n) : '';
    count.hidden = !count.textContent;
    const shown = busy ? 'working' : 'sparkles';
    if (glyph.dataset.icon !== shown) {
      glyph.dataset.icon = shown;
      glyph.replaceChildren(busy ? spinner() : icon('sparkles'));
    }
    btn.title = busy ? 'The chosen props are being painted. Their tiles say how far each one is.'
      : n ? `Qwen-Image 2.1 paints a new colour texture for ${plural(n, 'chosen prop')}, from a few sides each.`
        : boxesOf().some(omniChosen) ? 'The chosen props need a 3D model first: make them in 3D.'
          : 'Choose props first: tick the boxes beside them on the left.';
  }
  btn.sync = sync;
  btn.starting = () => starting;
  return btn;
}

/** What the viewport says when Omni-Texture can't run here (no Qwen-Image 2.1), with the config file one click away. */
function cantRun() {
  const gate = toolGate('omnitexture');
  const link = gate.kind === 'config'
    ? h('button', { class: 'btn ghost', onclick: () => showConfigFile() }, icon('folder'), 'Show the file') : null;
  return emptyState('alert', 'Omni-Texture can’t run here', `${gate.text} ${gate.detail || ''}`.trim(), link);
}

// ---- no prop shown: the stage says what to do -------------------------------------------------------------------

function buildIntro() {
  const go = goButton();
  const note = h('div', { class: 'ot-intro' });
  const stage = h('div', { class: 'sv-stagebox ot-stage' }, note, go, donateCorner());
  const el = h('div', { class: 'sv sv-omnitexture sv-single' }, stage);
  let shown = null;
  return {
    key: 'intro', el, go,
    update() {
      const run = canRun();
      const key = run ? 'choose' : `cant|${toolGate('omnitexture').text}`;
      if (key !== shown) {
        shown = key;
        note.replaceChildren(run ? emptyState('sparkles', 'Choose the props to Omni-Texture', INTRO,
          h('p', { class: 'hint ot-what' }, WHAT)) : cantRun());
      }
      go.sync();
    },
    destroy() {},
  };
}

function buildEmpty() {
  const el = h('div', { class: 'sv sv-omnitexture sv-single' },
    h('div', { class: 'sv-stagebox' }, emptyState('sparkles', 'No props here yet',
      'Make some props in Images and 3D first. Here you may then give the ones you choose a new, sharper colour texture.',
      h('button', { class: 'btn ghost', onclick: () => ctx?.goto('images') }, icon('image'), 'Go to Images')),
    donateCorner()));
  return { key: 'empty', el, update() {}, destroy() {} };
}

// ---- one prop -------------------------------------------------------------------------------------------------

function buildProp(prop) {
  const id = prop.id;
  const cur = () => getProp(id) || prop;
  // The result shows first; the 3D model a click away (kept while this prop is on screen).
  let side = 'result';
  const sides = segmented([
    { value: 'model', label: 'Original', title: 'The 3D model with its own texture, as the 3D stage made it' },
    { value: 'result', label: 'Omni-Texture', title: '' },      // (named by what the result is: renderMedia)
  ], side, (v) => {
    side = v;
    renderMedia(cur());
  });
  const pane = modelPane({ tools: [sides], onView: (mode) => { lastView = mode; } });
  pane.stage.classList.add('ot-stage');
  pane.maps.offerWireframe(wireframeOn('omnitexture', false), (on) => rememberWireframe('omnitexture', on));
  const busy = busyLayer('omnitexture');
  // While Qwen-Image 2.1 paints this prop: its picture as it forms, under the progress (stages-models.css).
  const live = h('div', { class: 'sv-live', hidden: true }, h('span', { class: 'sv-live-tag' }, 'Rough preview'));
  const go = goButton({ before: () => mask.settle() });
  // The prop is being worked on (its job starting, a fade, a picture being taken off): nothing paints meanwhile.
  let maskChanged = () => {};       // (set once the side panel's button is there)
  const mask = createMaskBar({
    busy: () => !!jobFor('omnitexture', id) || starting || go.starting() || !!changing || fade.fading(),
    changed: () => maskChanged(),
  });
  pane.stage.append(mask.cover, live, busy, go, mask.el);
  const media = h('div', { class: 'sv-media' }, pane.el);

  let liveUrl = '';
  let liveImg = null;
  let liveGone = null;
  const hideLive = () => {
    live.hidden = true;
    pane.stage.classList.remove('live');
    liveImg?.remove();
    liveImg = null;
  };
  /**
   * The newest rough picture of the view being painted (images.js livePreview), loaded before it is swapped in;
   * never the next prop's, which is on the card while this one finishes behind it.
   */
  function renderLive() {
    const url = livePreview(jobFor('omnitexture', id), liveUrl, id);
    if (url === liveUrl) return;
    liveUrl = url;
    clearTimeout(liveGone);
    if (!url) {
      if (liveImg) liveGone = setTimeout(hideLive, LIVE_LINGER_MS);
      else hideLive();
      return;
    }
    const next = h('img', { class: 'sv-live-img', alt: 'A rough preview of the view being painted', draggable: 'false' });
    next.onload = () => {
      if (liveUrl !== url) return;
      if (liveImg) liveImg.replaceWith(next);
      else live.prepend(next);
      liveImg = next;
      live.hidden = false;
      pane.stage.classList.add('live');
    };
    next.src = url;
  }

  /**
   * The prop's result to show and to paint: its Omni-Texture, or its own texture as the artist painted it - but not
   * one of its own on a 3D model made again since (it is begun anew on the model the prop has now).
   */
  const resultOf = (p) => (p.omnitexture?.glb && (omnitextureFresh(p) || !omniOwn(p)) ? p.omnitexture.glb : null);

  /** What the viewport shows now: {url, snapshot, step | model: the file, as a result's step or the 3D model} or {empty: <node>}. */
  function target(p) {
    const result = resultOf(p);
    if (result && side === 'result') {
      return { url: boxFile(p, result, 'omnitexture'), step: result,
        snapshot: thumbUrl(p, 'omnitexture') || thumbUrl(p, 'mesh') };
    }
    if (!p.mesh?.glb) {
      return { empty: emptyState('cube', 'No 3D model yet', `This prop becomes 3D in the 3D stage first.${
        canRun() ? ' Chosen, it is painted once it has its model.' : ''}`,
        h('button', { class: 'btn ghost', onclick: () => ctx?.goto('mesh') }, icon('cube'), 'Go to 3D')) };
    }
    return { url: boxFile(p, p.mesh.glb, 'mesh'), model: p.mesh.glb, snapshot: thumbUrl(p, 'mesh') };
  }

  let shown = null;
  let onScreen = null;    // the file the viewport holds: {step} (a result's) or {model} (the 3D model itself)
  let framed = false;     // after the first model, the switch keeps the camera: the artist compares up close
  function renderMedia(p) {
    const result = resultOf(p);
    const own = !!result && omniOwn(p);
    sides.hidden = !result;
    sides.set(side);
    const [, mine] = sides.buttons;
    mine.textContent = own ? 'Painted' : 'Omni-Texture';
    mine.title = own ? 'The same model with what you painted and had painted again on it'
      : 'The same model with its new, painted texture';
    const chosen = omniChosen(p);
    const painted = !!result && !own;       // a job's Omni-Texture
    pane.badges(
      !canRun() ? null
        : !chosen ? h('span', { class: 'chip', title: TICK },
          icon('info'), painted ? 'Not chosen: Project uses its 3D model’s own texture' : 'Not chosen: keeps its own texture')
          : !painted && p.mesh?.glb && !jobFor('omnitexture', p.id)
            ? h('span', { class: 'chip ot-chosen' }, icon('sparkles'), 'Chosen: waits for “Omni-Texture the chosen”') : null,
      painted && !omnitextureFresh(p) ? h('span', {
        class: 'chip warn', title: 'The 3D model was made again after this Omni-Texture. Paint it again to use the new one.',
      }, icon('alert'), '3D model changed: Omni-Texture again') : null);
    const t = target(p);
    const key = t.url ? t.url : `empty|${!p.mesh?.glb}`;
    if (key === shown) return;
    shown = key;
    onScreen = null;
    if (!t.url) {
      pane.message(t.empty);
      return;
    }
    mask.unbind();
    pane.show(t.url, 'glb', { keepView: framed, view: lastView, ...afterMove(ctx, t.snapshot) })
      .then((ok) => {
        if (!ok) return;
        framed = true;
        if (shown === key) onScreen = t.step ? { step: t.step } : { model: t.model };
        syncMask(cur());
        readyWorker(false);
      });
  }

  /**
   * A tool was taken up on the prop's 3D model (maskbar.js bind's `begin`): the hub makes a copy of that model the
   * first step of a result of its own texture, which the viewport then shows and the tools paint and mask like any
   * result. The copy is loaded as its own model: what the viewer holds of the 3D model itself is never painted (the
   * 3D stage and "Original" here show that one).
   */
  async function begin() {
    const project = projectId();
    const done = await api('POST', `${propApi(project, id)}/omnitexture/own`);
    if (projectId() !== project) return;
    side = 'result';
    replaceProp(done.prop);
  }

  /**
   * The tools (the marker, the clone stamp, the mask marker) are there while Qwen-Image can paint and no Omni-Texture
   * job of this prop waits or runs: on the result's step shown, which they paint - or on the 3D model of a prop
   * without a result, where the first one taken up begins one (begin).
   */
  function syncMask(p) {
    const viewer = pane.viewer;
    const result = resultOf(p);
    const step = onScreen?.step && onScreen.step === result && side === 'result' ? result : null;
    const model = !result && onScreen?.model && onScreen.model === p.mesh?.glb ? onScreen.model : null;
    if (!viewer || !(step || model) || !canRun() || jobFor('omnitexture', id)) {
      mask.unbind();
      return;
    }
    // The 3D model shown for a result taken off stands for that step: Ctrl+Z walks back to the result from it.
    const off = !!model && omniTakenOff(p);
    mask.bind({
      viewer, project: projectId(), propId: id, glb: step || model, step: off ? TAKEN_OFF : null,
      steps: step || off ? p.omnitexture.steps || [] : [],
      select: (mode) => pane.maps.select(mode, true), begin: step ? null : begin,
    });
  }

  // ---- right: the prop, its result's numbers and the pictures painted -------------------------------------------
  const heading = boxHeading();
  const stats = h('div');
  const pictures = h('div', { class: 'ot-views', onpointerenter: () => readyWorker(true) });
  // The repaints of what the artist masked stand among the views; the one clicked last - until one is clicked, the
  // newest that shows - owns the slider under them while it shows (stages/fadeslider.js), and the prop then shows
  // the model made again.
  let repaints = [];        // those the model shown holds (renderResult)
  let picked = -1;          // the one clicked last (none: the newest that shows)
  const pickedRepaint = () => repaints[picked] ?? repaints.findLast((r) => !r.hidden) ?? null;
  const fade = createFadeSlider(id, {
    before: () => mask.settle(),
    faded: (project, done) => {
      renameModels(project, id, done.renamed);
      replaceProp(done.prop);
    },
    changed: () => syncFade(),
  });
  // A picture's × takes it off the model (POST .../omnitexture/remove: the hub's worker alone, no Qwen-Image): the
  // model without it is the prop's next step, so Ctrl+Z (stages/maskbar.js) shows the picture again. The × beside
  // the heading takes the whole Omni-Texture off the same way: the step after it is none of its models
  // (rules.js TAKEN_OFF), and the prop shows its 3D model.
  // A click on a picture hides it on the model, or shows it again (POST .../omnitexture/hide, the worker alone
  // too): the picture stays in the list, dim under a crossed eye. As a fade, that is no step of its own - the next
  // click takes it back.
  const whole = h('button', {
    type: 'button', class: 'ot-x', hidden: true, title: `${WHOLE} (Ctrl+Z brings it back)`, 'aria-label': WHOLE,
    onclick: () => takeOff({ whole: true }, pictures),
  }, icon('x'));
  const picturesBox = h('div', { class: 'field', hidden: true },
    h('div', { class: 'field-label' }, h('span', null, 'Painted views'), whole), pictures, fade.row);
  let changing = null;      // the tiles whose pictures are being hidden, shown or taken off
  const waits = () => omniTakeOff(cur(), { job: jobFor('omnitexture', id), changing: changing || fade.fading() })
    .waits;
  function syncFade() {
    const made = pickedRepaint();
    const fades = made && !made.hidden ? made : null;        // (a hidden one has no edge to fade)
    fade.show(fades, { any: repaints.length > 0, locked: !!jobFor('omnitexture', id) || !!changing });
    pictures.querySelectorAll('.ot-repaint').forEach((node, i) => node.classList.toggle('picked', repaints[i] === fades));
    for (const kind of ['.ot-x', '.ot-eye']) {
      for (const button of pictures.querySelectorAll(kind)) button.disabled = waits();
    }
    whole.disabled = waits();
  }
  /**
   * The hub's worker is asked to be ready for a click on a picture of the result on screen: on the newest repaint
   * when the result comes on screen, on `every` one when the pointer comes to them.
   */
  function readyWorker(every) {
    if (onScreen?.step && repaints.length && !waits()) fade.ready(every);
  }
  /**
   * A change of the pictures on the model shown, asked of the hub (POST .../omnitexture/<verb> with `body`) while
   * `tiles` wait; told(project, answer) hears its answer before the prop it brings is shown.
   */
  async function change(verb, body, tiles, told) {
    if (waits()) return;
    const project = projectId();
    changing = tiles;
    for (const waiting of tiles) waiting.classList.add('changing');
    syncFade();
    try {
      await mask.settle();
      const done = await api('POST', `${propApi(project, id)}/omnitexture/${verb}`,
        { glb: cur().omnitexture?.glb, ...body });
      told(project, done);
      if (projectId() === project) replaceProp(done.prop);
    } catch (e) {
      toastError(e);
    } finally {
      for (const waiting of tiles) waiting.classList.remove('changing');
      changing = null;
      syncFade();
    }
  }
  const takeOff = (what, tile) => {
    picked = -1;            // (the repaints' places change)
    return change('remove', what, [tile], (project) => {
      if (what.whole) tookOff(project, id);
    });
  };
  /** what: {repaint: the model it made} or {views: true} - the painted views go together, all of them. */
  const setHidden = (what, hidden, tiles) => change('hide', { ...what, hidden }, tiles,
    (project, done) => renameModels(project, id, done.renamed));
  /**
   * A picture of the list in its tile. While the pointer is on it the model shows its `layer` alone. A click
   * anywhere on it hides it or shows it again - eye: {hidden: it is hidden now, words,
   * toggle(tile)}, drawn as the eye in its middle (none: the click does nothing) -, after `pick` (what else a
   * click means); Alt + click shows what Qwen-Image was given (`inputs`). take: {what: as the hub names it, words} -
   * the × in its corner, which takes it off the model.
   */
  function tile(picture, { layer, inputs, eye, take, pick }) {
    const clicked = (e) => {
      e.stopPropagation();
      if (e.altKey) {
        inputs();
      } else if (eye && !waits()) {
        pick?.();
        eye.toggle(box);
      }
    };
    const box = h('div', {
      class: `ot-tile${eye?.hidden ? ' off' : ''}`,
      onpointerenter: () => showLayer(layer),
      onpointerleave: () => showLayer(null),
      onclick: clicked,
    }, picture);
    if (eye) {
      box.append(h('button', {
        type: 'button', class: 'ot-eye', title: eye.words, 'aria-label': eye.words, onclick: clicked,
      }, icon(eye.hidden ? 'eyeoff' : 'eye')));
    }
    if (take) {
      box.append(h('button', {
        type: 'button', class: 'ot-x', title: `${take.words} (Ctrl+Z brings it back)`, 'aria-label': take.words,
        onclick: (e) => {
          e.stopPropagation();
          takeOff(take.what, box);
        },
      }, icon('x')));
    }
    return box;
  }
  const gate = gateLine('omnitexture');
  const error = errorLine('omnitexture');
  let starting = false;
  // The panel's own button paints the prop's whole Omni-Texture anew - but not while the artist is masking: then
  // it is the mask bar's "Render Masked", so a press meant for the mask never replaces the whole texture.
  const againWords = h('span', null, AGAIN.words);
  const againBtn = h('button', {
    class: 'btn ghost sv-wide', title: AGAIN.title,
    onclick: async () => {
      if (mask.masking()) {
        mask.render();
        return;
      }
      if (starting) return;
      starting = againBtn.disabled = true;
      await mask.settle();
      await attempt(() => startJob('omnitexture', [id]));
      starting = false;
      sync(cur());
    },
  }, icon('sparkles'), againWords);
  // As "Render Masked", the mask's words above it and "No Facets" on its left: the box beside the mask's, again.
  const againRow = h('div', { class: 'ot-again' }, mask.side.facets, againBtn);
  function syncAgain() {
    const masking = mask.masking();
    const says = masking ? MASKED : AGAIN;
    if (againWords.textContent !== says.words) {
      againWords.textContent = says.words;
      againBtn.title = says.title;
      againBtn.classList.toggle('primary', masking);
      againBtn.classList.toggle('ghost', !masking);
    }
    againBtn.disabled = starting || (masking && mask.working());
    mask.side.facets.hidden = !masking;
    mask.side.facets.disabled = againBtn.disabled;
    mask.side.words.hidden = !masking || againBtn.hidden;
  }
  maskChanged = syncAgain;
  const foot = h('div', { class: 'sv-foot' }, error, gate, mask.side.words, againRow);
  // Omni-Texture's own "Faster image generation" (faster.js): the repaints are Qwen-Image's pictures too. Apart from
  // the rest.
  const faster = fasterSwitch('omnitexture');
  const fasterBox = h('div', { class: 'ot-faster' }, faster);
  const sidePanel = h('div', { class: 'sv-side' },
    h('div', { class: 'sv-side-scroll' }, heading, stats, picturesBox, fasterBox), foot);
  const el = h('div', { class: 'sv sv-omnitexture' }, media, sidePanel);

  let shownViews = null;
  let layerOn = false;
  /** The model shows the layer at `url` alone (null: as it was), while the pointer is over a view's picture. */
  const showLayer = (url) => {
    if (!url && !layerOn) return;
    layerOn = !!url;
    pane.viewer?.setLayer(url);
  };
  function renderResult(p) {
    const block = resultOf(p) ? p.omnitexture : {};     // (a result that is not shown has nothing to list)
    const st = block.glb ? block.stats || {} : {};
    const file = (rel) => (rel ? boxFile(p, rel, 'omnitexture') : null);
    const views = omniViews(block).map((v) => ({ ...v, picture: file(v.picture), layer: file(v.layer) }));
    const count = repaints.length;
    repaints = omniRepaints(block).map((r) => ({
      ...r, folder: r.picture.slice(0, r.picture.lastIndexOf('/')), picture: file(r.picture), layer: file(r.layer),
    }));
    if (repaints.length !== count) picked = -1;       // (another repaint, or one fewer: none is clicked)
    const notes = [st.warning, ...(Array.isArray(st.warnings) ? st.warnings : [])];
    const drawn = new Set(views.map((v) => v.sheet || v.folder)).size;      // the pictures Qwen-Image painted
    stats.replaceChildren(statsBlock([
      ['Made in', typeof st.seconds === 'number' ? duration(st.seconds) : null],
      ['Views', views.length ? `${views.length}${drawn < views.length
        ? `, painted in ${plural(drawn, 'picture')}` : ''}` : null],
      ['Seed', st.seed ?? null],
    ], notes));
    const offered = omniTakeOff(p, { canRun: canRun() });
    const viewsOff = omniViewsHidden(block);
    whole.hidden = !offered.whole;
    const key = [offered.views, offered.repaints, viewsOff, omniViewsEye(p),
      ...[...views, ...repaints].map((v) => `${v.picture}>${v.layer}>${!!v.hidden}`)].join('|');
    if (key === shownViews) {
      syncFade();
      return;
    }
    shownViews = key;
    showLayer(null);
    picturesBox.hidden = !views.length && !repaints.length;
    const viewTiles = views.map((v, i) => tile(viewThumb(v, i, views, viewsOff), {
      layer: v.layer,
      inputs: () => showInputs(cur(), v),
      eye: omniViewsEye(p) && {
        hidden: viewsOff, words: viewsOff ? VIEWS_EYE.show : VIEWS_EYE.hide,
        toggle: () => setHidden({ views: true }, !viewsOff, viewTiles),
      },
      take: offered.views && { what: { view: v.folder }, words: 'Take this view off the model' },
    }));
    const repaintTiles = repaints.map((r, i) => tile(repaintThumb(r, i, repaints), {
      layer: r.layer,
      inputs: () => showInputs(cur(), { folder: r.folder, repaint: i + 1 }),
      eye: r.fadable && {
        hidden: r.hidden, words: r.hidden ? REPAINT_EYE.show : REPAINT_EYE.hide,
        toggle: (box) => setHidden({ repaint: r.glb }, !r.hidden, [box]),
      },
      pick: () => { picked = i; },
      take: offered.repaints && { what: { repaint: r.glb }, words: 'Take this repaint off the model' },
    }));
    pictures.replaceChildren(...viewTiles, ...repaintTiles);
    syncFade();
  }

  /** A painted view's picture (its tile takes the clicks): pointed at, the model shows the view alone. */
  const viewThumb = (v, i, views, hidden) => h('img', {
    class: `ot-view${v.layer ? ' has-layer' : ''}`, src: v.picture, alt: `The ${v.side} view`,
    title: `The ${v.side} view (${i + 1} of ${views.length}), ${hidden
      ? 'hidden with the other painted views: the model shows its own texture in their place.'
      : 'as it went onto the model.'}${
      v.layer ? ` While the pointer is here, the model shows ${hidden ? 'what this view painted' : 'this view alone'}.` : ''
    } ${INPUTS_KEY}`,
    loading: 'lazy', draggable: 'false',
  });

  /**
   * A repaint's picture (its tile takes the clicks): pointed at, the model shows where it went on. The one clicked
   * last is the edge fade's while it shows.
   */
  const repaintThumb = (made, i, all) => h('img', {
    class: `ot-view ot-repaint${made.layer ? ' has-layer' : ''}`, src: made.picture, alt: `Repaint ${i + 1}`,
    title: `Repaint ${i + 1} of ${all.length}: what you masked and had painted again, as your view saw it${
      made.hidden ? ' - hidden: it is not on the model. Click: show it again.' : '. Click: hide it, to compare.'}${
      made.layer ? ` While the pointer is here, the model shows where it ${made.hidden ? 'lies' : 'went on'}.` : ''
    } ${INPUTS_KEY}`,
    loading: 'lazy', draggable: 'false',
  });

  function sync(p) {
    const j = jobFor('omnitexture', id);
    const chosen = omniChosen(p);
    heading.set(p, !canRun() ? '' : chosen ? 'Chosen for Omni-Texture' : 'Not chosen: keeps its own texture',
      thumbUrl(p, 'omnitexture') || thumbUrl(p, 'mesh')
      || (p.images?.chosen ? thumbUrl(p, 'images') || boxFile(p, p.images.chosen, 'images') : null));
    againBtn.hidden = againRow.hidden = !chosen || !omniTextured(p) || !!j || !canRun();
    syncAgain();
    gate.sync();
    error.sync(p);
    busy.sync(p);
    go.sync();
    syncMask(p);
    fasterBox.hidden = !faster.applies();
    faster.sync();
  }

  return {
    key: id, el, go,
    update(p) {
      renderMedia(p);
      renderResult(p);
      renderLive();
      sync(p);
    },
    flush: () => mask.flush(),
    destroy() {
      clearTimeout(liveGone);
      liveUrl = '';
      mask.destroy();
      pane.destroy();
    },
  };
}

// ---- the stage view -------------------------------------------------------------------------------------------

function render() {
  if (!host) return;
  const boxes = boxesOf();
  // Nothing picked and nothing chosen: the viewport asks the artist to choose (the stage's first look).
  const showProp = boxes.length && (ctx.selectedId || boxes.some(omniChosen) || working());
  const prop = showProp ? focusBox(ctx, 'omnitexture', { needsWork: omniChosen, previousId: shownId }).box : null;
  shownId = prop?.id ?? null;
  const key = prop ? prop.id : boxes.length ? 'intro' : 'empty';
  if (view?.key !== key) {
    view?.destroy();
    view = prop ? buildProp(prop) : key === 'intro' ? buildIntro() : buildEmpty();
    host.replaceChildren(view.el);
    fadeIn(view.el);
  }
  view.update(prop);
}

export default {
  id: 'omnitexture',
  title: 'Omni-Texture',
  ctaLabel: 'Retopo',
  ctaHover: 'Goes on to Retopo, where every 3D model gets clean polygons and UVs in the sizes you ask for. Nothing '
    + 'starts by itself there either: its “Retopo All” does. It paints nothing: Omni-Texture only starts from '
    + '“Omni-Texture the chosen”.',
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

  status(prop) {
    return jobState('omnitexture', prop) || omniStatus(prop, {
      canRun: canRun(), meshComing: !!(jobFor('mesh', prop.id) || jobFor('images', prop.id)),
    });
  },

  /**
   * The box beside a prop's tile, ticked while it is chosen for Omni-Texture; none while Omni-Texture can't run here,
   * nor for a tileable texture (it has no 3D model to paint).
   */
  tileCheck: (prop) => (canRun() && !prop.draft && !tileable(prop) ? {
    on: omniChosen(prop),
    title: omniChosen(prop) ? 'Chosen for Omni-Texture: click to leave it out' : 'Choose it for Omni-Texture',
  } : null),

  /**
   * A click on that box chooses the prop or leaves it out (saved like any setting). A prop whose Omni-Texture is
   * queued or being made stays chosen until its job is stopped.
   */
  checkTile(id) {
    const prop = getProp(id);
    if (!prop || prop.draft || !canRun()) return;
    if (omniChosen(prop) && jobFor('omnitexture', id)) {
      toast(`${prop.name} is being OmniTextured. Stop its job (over the model) to leave it out.`, 'info');
      return;
    }
    const chosen = !omniChosen(prop);
    queueSettings(id, 'omnitexture', { chosen });
    if (!chosen) return;
    render();           // (the first prop chosen takes the viewport: its view has the button)
    ring(view?.go);     // what the tick is for: the button that paints the chosen props
  },

  /** Qwen-Image 2.1 starts early only while a chosen prop waits for its Omni-Texture (most props never come here). */
  warmWish: () => (canRun() && omniBatch(boxesOf(), liveDeps()).length ? { tool: 'qwen', stage: 'omnitexture' } : null),

  /** The prop the workbench shows. */
  shownId: () => shownId,

  ctaEnabled() {
    if (canCarry('retopo')) return { enabled: true };
    return {
      enabled: false,
      hint: boxesOf().length ? 'Make the 3D models first.' : 'Make some props in Images first, or import a 3D model of your own.',
    };
  },

  /** Makes the missing 3D models, then retopologizes every prop. Never an Omni-Texture: that is its own button's. */
  async onCta() {
    await view?.flush?.();   // the hand paint still on its way into its files gets there before any job
    await carryDown('retopo');
  },
};
