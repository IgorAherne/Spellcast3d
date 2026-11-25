// The Omni-Texture viewer's tools, on the prop's result - or on its 3D model, which the first tool taken up makes a
// result of its own (bind's `begin`: paintbar.js) -: Projection's paint bar (stages/paintbar.js) - the marker and the
// clone stamp, whose strokes are saved into the step shown as hand paint (the texels painted alone: the hub keeps
// them as a layer over the steps it makes again) - with a tool of its own at the column's foot, the mask marker. It
// paints magenta over what the artist wants made again (the model's overlay, paint3d.js, on the page alone; the x to
// its left discards the mask), and once something is masked "Render Masked" stands at the glass's left end. That sends the hub what is masked (the overlay's
// texels covered at least half way) and the viewport's camera in the model's own space; the hub's job has Qwen-Image
// paint the masked surface again as that camera sees it (adapters/omnitexture.py retouch) and the prop gets a new
// result, one more of its omnitexture.steps. While the mask marker is up, the viewport outside the square that camera
// sees is covered in black and the view holds still: a slight move while masking, and the mask would no longer hide
// all it hid, and Qwen-Image would see the rest. "Render Masked" sends the view as it is at the press - the marker put
// down, the artist may have turned the model first - and puts the marker down.
// Ctrl+Z takes back the newest change first: a stroke (paint3d.js keeps them per model), or a repaint, a picture
// taken off the model or the whole result taken off it (their × on the right: stages/omnitexture.js), each a step of
// its own - the prop then shows the step before it again with its mask as it was, stroke by stroke (the page keeps a
// model's mask when it leaves the model, and puts it on again when that model is loaded once more). Ctrl+Y goes
// forward the same way. The magenta is only ever on the page: nothing is saved of it but the mask a repaint was made
// from.

import { api, propApi, putBytes } from '../api.js';
import { replaceProp, startJob } from '../actions.js';
import { spinner } from '../controls.js';
import { h } from '../dom.js';
import { icon } from '../icons.js';
import { projectId } from '../store.js';
import { toast, toastError } from '../ui.js';
import { isNewest, undoList, undoStamp } from '../undo.js';
import { createPaintBar } from './paintbar.js';
import { TAKEN_OFF } from './rules.js';

export const MASK_COLOUR = '#ff00ff';
const MASKED = 128;                // an overlay texel covered this far or more is masked (half way)
const HINT = 'Mask marker: the view is locked · paint over what to make again, then “Render Masked”';
const LOST = 'The mask went with it: paint it again then.';

const diverged = new WeakSet();    // sessions masked again after a Ctrl+Z: Ctrl+Y no longer leads on from them
const stamps = new Map();          // "project/prop/glb" of a step's model -> the stamp of the change that made it
const kept = new Map();            // "project/prop/glb" -> the mask on that model as the page left it (paint3d keep)
const KEPT = 6;                    // ... of that many models at most: each holds its strokes' tiles
const lastShown = new Map();       // "project/prop" -> the model last shown of it: tells a repaint's result as it arrives

/** The texels masked, from the overlay's coverage (paint3d.js readOverlay) -> {mask: a byte per texel, 255 masked; count}. */
export function maskOf(cover) {
  const mask = new Uint8Array(cover.length);
  let count = 0;
  for (let t = 0; t < cover.length; t++) {
    if (cover[t] < MASKED) continue;
    mask[t] = 255;
    count++;
  }
  return { mask, count };
}

/** The tangent of half the angle the largest centred square of a `width` x `height` viewport spans (fov: vertical). */
export const squareTan = (fovDegrees, width, height) => Math.tan((fovDegrees * Math.PI) / 360) * (Math.min(width, height) / (height || 1));

/** Where `glb` is among the prop's steps, and the ones before and after it -> {index, before, after} (null: none). */
export function stepsAround(steps, glb) {
  const list = Array.isArray(steps) ? steps : [];
  const index = list.indexOf(glb);
  return { index, before: index > 0 ? list[index - 1] : null, after: index >= 0 && index < list.length - 1 ? list[index + 1] : null };
}

/**
 * A prop's models were made again under other names (a repaint's fade, stages/omnitexture.js: `renamed`, step as
 * it was -> step as it is): what the page keeps per model - the repaints' places in Ctrl+Z's order, the masks - follows.
 */
export function renameModels(project, propId, renamed) {
  const prop = `${project}/${propId}`;
  for (const [was, now] of Object.entries(renamed || {})) {
    for (const map of [stamps, kept]) {
      if (!map.has(`${prop}/${was}`)) continue;
      map.set(`${prop}/${now}`, map.get(`${prop}/${was}`));
      map.delete(`${prop}/${was}`);
    }
    if (lastShown.get(prop) === was) lastShown.set(prop, now);
  }
}

/** The page took a prop's result off as a whole (stages/omnitexture.js): the newest change, Ctrl+Z's first. */
export function tookOff(project, propId) {
  stamps.set(`${project}/${propId}/${TAKEN_OFF}`, undoStamp());
}

/**
 * Saves the hand paint on the step `glb` of the prop's Omni-Texture: the texels painted alone (readAll's patch, alpha
 * 255 where painted, 0 elsewhere) -> the prop. Nothing is sent when no texel differs from the files (a stroke taken
 * back before it was saved).
 */
export function saveHandPaint({ project, propId, glb }, image) {
  if (!image.patch) return null;
  const query = `glb=${encodeURIComponent(glb)}&width=${image.width}&height=${image.height}`;
  return putBytes(`${propApi(project, propId)}/omnitexture/paint?${query}`, image.patch);
}

/** A mask (a byte per texel) as a PNG's bytes. */
async function maskPng(mask, size) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const context = canvas.getContext('2d');
  const image = context.createImageData(size, size);
  for (let t = 0, i = 0; t < mask.length; t++, i += 4) {
    image.data[i] = image.data[i + 1] = image.data[i + 2] = mask[t];
    image.data[i + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
  if (!blob) throw new Error('The mask could not be made into a picture.');
  return new Uint8Array(await blob.arrayBuffer());
}

/** The viewer's camera in the space of the model on screen: {eye, forward, up, tan} (tan: squareTan). */
function cameraOf(viewer) {
  const object = viewer.current.children[0];
  object.updateWorldMatrix(true, false);
  viewer.camera.updateMatrixWorld();
  const inverse = object.matrixWorld.clone().invert();
  const v = viewer.camera.position.clone();
  const eye = viewer.camera.getWorldPosition(v).applyMatrix4(inverse).toArray();
  const forward = viewer.camera.getWorldDirection(v).transformDirection(inverse).toArray();
  const up = v.set(0, 1, 0).transformDirection(viewer.camera.matrixWorld).transformDirection(inverse).toArray();
  const canvas = viewer.renderer.domElement;
  return { eye, forward, up, tan: squareTan(viewer.camera.fov, canvas.clientWidth || canvas.width, canvas.clientHeight || canvas.height) };
}

/**
 * The tools of one Omni-Texture view: {el, cover (nodes for the viewport), bind({viewer, project, propId, glb, steps,
 * step?, select(view), begin?}), unbind(), settle(), flush(), destroy()}. step: the prop's step the model stands for
 * when it is not the model itself (TAKEN_OFF: the 3D model, shown for a result taken off). cover: black outside the square "Render Masked" sends,
 * while the mask marker is up; select: picks a map view (the map bar's own); busy(): the stage works on the prop
 * (its job starting, a fade, a picture being taken off): nothing paints and Ctrl+Z waits. settle(): see below.
 */
export function createMaskBar({ busy: stageBusy = () => false } = {}) {
  let binding = null;       // the step shown: {viewer, project, propId, glb, steps, step, select}
  let token = 0;
  let working = false;      // the mask is on its way to the hub, or a step is being taken

  const key = (glb) => `${binding.project}/${binding.propId}/${glb}`;
  const stepShown = () => binding.step ?? binding.glb;
  const around = () => (binding ? stepsAround(binding.steps, stepShown()) : { index: -1, before: null, after: null });
  /** The stamp of the change that made the step shown (0: it is the first step, or one from before this visit). */
  const renderStamp = () => (binding && around().before ? stamps.get(key(stepShown())) || 0 : 0);

  const glyph = h('span', { class: 'mk-glyph' }, icon('sparkles'));
  const render = h('button', {
    type: 'button', class: 'btn primary small mk-render', hidden: true,
    title: 'Qwen-Image repaints what you masked, as you see it from here, and it goes onto the model',
    onclick: () => renderMasked(),
  }, glyph, 'Render Masked');
  const marker = h('button', {
    type: 'button', class: 'pt-tool', 'aria-label': 'Mask marker',
    title: 'Mask marker: paint over what should be made again, then “Render Masked”. The view is locked while it '
      + 'is up (Esc puts it down)',
  }, icon('mask'));
  // While the marker is up, the viewport outside the square "Render Masked" sends (squareTan) is covered in black,
  // so the artist frames what is to be painted inside it.
  const cover = h('div', { class: 'mk-cover', hidden: true }, h('div'));
  const bar = createPaintBar({
    save: saveHandPaint,
    patches: true,
    overlay: {
      button: marker, colour: MASK_COLOUR, hint: HINT, slot: render, lost: LOST, still: true,
      wipe: 'Discard the mask (Ctrl+Z brings it back)',
      up: () => { cover.hidden = false; },
      down: () => { cover.hidden = true; },
    },
    busy: () => working || stageBusy(),
    undo: walk,
    changed: (layer) => {
      const s = bar.session();
      if (layer === 'overlay' && s && around().after) diverged.add(s);
      sync();
    },
  });

  function sync() {
    const busy = working || stageBusy();
    render.hidden = !bar.session()?.painted('overlay');
    render.disabled = busy;
    marker.disabled = busy;
    const spinning = working && !render.hidden;
    if (glyph.dataset.busy !== String(spinning)) {
      glyph.dataset.busy = String(spinning);
      glyph.replaceChildren(spinning ? spinner() : icon('sparkles'));
    }
  }

  // ---- "Render Masked" ------------------------------------------------------------------------------------------
  async function renderMasked() {
    const s = bar.session();
    if (working || stageBusy() || !s || !binding || bar.painting()) return;
    const { viewer, project, propId, glb } = binding;
    const mine = token;
    working = true;
    sync();
    try {
      await bar.flush();         // the hand paint into the files first: the hub takes none while the job is under way
      const { data, width } = await s.readOverlay();
      const { mask, count } = maskOf(data);
      if (!count) {
        toast('Nothing is masked yet: paint over what should be made again first.', 'info');
        return;
      }
      const camera = cameraOf(viewer);
      const png = await maskPng(mask, width);
      if (mine !== token) return;
      const sent = await putBytes(`${propApi(project, propId)}/omnitexture/mask?glb=${encodeURIComponent(glb)}`, png);
      bar.pick(null);
      diverged.delete(s);          // what follows this mask from now on is the repaint made of it
      keepMask();
      await startJob('omnitexture', [propId], { params: { retouch: { mask: sent.mask, ...camera } } });
    } catch (e) {
      kept.delete(`${project}/${propId}/${glb}`);
      console.warn(e);
      toastError(e);
    } finally {
      working = false;
      sync();
    }
  }

  // ---- Ctrl+Z / Ctrl+Y: after the strokes, the prop's steps ------------------------------------------------------
  /** The mask on the model shown, kept for when the page shows that model again (a step retraced, a model made again). */
  function keepMask() {
    if (!binding) return;
    const mark = key(binding.glb);
    kept.delete(mark);
    const s = bar.session();
    const mask = s?.keep('overlay');
    if (!mask || !(mask.steps.length || mask.undone.length || mask.strokes.overlay)) return;
    kept.set(mark, { ...mask, diverged: diverged.has(s) });
    for (const oldest of kept.keys()) {
      if (kept.size <= KEPT) break;
      kept.delete(oldest);
    }
  }

  /** The prop shows another of its steps (`forward`: a repaint again, the newest change once more). */
  async function step(to, forward = false) {
    const { project, propId } = binding;
    working = true;
    sync();
    keepMask();
    try {
      if (forward) stamps.set(`${project}/${propId}/${to}`, undoStamp());
      const prop = await api('POST', `${propApi(project, propId)}/omnitexture/step`, { glb: to });
      if (projectId() === project) replaceProp(prop);
    } catch (e) {
      toastError(e);
    } finally {
      working = false;
      sync();
    }
  }

  /** Ctrl+Z / Ctrl+Y once no stroke on the model is the change to take back or bring back (paintbar.js asks). */
  function walk(which) {
    if (!binding) return false;
    const { before, after } = around();
    if (which === 'undo') {
      const repaint = renderStamp();
      if (!repaint || !isNewest(repaint)) return false;
      step(before);
      return true;
    }
    const s = bar.session();
    if (after == null || (s && diverged.has(s))) return false;
    step(after, true);
    return true;
  }
  const dropUndo = undoList(renderStamp);

  // ---- binding to a model -----------------------------------------------------------------------------------
  async function bind(next) {
    if (binding && next && binding.glb === next.glb && binding.propId === next.propId && binding.viewer === next.viewer) {
      binding = { ...binding, ...next };
      sync();
      return;
    }
    unbind();
    if (!next?.viewer) return;
    const mine = token;
    await bar.bind(next);
    if (mine !== token) return;
    binding = { ...next };
    // A step seen for the first time after the one before it (a repaint, a picture taken off): the newest change
    // on the page, so Ctrl+Z takes it back first.
    const shown = key(stepShown());
    const prop = `${binding.project}/${binding.propId}`;
    const { before } = around();
    if (before && lastShown.get(prop) === before && !stamps.has(shown)) stamps.set(shown, undoStamp());
    lastShown.set(prop, stepShown());
    // A mask kept of this model goes onto it again, strokes and all, when the viewer loaded the model anew (while
    // the viewer held it, the mask is still on: put() then refuses).
    const mark = key(binding.glb);
    const mask = kept.get(mark);
    kept.delete(mark);
    const s = mask && bar.session(true);
    if (s?.put(mask) && mask.diverged) diverged.add(s);
    sync();
  }

  function unbind() {
    token++;
    bar.unbind();
    binding = null;
    sync();
  }

  return {
    el: bar.el,
    cover,
    bind,
    unbind,
    /**
     * The page is about to have the hub make the prop's steps again (a fade, a picture taken off, a job): the hand
     * paint goes into the files first - the hub takes none while that is under way - and the mask on the model is
     * kept for when it comes back -> once the paint is in the files.
     */
    async settle() {
      await bar.flush();
      keepMask();
    },
    flush: () => bar.flush(),
    destroy() {
      if (!cover.hidden) bar.pick(null);       // the marker is this view's: no other arrives with its view locked
      unbind();
      dropUndo();
      bar.destroy();
    },
  };
}
