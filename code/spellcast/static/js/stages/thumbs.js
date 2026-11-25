// Small pictures of every prop, made in the page and stored on the hub (PUT .../thumbs/<stage>), so the
// page shows them at once from then on: the prop's picture made small for its tile in the
// props list (the chosen picture is about 1 MB; its thumbnail a few KB; of a sheet sliced into views, its
// Front slice: rules.thumbCrop), and a snapshot of each newest
// 3D result (the textured high poly, its Omni-Texture, the retopo's quads over clay, the baked low poly), rendered
// offscreen with viewer3d, for the views' peeks and headings and the tile of a prop made from the
// artist's own model.
// A picture's thumbnail says what it shows (store.thumbSignature: the chosen picture and the part of it),
// and counts only while the prop still shows that (store.thumbOf): another picture chosen, a generation
// finished, the Front slice moved, a picture imported or removed, here or in another tab - the thumbnail
// from before is no thumbnail, the tile shows the chosen picture itself (cut to the same part by CSS:
// cropped.js, so nothing on screen waits for a thumbnail - it only makes the next page load quick), and a
// new one is drawn. The hub
// checks the same (it refuses a thumbnail of what the prop no longer shows, and drops one it has), and
// drops a 3D snapshot when its stage gets a new result. A result that arrives while the page is open (a job
// just finished, another picture chosen, a sheet sliced) is drawn first and soon: until a 3D snapshot is
// there, the page shows the one of the stage before.
// No result is given up for good: one the hub refused because the prop changed meanwhile is simply drawn
// again as the prop is now; one that failed (a broken file, a hub that can't be reached) rests for a while
// after a few tries, then is tried again.

import { fileUrl, propApi } from '../api.js';
import { ACTIVE_JOB, boxesOf, bustFor, getProp, projectId, store, thumbOf, thumbSignature } from '../store.js';
import { flushSettings, hasUnsaved, replaceProp } from '../actions.js';
import { pending } from './common.js';
import { paintingNow } from './paintbar.js';
import { thumbCrop } from './rules.js';

const SIZE = 256;          // px, a 3D snapshot: the views show them at 64-200 px
const PICTURE_SIZE = 384;  // px, the longest side of a picture's thumbnail: a tile is about 200 px wide, crisp on a 2x screen
const START_DELAY = 1500;  // ms of quiet after a change before the first render (the views load first)
const FRESH_DELAY = 150;   // ms before drawing a result that has just arrived
const GAP = 400;           // ms between two renders of a 3D model
const PICTURE_GAP = 60;    // ms between two pictures (they are quick to make)
const MAX_TRIES = 3;       // a result that failed this often in a row (a broken file, a hub that can't keep it) rests ...
const RETRY_PAUSE = 30000; // ... this many ms before each further try

/** Per stage: the result file to draw, and how. The pictures come first: every tile shows one. */
const SOURCES = {
  images: (b) => b.images?.chosen && { rel: b.images.chosen, kind: 'picture', crop: thumbCrop(b.images) },
  mesh: (b) => b.mesh?.glb && { rel: b.mesh.glb, kind: 'glb' },
  omnitexture: (b) => b.omnitexture?.glb && { rel: b.omnitexture.glb, kind: 'glb' },
  retopo: (b) => b.retopo?.obj && { rel: b.retopo.obj, kind: 'obj' },
  projection: (b) => b.projection?.glb && { rel: b.projection.glb, kind: 'glb' },
};

let queue = new Map();       // "project/prop/stage" -> task, in the order they are drawn
const failures = new Map();  // result -> {n, at}: tries in a row that failed, and when the last one did
let lacked = new Map();      // "project/prop/stage" -> the result that lacked its thumbnail at the last sync
let seenProject = null;      // the project of the last sync: what it lacks when it opens is not news
let timer = null;
let due = 0;                 // when `timer` starts drawing
let running = false;
let settled = Promise.resolve();   // the workbench stopped moving (no model is loaded while it moves)

/** A job of this stage is still going, so its tool wants the graphics card to itself. */
const stageBusy = (stage) => (store.jobs || []).some((j) => j.stage === stage && ACTIVE_JOB.has(j.status));

/**
 * The thumbnail `prop` lacks in `stage` now, or null: it has no result there, its result is being made right
 * now (it is replaced in a moment: drawn once it is there), or its thumbnail shows it (store.thumbOf).
 * A 3D snapshot also waits while a job of its stage runs: drawing one loads the whole model into a second
 * 3D context, and the tool needs that video memory - the sweep after the job ends draws them all. And while a
 * model is being painted by hand (paintbar.js paintingNow): loading one stalls the page for a moment, which a
 * stroke would show; the page's next sync draws it.
 * {url, kind, crop: what to draw; of: what a picture's thumbnail shows (store.thumbSignature); result: the
 * same in one string, with the result's file}.
 */
export function thumbTask(pid, prop, stage) {
  const src = SOURCES[stage](prop);
  if (!src || thumbOf(prop, stage) || pending(stage, prop)) return null;
  if (src.kind !== 'picture' && (stageBusy(stage) || paintingNow())) return null;
  const url = fileUrl(pid, prop.id, src.rel, bustFor(stage, prop.id));
  const of = stage === 'images' ? thumbSignature(prop.images) : null;
  return { pid, id: prop.id, stage, url, kind: src.kind, crop: src.crop || null, of, result: `${stage}|${url}|${of ?? ''}` };
}

/**
 * Queues a thumbnail for every prop result that has none. Call it whenever the project may have changed;
 * the work happens in the background, one at a time: results that have just arrived first, then the
 * pictures, then the snapshots of `first` (a stage), then the rest. stillSettled: the shell's
 * ctx.settled (loading a model while the workbench moves would stall it).
 */
export function syncThumbs(first = null, stillSettled = null) {
  if (stillSettled) settled = stillSettled;
  const pid = projectId();
  if (!pid) return;
  const opening = pid !== seenProject;
  seenProject = pid;
  const fresh = [];
  const rest = [];
  const lacking = new Map();
  const rank = (s) => (s === 'images' ? 0 : s === first ? 1 : 2);
  const stages = Object.keys(SOURCES).sort((a, b) => rank(a) - rank(b));
  for (const stage of stages) {
    for (const prop of boxesOf()) {
      const task = thumbTask(pid, prop, stage);
      if (!task) continue;
      const key = `${pid}/${prop.id}/${stage}`;
      lacking.set(key, task.result);
      if (resting(task.result)) continue;
      // News: what the prop did not lack a moment ago (its picture, or the part of it, has just changed).
      (!opening && lacked.get(key) !== task.result ? fresh : rest).push(task);
    }
  }
  lacked = lacking;
  const before = queue;
  queue = new Map();
  for (const task of fresh) queue.set(`${pid}/${task.id}/${task.stage}`, task);
  // What waited keeps its place while it is still what its prop lacks (the same picture and part of it).
  for (const [key, task] of before) if (task.pid === pid && rest.some((t) => t.result === task.result)) queue.set(key, task);
  for (const task of rest) {
    const key = `${pid}/${task.id}/${task.stage}`;
    if (!queue.has(key)) queue.set(key, task);
  }
  if (queue.size && !running) {
    // Sooner for news, never later: the page syncs on every change and every few seconds, and a wait that
    // started anew each time would never end.
    const wait = fresh.length ? FRESH_DELAY : START_DELAY;
    if (timer === null || Date.now() + wait < due) {
      clearTimeout(timer);
      due = Date.now() + wait;
      timer = setTimeout(pump, wait);
    }
  }
}

/**
 * How a try for `task` ended, from the hub's answer to its upload (its prop, before the page's own unsaved
 * changes are laid over it): 'kept'; 'changed' - refused because the hub's prop shows another picture, or
 * another part of it, than was drawn (the page is a moment ahead of the hub, or behind it): no fault of the
 * picture, the next sync draws what the prop shows then; else 'failed'.
 */
export function tryOutcome(prop, task) {
  if (kept(prop, task)) return 'kept';
  return task.stage === 'images' && thumbSignature(prop?.images) !== task.of ? 'changed' : 'failed';
}

/** Notes how a try for `result` ended (tryOutcome; a try that threw: 'failed'). Only failures count. */
export function noteTry(result, outcome, now = Date.now()) {
  if (outcome === 'kept') failures.delete(result);
  else if (outcome === 'failed') failures.set(result, { n: (failures.get(result)?.n || 0) + 1, at: now });
}

/** `result` failed MAX_TRIES times in a row and the last time less than RETRY_PAUSE ago: not tried now. */
export function resting(result, now = Date.now()) {
  const f = failures.get(result);
  return !!f && f.n >= MAX_TRIES && now - f.at < RETRY_PAUSE;
}

async function pump() {
  if (running) return;
  timer = null;
  running = true;
  let viewer = null;
  try {
    while (queue.size) {
      if (document.hidden) {   // nobody sees the page now: go on when the tab is back
        await new Promise((resolve) => document.addEventListener('visibilitychange', resolve, { once: true }));
        continue;
      }
      await settled;
      if (!queue.size) break;
      const [key, queued] = queue.entries().next().value;
      queue.delete(key);
      if (projectId() !== queued.pid) continue;
      const picture = queued.kind === 'picture';
      // The hub checks a picture's thumbnail against the views it has: views drawn a moment ago go there first.
      if (picture) await flushSettings().catch(() => {});
      // What the prop lacks now, not what it lacked when this was queued: the picture may have changed since.
      const now = projectId() === queued.pid ? getProp(queued.id) : null;
      const task = now && !now.draft && !(picture && hasUnsaved()) ? thumbTask(queued.pid, now, queued.stage) : null;
      if (!task) continue;   // nothing to draw any more, or not yet: the next sync queues what is missing then
      let outcome = 'failed';
      try {
        let blob;
        if (picture) {
          blob = await pictureThumbnail(task.url, task.crop);
        } else {
          viewer ||= await import('../viewer3d.js');
          blob = await viewer.renderThumbnail(task.url, task.kind, { size: SIZE });
        }
        const prop = await upload(task, blob);
        // Kept by the hub: should it ever be missing again, the hub dropped it, and it is drawn again.
        // Refused because its prop shows another picture than this page drew: that does not count against
        // the picture (tryOutcome), which is drawn again once the two agree.
        outcome = tryOutcome(prop, task);
        // The hub's prop, also when it refused the thumbnail (its prop shows another picture than this page
        // thought: now the page has that one, and draws its thumbnail next).
        if (projectId() === task.pid) replaceProp(prop);
      } catch (e) {
        console.warn(`No thumbnail for ${key}:`, e);
      }
      noteTry(task.result, outcome);
      // Not kept (the hub has not heard yet of what this page shows, e.g. a picture chosen this instant): a
      // moment longer before the next try.
      await new Promise((resolve) => setTimeout(resolve, picture && outcome === 'kept' ? PICTURE_GAP : GAP));
    }
  } finally {
    running = false;
    viewer?.releaseThumbnails();
  }
}

/** The hub's answer to an upload has the thumbnail of `task` (a picture's: of what the task drew). */
export function kept(prop, task) {
  const t = prop?.thumbs?.[task.stage];
  return !!t?.path && (task.stage !== 'images' || t.of === task.of);
}

/**
 * The picture at `url` made small (its longest side PICTURE_SIZE px at most), as a WEBP: all of it, or its
 * part `crop` ([x, y, w, h] in its pixels, kept inside it).
 */
async function pictureThumbnail(url, crop = null) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`the hub answered ${res.status}`);
  const bitmap = await createImageBitmap(await res.blob());
  try {
    let [x, y, cw, ch] = crop || [0, 0, bitmap.width, bitmap.height];
    x = Math.min(Math.max(0, x), bitmap.width - 1);
    y = Math.min(Math.max(0, y), bitmap.height - 1);
    cw = Math.max(1, Math.min(cw, bitmap.width - x));
    ch = Math.max(1, Math.min(ch, bitmap.height - y));
    const s = Math.min(1, PICTURE_SIZE / Math.max(cw, ch));
    const w = Math.max(1, Math.round(cw * s));
    const hh = Math.max(1, Math.round(ch * s));
    const canvas = new OffscreenCanvas(w, hh);
    const g = canvas.getContext('2d');
    g.imageSmoothingQuality = 'high';
    g.drawImage(bitmap, x, y, cw, ch, 0, 0, w, hh);
    return await canvas.convertToBlob({ type: 'image/webp', quality: 0.85 });
  } finally {
    bitmap.close();
  }
}

/**
 * PUT .../thumbs/<stage> with the picture, and for a picture's thumbnail what it shows (?of=: the hub keeps
 * it only while its prop shows that); resolves with the hub's prop.
 */
async function upload(task, blob) {
  const of = task.of ? `?of=${encodeURIComponent(task.of)}` : '';
  const res = await fetch(`${propApi(task.pid, task.id)}/thumbs/${task.stage}${of}`,
    { method: 'PUT', headers: { 'Content-Type': blob.type }, body: blob });
  if (!res.ok) throw new Error(`the hub answered ${res.status}`);
  return res.json();
}
