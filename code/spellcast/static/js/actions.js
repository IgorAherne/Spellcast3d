// Everything that talks to the hub and updates the store: projects, props (and the draft),
// settings autosave, jobs (with polling), tools and the status poll.

import { api, projectApi, propApi, uploadFiles } from './api.js';
import { ACTIVE_JOB, PROP_ENDED, RUNNING, bump, emit, getProp, projectId, store, toolName, toolStatus } from './store.js';
import { confirmDialog, toast } from './ui.js';
import { plural } from './dom.js';
import { nameFromPrompt } from './stages/rules.js';

// ---- projects ---------------------------------------------------------------------------------

function setProject(project) {
  project.props ||= [];
  project.ui ||= {};
  store.project = project;
  store.jobs = [];
  store.jobErrors = {};
  // New file URLs after every open: a result replaced while another project was open is never shown
  // from the 3D viewer's cache or an old snapshot (the cache drops old entries by itself).
  store.bust = {};
  store.openStamp = Date.now();
}

async function refreshProjects() {
  store.projects = (await api('GET', '/api/projects')) || [];
  emit('projects');
}

const UNSAVED = 'Your latest changes are not saved yet. Make sure the black Spellcast3D console window is '
  + 'open, then try again.';
let openSeq = 0;   // only the project picked last is opened, whatever order the answers come in

async function switchTo(seq) {
  await flushSettings();
  if (hasUnsaved()) throw new Error(UNSAVED);   // they belong to the project on screen
  return seq === openSeq;
}

function opened(project) {
  setProject(project);
  emit('project', { hard: true, opened: true });
  pollJobsNow();
  refreshProjects().catch(() => {});   // the menu's recent list follows what was opened last
}

/** Opens `load()`'s project unless another pick came after it (only the last pick wins). */
async function openWith(load) {
  const seq = ++openSeq;
  try {
    if (!await switchTo(seq)) return;
    const project = await load();
    if (seq === openSeq) opened(project);
  } catch (e) {
    if (seq !== openSeq) return;   // a newer pick won: no error for this one
    emit('projects');              // the project menu shows the open project again
    throw e;
  }
}

/** The project opened last, or a fresh untitled one: the workbench is ready without any question. */
export const openLastProject = () => openWith(() => api('POST', '/api/projects/open-last'));

export const openProject = (id) => openWith(() => api('GET', projectApi(id)));

/** A new project: untitled without a name (it gets one later with "Save as"). */
export const createProject = (name) => openWith(() => api('POST', '/api/projects', name ? { name } : {}));

/** Names the open project (also renames its folder). The hub refuses while one of its jobs runs. */
export async function renameProject(name) {
  await flushSettings();
  const fresh = await api('POST', `${projectApi(projectId())}/rename`, { name });
  fresh.props ||= [];
  fresh.ui = store.project.ui;
  for (const prop of fresh.props) applyPending(prop);
  store.project = fresh;
  emit('project', { hard: false });
  refreshProjects().catch(() => {});
}

/** Re-reads the project after the hub changed it (job finished...), keeping local UI state and unsaved edits. */
export async function reloadProject() {
  if (!store.project) return;
  const id = projectId();
  const fresh = await api('GET', projectApi(id));
  if (projectId() !== id) return;
  fresh.props ||= [];
  fresh.ui = store.project.ui;
  for (const prop of fresh.props) applyPending(prop);
  const sameProps = fresh.props.map((p) => p.id).join() === store.project.props.map((p) => p.id).join();
  store.project = fresh;
  emit('project', { hard: !sameProps });
}

let uiUnsaved = false;   // the view position could not be saved (hub unreachable): retried on reconnect
const saveUiSoon = (() => {
  let timer = null;
  return () => {
    clearTimeout(timer);
    const id = projectId();
    timer = setTimeout(() => {
      if (projectId() !== id) return;
      uiUnsaved = false;
      api('PATCH', projectApi(id), { ui: store.project.ui }).catch((e) => {
        if (e.status === 0) uiUnsaved = true;
        console.warn('Could not save the view position:', e.message);
      });
    }, 400);
  };
})();

/** Merges into project.ui and persists it (debounced). */
export function updateUi(patch) {
  Object.assign(store.project.ui, patch);
  saveUiSoon();
}

// ---- props and the draft ----------------------------------------------------------------------

/** Replaces a prop in the store with the hub's version (keeping unsaved edits) and notifies views. */
export function replaceProp(prop) {
  const props = store.project?.props;
  const i = props ? props.findIndex((p) => p.id === prop.id) : -1;
  if (i < 0) return;
  applyPending(prop);
  props[i] = prop;
  emit('prop', prop);
}

/** A new prop at the end of the list, named `name` (the hub's settings for it are the last prop's) -> the prop. */
export async function addProp(name) {
  const pid = projectId();
  const prop = await api('POST', `${projectApi(pid)}/props`, { name });
  if (projectId() === pid && store.project && !store.project.props.some((p) => p.id === prop.id)) {
    store.project.props.push(prop);
    emit('project', { hard: true });
  }
  return prop;
}

export async function renameProp(propId, name) {
  replaceProp(await api('PATCH', propApi(projectId(), propId), { name }));
}

/** Deletes a prop -> the name the hub keeps it under in the project's "deleted" folder, for restoreProp. */
export async function deleteProp(propId) {
  const { deleted } = await api('DELETE', propApi(projectId(), propId));
  delete pending[propId];
  delete pendingViews[propId];
  store.project.props = store.project.props.filter((p) => p.id !== propId);
  emit('project', { hard: true });
  return deleted;
}

/** Brings back a prop deleteProp took away (`deleted`: the name it gave) at its place in the list. */
export async function restoreProp(deleted) {
  await api('POST', `${projectApi(projectId())}/deleted/${encodeURIComponent(deleted)}/restore`);
  await reloadProject();
}

/**
 * Imports pictures into a prop in one upload: each becomes one of its pictures. Two or more also become
 * its views: `sides` gives the side each one shows (null: none; slicer.importSides), else front, ← left,
 * back, right →, in the order given. The front one (else the first) becomes the chosen picture.
 */
export async function uploadImages(propId, files, onProgress, sides = null) {
  const path = `${propApi(projectId(), propId)}/images/upload`;
  bump('images', propId);
  replaceProp(await uploadFiles(path, files.map((file) => ({ file, path: file.name || 'image.png' })), {
    field: 'files', onProgress, fields: sides ? { sides: JSON.stringify(sides) } : {},
  }));
}

/**
 * Adds example pictures to a prop (Describe + example: images.references, in this order, after the ones it
 * has; the hub refuses more than 10 in all). onProgress follows the upload.
 */
export async function addReferences(propId, files, onProgress = null) {
  const path = `${propApi(projectId(), propId)}/images/references`;
  const prop = await uploadFiles(path, files.map((file) => ({ file, path: file.name || 'image.png' })), { field: 'files', onProgress });
  bump('refs', propId);   // a number freed by a removal may be used again
  replaceProp(prop);
}

/** Removes one example picture of a prop (`path`, as images.references has it), or all of them. */
export async function removeReferences(propId, path = null) {
  const query = path ? `?path=${encodeURIComponent(path)}` : '';
  replaceProp(await api('DELETE', `${propApi(projectId(), propId)}/images/references${query}`));
}

/**
 * The viewport's picture with the artist's marks painted over it (a PNG), for "Re-Render Viewport" -> its path in
 * the prop, the job's from_viewport. `mask` (Smart Mask: where the artist brushed, a PNG) goes up with it.
 */
export async function saveViewport(propId, png, mask = null) {
  const form = new FormData();
  form.append('file', png, 'viewport.png');
  if (mask) form.append('mask', mask, 'mask.png');
  return (await api('POST', `${propApi(projectId(), propId)}/images/viewport`, form)).path;
}

/**
 * Saves a prop's picture (`path`: one of its pictures, or the viewport's from saveViewport) into its export folder,
 * which the hub then shows in File Explorer -> the hub's answer (preview.toastSaved).
 */
export async function savePicture(propId, path) {
  return api('POST', `${propApi(projectId(), propId)}/images/save`, { path });
}

const choosing = {};   // propId -> the picture just chosen here, until the hub has answered

/**
 * Chooses one of a prop's pictures. The page shows it at once (the mark in the strip, the picture) and the
 * hub's answer confirms it; until then an answer to something asked earlier (a thumbnail, a save) cannot put
 * the picture from before back (applyPending). Refused: the page shows what the hub has again.
 */
export async function selectImage(propId, path) {
  const prop = getProp(propId);
  if (prop && prop.images?.chosen !== path && prop.images?.candidates?.includes(path)) {
    choosing[propId] = path;
    prop.images = { ...prop.images, chosen: path };
    emit('prop', prop);
  }
  try {
    const fresh = await api('POST', `${propApi(projectId(), propId)}/images/select`, { path });
    if (choosing[propId] === path) delete choosing[propId];
    replaceProp(fresh);
  } catch (e) {
    if (choosing[propId] === path) {
      delete choosing[propId];
      reloadProject().catch(() => {});
    }
    throw e;
  }
}

/**
 * Takes one of a prop's pictures (`path`, as images.candidates has it) out of the prop, with its views; its
 * file stays with the hub, for restoreImage. When it was the chosen one, the hub chooses its neighbour.
 */
export async function removeImage(propId, path) {
  await flushSettings();   // views drawn a moment ago are saved first: the hub then drops this picture's
  delete choosing[propId];
  replaceProp(await api('POST', `${propApi(projectId(), propId)}/images/remove`, { path }));
}

/**
 * Brings back a picture removeImage took out: at `index` of the prop's pictures, with the `views` it had,
 * and chosen again when `chosen`.
 */
export async function restoreImage(propId, path, { index = null, views = [], chosen = false } = {}) {
  await flushSettings();
  if (chosen) delete choosing[propId];
  replaceProp(await api('POST', `${propApi(projectId(), propId)}/images/restore`, { path, index, views, chosen }));
}

// ---- settings autosave (PATCH props, 500 ms debounce) and the views (PUT images/views, 300 ms) ----

const pending = {};   // propId -> {stage: {key: value}}
const pendingViews = {};   // propId -> the views of its pictures, not sent yet
const sendingViews = {};   // propId -> the views on their way to the hub
let saveTimer = null;
let viewsTimer = null;
let inflight = Promise.resolve();

function applyPending(prop) {
  for (const [stage, patch] of Object.entries(pending[prop.id] || {})) {
    prop[stage] ||= {};
    prop[stage].settings = { ...(prop[stage].settings || {}), ...patch };
  }
  const views = pendingViews[prop.id] || sendingViews[prop.id];
  if (views) prop.images = { ...(prop.images || {}), views };
  const chosen = choosing[prop.id];
  if (chosen && prop.images?.candidates?.includes(chosen)) prop.images = { ...prop.images, chosen };
}

/** Records a settings edit locally right away and saves it to the hub shortly after. */
export function queueSettings(propId, stage, patch) {
  const prop = getProp(propId);
  if (!prop) return;
  prop[stage] ||= {};
  prop[stage].settings = { ...(prop[stage].settings || {}), ...patch };
  const p = (pending[propId] ||= {});
  p[stage] = { ...(p[stage] || {}), ...patch };
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flushSettings, 500);
}

let placing = Promise.resolve();   // the placements on their way to the hub, one after the other

/**
 * Where the prop's model stands in its saved files (the Move / Rotate handles of Retopo and Export; null: as it was
 * made): kept on the prop at once and sent to the hub, which saves every file with it from then on.
 */
export function savePlacement(propId, placement) {
  const prop = getProp(propId);
  if (!prop) return placing;
  prop.placement = placement;
  const project = projectId();
  placing = placing.then(() => api('PATCH', propApi(project, propId), { placement })).catch((e) => {
    toast(`Where the model stands could not be saved: ${e.message || e}`, 'error');
  });
  return placing;
}

/**
 * Records the views sliced from a prop's pictures ([{view, src, rect}], see rules.VIEW_ORDER) locally
 * right away, so the page shows them, and saves them to the hub shortly after.
 */
export function queueViews(propId, views) {
  const prop = getProp(propId);
  if (!prop) return;
  prop.images = { ...(prop.images || {}), views };
  pendingViews[propId] = views;
  clearTimeout(viewsTimer);
  viewsTimer = setTimeout(flushSettings, 300);
  emit('prop', prop);
}

let unsavedWarned = false;   // one "not saved yet" toast per outage (the banner says the rest)

/** Edits the hub has not got yet (it could not be reached): a job must not start with the old values. */
export const hasUnsaved = () => Object.keys(pending).length > 0 || Object.keys(pendingViews).length > 0;

function take(map, key) {
  const value = map[key];
  delete map[key];
  return value;
}

/**
 * The hub's answer to a save replaces the prop quietly: the workbench already shows these values. Unless the
 * answer has other pictures, or another chosen one, than the page knew (the hub got them meanwhile), or another
 * clean mesh in use (the hub picks the one of a prop that now asks for a single size): then the views are told, so
 * the strip, the picture and the pick never lag behind the store.
 */
function saved(project, prop) {
  unsavedWarned = false;
  if (projectId() !== project) return;
  applyPending(prop);
  const props = store.project.props;
  const i = props.findIndex((p) => p.id === prop.id);
  if (i < 0) return;
  const pictures = (p) => `${p.images?.chosen}|${(p.images?.candidates || []).join()}|${p.retopo?.obj}|${p.retopo?.accepted}`;
  const changed = pictures(props[i]) !== pictures(prop);
  props[i] = prop;
  if (changed) emit('prop', prop);
}

/** A save failed: kept (under any newer edit) when the hub was unreachable, else said and dropped. */
function notSaved(project, e, keep) {
  // A 4xx (prop deleted, bad value) would fail forever: dropped. Unreachable / 5xx: kept.
  if (projectId() === project && (e.status === 0 || e.status >= 500)) {
    keep();
    if (!unsavedWarned) {
      unsavedWarned = true;
      toast(`Not saved yet: ${e.message} Your changes are kept and saved as soon as Spellcast3D answers again.`, 'error');
    }
    return false;
  }
  toast(`Your last change was not saved: ${e.message}`, 'error');
  return true;
}

/**
 * Sends every queued edit now (the settings and the views). Resolves once the hub has them. An edit the
 * hub could not be reached for is kept (under any newer edit) and sent again when it answers; other
 * failures are shown as a toast.
 */
export function flushSettings() {
  clearTimeout(saveTimer);
  clearTimeout(viewsTimer);
  const ids = [...new Set([...Object.keys(pending), ...Object.keys(pendingViews)])];
  if (!ids.length || !store.project) return inflight;
  // A prop's settings, then its views: one answer after the other, so the last one has both.
  const batch = ids.map((propId) => [propId, take(pending, propId), take(pendingViews, propId)]);
  for (const [propId, , views] of batch) if (views) sendingViews[propId] = views;
  const project = projectId();
  inflight = inflight.then(() => Promise.all(batch.map(async ([propId, stages, views]) => {
    if (stages) {
      const body = Object.fromEntries(Object.entries(stages).map(([stage, settings]) => [stage, { settings }]));
      try {
        saved(project, await api('PATCH', propApi(project, propId), body));
      } catch (e) {
        notSaved(project, e, () => {
          const p = (pending[propId] ||= {});
          for (const [stage, settings] of Object.entries(stages)) p[stage] = { ...settings, ...(p[stage] || {}) };
        });
      }
    }
    if (views) {
      try {
        saved(project, await api('PUT', `${propApi(project, propId)}/images/views`, { views }));
      } catch (e) {
        // Refused: the page shows the views the hub has again.
        if (notSaved(project, e, () => { pendingViews[propId] ||= views; }) && projectId() === project) {
          reloadProject().catch(() => {});
        }
      } finally {
        if (sendingViews[propId] === views) delete sendingViews[propId];
      }
    }
  })));
  return inflight;
}

addEventListener('pagehide', () => flushSettings());

// ---- jobs -------------------------------------------------------------------------------------

// What a finished job made: [one prop, several props].
const MADE = {
  images: ['Pictures', 'Pictures'], mesh: ['3D model', '3D models'], omnitexture: ['Omni-Texture', 'Omni-Textures'],
  retopo: ['Clean mesh', 'Clean meshes'],
  projection: ['Baked textures', 'Baked textures'],
};
const HIDDEN_POLL = 30000;   // a tab nobody looks at checks only now and then
let jobTimer = null;
let jobPolls = 0;         // numbers the job polls: an answer older than one already handled is dropped
let jobPollHandled = 0;
// job id -> {status, states: prop id -> where it stood, polled} from the previous poll, and `shown`: the props
// whose result showed the moment they finished, while the job went on
let seen = new Map();
let polledProject = null;
const mine = new Set();   // jobs started from this page, so one that finishes between two polls still reports
const reported = new Set();   // jobs whose end was announced (toast, prop errors) - never twice

function pollJobsNow() {
  clearTimeout(jobTimer);
  pollJobs();
}

async function pollJobs() {
  const project = projectId();
  if (!project) return;
  const seq = ++jobPolls;
  let jobs = null;
  try {
    jobs = await api('GET', `/api/jobs?project=${encodeURIComponent(project)}`);
  } catch { /* the status poll reports connection problems */ }
  if (projectId() !== project) return;
  if (jobs && seq > jobPollHandled) {
    jobPollHandled = seq;
    handleJobs(project, jobs);
  }
  const busy = store.jobs.some((j) => ACTIVE_JOB.has(j.status));
  clearTimeout(jobTimer);
  jobTimer = setTimeout(pollJobs, document.hidden ? HIDDEN_POLL : busy ? 1000 : 6000);
}

const statesOf = (job) => Object.fromEntries(Object.entries(job.props || {}).map(([id, p]) => [id, p?.state]));

function handleJobs(project, jobs) {
  const first = polledProject !== project;
  polledProject = project;
  if (first) seen = new Map();
  let reload = false;
  if (!first) reload = reportVanished(project, jobs);
  // "stage:prop id" of the props a newer job still has work on (the hub lists the newest job first): how it
  // goes there is what the tile says, not what an older job said of it (a prop made again after it failed).
  const taken = new Set();
  for (const job of jobs) {
    const before = seen.get(job.id);
    const states = statesOf(job);
    const shown = before?.shown || new Set();
    const finished = !ACTIVE_JOB.has(job.status);
    // On the first poll of a project only this page's own jobs are announced (they may have finished
    // while another project was open); older jobs were announced already, or long ago.
    if (finished && !reported.has(job.id)
        && (first ? mine.has(job.id) : before ? ACTIVE_JOB.has(before.status) : mine.has(job.id))) {
      finishJob(job, shown, taken);
      reload = true;
    } else if (!finished) {
      // A prop of a job that goes on with its other props: its result shows, and what went wrong with it is
      // said on its tile, the moment the hub says so - not when the whole job has ended. This holds for a
      // job seen for the first time too (one started in another window), but not on the first poll of a
      // project: it was just read, with everything its jobs have finished so far.
      for (const [id, state] of Object.entries(states)) {
        if (state === before?.states[id]) continue;
        const key = `${job.stage}:${id}`;
        if (state === 'done' && !first) {
          bump(job.stage, id);
          shown.add(id);
          reload = true;
        } else if (state === 'failed' && job.errors?.[id] && !taken.has(key)) {
          store.jobErrors[key] = job.errors[id];
        }
      }
    }
    if (!finished) for (const id of Object.keys(states)) taken.add(`${job.stage}:${id}`);
    seen.set(job.id, { status: job.status, states, shown, polled: true });
  }
  store.jobs = jobs;
  emit('jobs');
  if (reload) reloadProject().catch((e) => console.warn(e));
}

/**
 * Jobs the hub no longer lists although it listed them running: jobs live in the hub's memory, so it
 * was restarted (its console window closed and opened again) and they were stopped. The props they were
 * not through with (job.props: any state but done, failed, cancelled) get a note, and one toast says so.
 * Returns true when the project should be re-read.
 */
function reportVanished(project, jobs) {
  const ids = new Set(jobs.map((j) => j.id));
  const gone = store.jobs.filter((j) => j.project === project && ACTIVE_JOB.has(j.status) && !ids.has(j.id)
    && seen.get(j.id)?.polled);
  if (!gone.length) return false;
  let props = 0;
  for (const job of gone) {
    reported.add(job.id);
    seen.delete(job.id);
    mine.delete(job.id);
    for (const id of job.prop_ids) {
      if (PROP_ENDED.has(job.props?.[id]?.state)) continue;
      store.jobErrors[`${job.stage}:${id}`] = 'Spellcast3D was closed while this was being made, so it was not '
        + 'finished. Press the button again to make it.';
      props += 1;
    }
  }
  if (props) {
    toast(`Spellcast3D was restarted, so ${plural(props, 'prop')} ${props === 1 ? 'was' : 'were'} not finished. `
      + 'Their tiles say which; run them again.', 'warn');
  }
  return true;
}

/**
 * A job has ended: its results show, its props say what went wrong, and a toast sums it up. shown: the props
 * whose result showed already, when they finished (it is not loaded a second time); taken: the props a newer
 * job still has work on ("stage:prop id": handleJobs).
 */
function finishJob(job, shown = new Set(), taken = new Set()) {
  reported.add(job.id);
  // A new prop whose first pictures were being made may have no name yet: the one its prompt gives it.
  const names = (ids) => ids.map((id) => {
    const p = getProp(id);
    return p?.name?.trim() || nameFromPrompt(p?.images?.settings?.prompt) || 'The new prop';
  });
  for (const id of job.prop_ids) if (!shown.has(id)) bump(job.stage, id);
  const errors = Object.entries(job.errors || {});
  const note = (id, msg) => { if (!taken.has(`${job.stage}:${id}`)) store.jobErrors[`${job.stage}:${id}`] = msg; };
  for (const [id, msg] of errors) note(id, msg);
  if (job.status === 'error' && !errors.length && job.progress?.message) {
    for (const id of job.prop_ids) note(id, job.progress.message);
  }
  // What the tool said about a finished result (job.warnings: "this took longer: fewer example pictures are
  // faster", "Stopped after 2 of 4 pictures"): the job's own last words are only "Done.".
  const warnings = Object.entries(job.warnings || {}).filter(([, w]) => typeof w === 'string' && w);
  for (const [id, w] of warnings.slice(0, 3)) {
    toast(job.prop_ids.length > 1 ? `${names([id])[0]}: ${w}` : w, 'warn', 9000);
  }
  if (job.status === 'cancelled') {
    // Qwen-Image 2.1 finishes all the variations of a prop together at the end: what it was drawing is gone.
    if (!warnings.length) {
      toast(job.stage === 'images' && job.tool === 'qwen' ? 'Stopped. Pictures Qwen-Image 2.1 had not finished are not kept.'
        : 'Stopped. Anything finished before that is kept.', 'info');
    }
  } else if (errors.length) {
    // One toast per distinct problem: a tool that failed to start fails every prop the same way.
    const byMessage = new Map();
    for (const [id, msg] of errors) byMessage.set(msg, [...(byMessage.get(msg) || []), id]);
    const groups = [...byMessage];
    for (const [msg, ids] of groups.slice(0, 3)) {
      toast(ids.length === 1 ? `${names(ids)[0]}: ${msg}` : `${msg} (${plural(ids.length, 'prop')})`, 'error');
    }
    if (groups.length > 3) toast('More props failed for other reasons. Hover them to see why.', 'error');
  } else if (job.status === 'error') {
    toast(job.progress?.message || 'The job failed. Settings > Debugging shows the tool log. Please try again.', 'error');
  } else if (job.status === 'done') {
    // The pictures made for the prop being prepared are right there on the workbench.
    if (job.prop_ids.every((id) => getProp(id)?.draft)) return;
    const one = job.prop_ids.length === 1;
    const who = one ? names(job.prop_ids)[0] : plural(job.prop_ids.length, 'prop');
    toast(`${MADE[job.stage]?.[one ? 0 : 1] || 'Results'} ready for ${who}.`, 'success');
  }
}

/**
 * Queues a job of `stage` for the props. tool: the tool that runs it (Retopo: the props' method, one job
 * per method; the hub refuses props set to another one); without it the hub uses the stage's own tool.
 * params: extra job parameters.
 */
export async function startJob(stage, propIds, { tool = null, params = null } = {}) {
  await flushSettings();
  if (hasUnsaved()) throw new Error(`${UNSAVED} Nothing was started.`);
  const body = { stage, project: projectId(), prop_ids: propIds };
  if (tool) body.tool = tool;
  if (params) body.params = params;
  const job = await api('POST', '/api/jobs', body);
  mine.add(job.id);
  for (const id of propIds) delete store.jobErrors[`${stage}:${id}`];
  store.jobs = [job, ...store.jobs.filter((j) => j.id !== job.id)];
  seen.set(job.id, { status: job.status, states: statesOf(job) });
  emit('jobs');
  pollJobsNow();
  return job;
}

export async function cancelJob(job) {
  if (job.prop_ids.length > 1) {
    const ok = await confirmDialog({
      title: 'Stop this batch?',
      message: `This stops the whole batch of ${plural(job.prop_ids.length, 'prop')}. Props that are already finished keep their results.`,
      ok: 'Stop batch', danger: true,
    });
    if (!ok) return;
  }
  const updated = await api('POST', `/api/jobs/${encodeURIComponent(job.id)}/cancel`);
  store.jobs = store.jobs.map((j) => (j.id === updated.id ? updated : j));
  emit('jobs');
  pollJobsNow();
}

// ---- tools ------------------------------------------------------------------------------------

function patchTool(tool, data) {
  if (!store.status?.tools || !data || typeof data !== 'object') return;
  store.status.tools[tool] = { ...store.status.tools[tool], ...data };
  emit('status');
}

export async function startTool(tool) {
  patchTool(tool, await api('POST', `/api/tools/${tool}/start`));
  pollStatusSoon(700);
}

/**
 * Asks the hub to have a tool ready before a job asks for it (`wish`: rules.js warmAsk). Nothing is shown, and a
 * request that fails is only logged.
 */
export function warmTool(tool, wish = {}) {
  api('POST', `/api/tools/${tool}/warm`, wish)
    .then(({ warmed, ...status }) => {
      if (!warmed) return;
      patchTool(tool, status);
      pollStatusSoon(700);
    })
    .catch((e) => console.warn(`Could not start ${tool} early:`, e));
}

async function stopTool(tool) {
  patchTool(tool, await api('POST', `/api/tools/${tool}/stop`));
  pollStatusSoon(700);
}

/**
 * Stop and start again (e.g. so new settings apply). The hub starts it once the stop has finished; with a `wish`
 * (rules.js warmAsk: e.g. the image generator's {stage}) it starts it for that, as an arrival at the stage would.
 */
export async function restartTool(tool, wish = null) {
  await stopTool(tool);
  if (!wish) {
    await startTool(tool);
    return;
  }
  const status = await api('POST', `/api/tools/${tool}/warm`, { ...wish, displace: true });
  delete status.warmed;
  patchTool(tool, status);
  pollStatusSoon(700);
}

/** Asks before stopping a tool, or before a start that stops another GPU tool. Returns true if acted. */
export async function toggleTool(tool) {
  const st = toolStatus(tool);
  if (!st) return false;
  const name = toolName(tool);
  if (RUNNING.has(st.state)) {
    const ok = await confirmDialog({
      title: `Stop ${name}?`,
      message: `This frees the memory ${name} uses. Anything it is generating right now will be stopped.`
        // Instant Meshes keeps the viewers' guide strokes only in memory.
        + (tool === 'imesh' ? ' Retopo guide strokes you have not saved are lost.' : '')
        + ' It starts again automatically when you need it.',
      ok: `Stop ${name}`, danger: true,
    });
    if (ok) await stopTool(tool);
    return ok;
  }
  if (st.state === 'stopping') return false;
  if (st.state === 'installing') {
    await confirmDialog({
      title: `${name} is still being installed`,
      message: 'It starts by itself as soon as its installation is finished. Anything that needs it waits until then.',
      ok: 'OK', cancel: null,
    });
    return false;
  }
  if (st.state === 'not_installed' || st.state === 'unsupported' || st.supported === false || st.installed === false) {
    await confirmDialog({
      title: `${name} can't start`,
      message: st.reason || st.message || `${name} is not installed. Close Spellcast3D and run run-spellcast3d.bat again to finish the installation.`,
      ok: 'OK', cancel: null,
    });
    return false;
  }
  const others = gpuToolsToStop(tool);
  if (others.length) {
    const ok = await confirmDialog({
      title: `Start ${name}?`,
      message: `Only one GPU tool runs at a time, so ${others.join(' and ')} will be stopped to free video memory.`,
      ok: `Start ${name}`,
    });
    if (!ok) return false;
  }
  await startTool(tool);
  return true;
}

/**
 * The GPU tools a start of `tool` stops: those on the graphics card (holds_gpu), when `tool` takes the card as it
 * starts (gpu_at_start). One that takes it only at its first job (Pixal3D) starts beside them, and a tool running
 * without the card is not stopped (a status without holds_gpu counts every running GPU tool).
 */
function gpuToolsToStop(tool) {
  const tools = store.status?.tools || {};
  if (!tools[tool]?.uses_gpu || tools[tool].gpu_at_start === false) return [];
  if (store.settings?.gpu_exclusive === false) return [];
  return Object.entries(tools)
    .filter(([id, t]) => id !== tool && t.uses_gpu && RUNNING.has(t.state) && t.holds_gpu !== false)
    .map(([id]) => toolName(id));
}

// ---- status poll ------------------------------------------------------------------------------

let statusTimer = null;

export function pollStatusSoon(ms = 0) {
  clearTimeout(statusTimer);
  statusTimer = setTimeout(pollStatus, ms);
}

async function pollStatus() {
  try {
    const status = await api('GET', '/api/status');
    const before = store.status;
    const wasOffline = !store.connected;
    store.status = status;
    store.connected = true;
    emit('status', before);
    if (wasOffline) {   // back: send what could not be saved meanwhile
      flushSettings();
      if (uiUnsaved) saveUiSoon();
    }
  } catch (e) {
    if (e.status === 0) {
      store.connected = false;
      emit('status', store.status);
    }
  }
  const starting = Object.values(store.status?.tools || {}).some((t) => t.state === 'starting' || t.state === 'stopping')
    || !!store.status?.multiview?.downloading || !!store.status?.qwen_full?.downloading
    || !!store.status?.flux3?.downloading || !!store.status?.hi3d?.downloading || !!store.status?.hi3d?.finishing;
  pollStatusSoon(document.hidden ? HIDDEN_POLL : starting || !store.connected ? 1500 : 2500);
}

/**
 * Starts the one-time download of Pixal3D's multi-view models (about 22 GB): the hub fetches them in the
 * background, and the status poll follows it (status.multiview: {available, downloading, done_bytes,
 * total_bytes, error}).
 */
export async function downloadMultiview() {
  const multiview = await api('POST', '/api/pixal3d/multiview/download');
  if (store.status && multiview) {
    store.status.multiview = multiview;
    emit('status');
  }
  pollStatusSoon(1500);
}

/**
 * Starts the one-time download of Hi3D's AI models (about 11.4 GB, into models\hi3d): the hub fetches them in the
 * background and then measures Hi3D's convolutions on the graphics card; the status poll follows it (status.hi3d:
 * {available, code, downloading, waiting, finishing, done_bytes, total_bytes, error}).
 */
export async function downloadHi3d() {
  const hi3d = await api('POST', '/api/hi3d/download');
  if (store.status && hi3d) {
    store.status.hi3d = hi3d;
    emit('status');
  }
  pollStatusSoon(1500);
}

// The Images stage's models that are a download of their own: its route and its key in /api/status.
const MODEL_DOWNLOADS = {
  full: ['/api/qwen/full/download', 'qwen_full'],
  flux3: ['/api/flux3/download', 'flux3'],
};

/**
 * Starts the download of an optional image model - 'full', the full Qwen-Image 2.1 (about 4 GB at the default size;
 * the installer brings only the Turbo), into models\qwen-image, or 'flux3', FLUX.3 Image Klein (about 7 GB), into
 * models\flux3-klein: the hub fetches it in the background and opens its folder in File Explorer, where it arrives;
 * the status poll follows it (status.qwen_full / status.flux3: {available, downloading, waiting, done_bytes,
 * total_bytes, error}; flux3 also says "engine").
 */
export async function downloadModel(model) {
  const [path, key] = MODEL_DOWNLOADS[model];
  const { opened, ...state } = await api('POST', path) || {};   // "opened" (File Explorer) is no part of the status
  if (store.status && state.total_bytes) {
    store.status[key] = state;
    emit('status');
  }
  pollStatusSoon(1500);
}

// A hidden tab polls every 30 s (enough to notice a finished job); coming back catches up at once.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) return;
  pollStatusSoon(0);
  if (store.project) pollJobsNow();
});

// ---- Forge model list (Forge Neo only: spellcast3d-config.ini picks the image generator) -------

let modelsReq = null;

export function loadModels(refresh = false) {
  if (modelsReq && !refresh) return modelsReq;
  store.models = { ...store.models, state: 'loading' };
  emit('models');
  modelsReq = api(refresh ? 'POST' : 'GET', refresh ? '/api/forge/refresh' : '/api/forge/models')
    .then((data) => { store.models = { state: 'ready', data, message: '' }; })
    .catch((e) => { store.models = { state: e.status === 409 ? 'offline' : 'error', data: null, message: e.message }; })
    .finally(() => { modelsReq = null; emit('models'); });
  return modelsReq;
}
