// App state shared by all modules, plus a minimal event bus.
//
// Events: 'status' (tools / GPU / install; the previous status as argument), 'projects' (list),
// 'project' ({hard, opened}: hard = the prop list changed, opened = another project is open now;
// soft = refresh results in place), 'prop' (prop), 'jobs', 'models'.

import { STAGE_TOOLS, thumbCrop, toolOf } from './stages/rules.js';

export { STAGE_TOOLS };

export const TOOL_NAMES = {
  qwen: 'Qwen-Image 2.1', forge: 'Forge Neo', pixal3d: 'Pixal3D', imesh: 'Instant Meshes', lato2: 'LATO.2', triflow: 'TriFlow',
};
// What each tool is for, in the words the status line, the install card and Settings > Debugging use.
export const TOOL_ROLES = {
  qwen: 'image generator', forge: 'image generator', pixal3d: '3D generator', imesh: 'retopology tool',
  lato2: 'AI retopology tool', triflow: 'AI retopology tool',
};
// The same per stage: Instant Meshes is the texture baker in the Projection stage.
export const STAGE_ROLES = {
  images: 'image generator', mesh: '3D generator', omnitexture: 'image generator', retopo: 'retopology tool',
  projection: 'texture baker',
};
// Said while a tool loads: "Loading the image generator… (the first time takes about a minute)".
export const FIRST_START = {
  qwen: 'this can take a minute', forge: 'the first time takes about a minute', pixal3d: 'about half a minute',
  lato2: 'this can take a minute', triflow: 'this can take a minute',
};
// How long a start takes ("Starting again takes about a minute"). Pixal3D answers as soon as its libraries are
// loaded: it reads each of its models from disk when a job needs it.
export const TOOL_ETA = {
  qwen: 'about a minute', forge: '30-90 seconds', pixal3d: 'about half a minute', imesh: 'a few seconds', lato2: 'about a minute', triflow: 'about a minute',
};
export const RUNNING = new Set(['starting', 'ready', 'busy']);
export const ACTIVE_JOB = new Set(['queued', 'starting_tool', 'running']);
// Where a prop stands in a job (job.props[id].state) once the job is through with it; until then it is
// 'waiting' (not begun), 'working' (on the tool now) or 'resting' (begun, waits for its next step).
export const PROP_ENDED = new Set(['done', 'failed', 'cancelled']);

// The stages from top to bottom, and the tool that runs each one by default (Instant Meshes does two).
// Retopo has a choice of tools (STAGE_TOOLS): each prop's own method runs it (stageTool). Images runs on the
// image generator spellcast3d-config.ini picks (imageTool); Omni-Texture on Qwen-Image 2.1 only (omniTool).
export const STAGES = ['images', 'mesh', 'omnitexture', 'retopo', 'projection', 'rig'];
export const STAGE_TOOL = {
  get images() { return imageTool(); },
  mesh: 'pixal3d',
  get omnitexture() { return omniTool(); },
  retopo: 'imesh',
  projection: 'imesh',
  rig: null,
};

/**
 * The image generator Spellcast3D runs: the one spellcast3d-config.ini picks, the only one the hub has
 * (/api/status stage_tools.images): 'qwen' or 'forge'; null while that file has a mistake (none is
 * installed then: config.problems says which line). Before the hub has answered: the default.
 */
export function imageTool() {
  const st = store.status;
  if (!st?.tools) return STAGE_TOOLS.images[0];
  const listed = st.stage_tools?.images;
  if (Array.isArray(listed)) return listed[0] ?? null;
  return STAGE_TOOLS.images.find((t) => st.tools[t]) ?? null;   // a hub from before stage_tools
}

/**
 * The tool that paints Omni-Textures: Qwen-Image 2.1, while it is the image generator the hub has (/api/status
 * stage_tools.omnitexture), else null (spellcast3d-config.ini picks Forge Neo, or has a mistake). Before the hub
 * has answered: Qwen-Image.
 */
export function omniTool() {
  const st = store.status;
  if (!st?.tools) return 'qwen';
  const listed = st.stage_tools?.omnitexture;
  if (Array.isArray(listed)) return listed[0] ?? null;
  return imageTool() === 'qwen' ? 'qwen' : null;   // a hub from before the stage
}

/** What is wrong in spellcast3d-config.ini, in plain words (/api/status config.problems): [] when nothing. */
export const configProblems = () => (store.status?.config?.problems || []).filter((p) => typeof p === 'string' && p);

/**
 * spellcast3d-config.ini picks something else now than Spellcast3D started with, in plain words ("… now
 * picks Forge Neo. Close Spellcast3D and start it again to switch …"; /api/status config.changed), or ''.
 */
export const configChanged = () => {
  const text = store.status?.config?.changed;
  return typeof text === 'string' ? text : '';
};

/** The tool that does this prop's work in `stage` (Retopo: the prop's method); the stage's default without a prop. */
export const stageTool = (stage, prop = null) => (prop && stage === 'retopo' ? toolOf(stage, prop) : STAGE_TOOL[stage] ?? null);

// The tools one of several doing the same work: the page names them ("Loading LATO.2…", "Loading
// Qwen-Image 2.1…": spellcast3d-config.ini picks the image generator); the others by what they do
// ("Loading the 3D generator…").
const NAMED = new Set(['qwen', 'forge', 'lato2', 'triflow']);

/** A tool in a sentence: "LATO.2", "Qwen-Image 2.1", or "the 3D generator". */
export const toolNoun = (tool) => (NAMED.has(tool) ? toolName(tool) : `the ${TOOL_ROLES[tool] || tool}`);

/**
 * The tool doing `stage`'s work in a sentence: a named one by its name ("Qwen-Image 2.1", "LATO.2"), the
 * stage's default tool by its role ("the retopology tool").
 */
export const stageNoun = (stage, tool = STAGE_TOOL[stage]) => {
  if (tool && NAMED.has(tool)) return toolName(tool);
  return !tool || tool === STAGE_TOOL[stage] ? `the ${STAGE_ROLES[stage]}` : toolNoun(tool);
};

/** "the image generator" -> "The image generator" (a sentence starts with it). */
export const capital = (text) => text.charAt(0).toUpperCase() + text.slice(1);

export const store = {
  status: null,
  connected: true,
  projects: [],
  project: null,
  jobs: [],
  settings: null,
  models: { state: 'unknown', data: null, message: '' },
  bust: {},       // "stage:propId" -> cache-busting stamp for that stage's output files
  // The stamp of every other file URL. It changes each time a project is opened, so a result that was
  // replaced while the page was not looking (a job that finished meanwhile) is never shown from a cache.
  openStamp: Date.now(),
  jobErrors: {},  // "stage:propId" -> last error message for that prop
};

const subs = {};

export function on(evt, fn) {
  (subs[evt] ||= new Set()).add(fn);
  return () => subs[evt].delete(fn);
}

export function emit(evt, arg) {
  // Before any view hears of a change: no prop keeps the thumbnail of a picture it no longer shows
  // (dropStaleThumb), whoever changed it - the hub's answer, a job, a click shown before the hub answered.
  for (const prop of store.project?.props || []) dropStaleThumb(prop);
  if (evt === 'prop') dropStaleThumb(arg);
  for (const fn of [...(subs[evt] || [])]) {
    try { fn(arg); } catch (e) { console.error(`[${evt}]`, e); }
  }
}

export const getProp = (id) => store.project?.props.find((p) => p.id === id) || null;

/** The open project's id (its folder name, used in every URL), or null. */
export const projectId = () => store.project?.id ?? null;

/** The project's draft (the new prop being prepared in Images, before it has its picture), or null. */
export const draftOf = () => store.project?.props.find((p) => p.draft) || null;

/** The props in the props list, in order: every prop except the draft (internally: "boxes"). */
export const boxesOf = () => store.project?.props.filter((p) => !p.draft) || [];

export const toolStatus = (tool) => store.status?.tools?.[tool] || null;

export const toolName = (tool) => toolStatus(tool)?.display_name || TOOL_NAMES[tool] || tool;

export const bustFor = (stage, propId) => store.bust[`${stage}:${propId}`] || store.openStamp;

export function bump(stage, propId) {
  store.bust[`${stage}:${propId}`] = Date.now();
}

// ---- thumbnails (stages/thumbs.js makes them; the hub keeps them in prop.thumbs) -------------------

/**
 * What the thumbnail of a prop's picture must show: its chosen picture, and the part of it while that
 * picture is a sheet sliced into views (its Front slice, rules.thumbCrop): "images/0003.png" or
 * "images/0003.png#0,0,512,512"; null without a chosen picture. The hub keeps it with the thumbnail
 * (thumbs.images.of) and gives the same for a prop (projects.thumb_signature): a thumbnail whose `of` is not
 * this shows something else, whoever changed the picture or its views meanwhile.
 */
export function thumbSignature(images) {
  if (!images?.chosen) return null;
  const crop = thumbCrop(images);
  return crop ? `${images.chosen}#${crop.join(',')}` : images.chosen;
}

/**
 * The prop's thumbnail of `stage` ({path, v}) while it shows what the prop is now, else null: a picture's
 * thumbnail made from another picture or another part of it (thumbSignature) counts as none. (The hub drops
 * a 3D snapshot itself when its stage gets a new result.)
 */
export function thumbOf(prop, stage) {
  const t = prop?.thumbs?.[stage];
  if (!t?.path) return null;
  return stage !== 'images' || t.of === thumbSignature(prop.images) ? t : null;
}

/**
 * Takes the picture's thumbnail off a prop when it shows something else than the prop's picture now
 * (thumbOf): every prop that enters the store, or whose views change in the page, goes through here.
 * Housekeeping only: what shows a prop's small picture asks thumbOf itself at that moment
 * (cropped.propPicture, common.thumbUrl), so a thumbnail left on a prop is never shown for another picture.
 */
export function dropStaleThumb(prop) {
  if (prop?.thumbs?.images && !thumbOf(prop, 'images')) {
    const { images: _stale, ...kept } = prop.thumbs;
    prop.thumbs = kept;
  }
  return prop;
}

/**
 * The job that still has work to do on this prop in this stage, if any: {job, phase, fraction, message}.
 * The hub says where each prop of a job stands (job.props[id]: its state, how far it is, its words), and
 * that is all this reads. phase: 'queued' or 'starting' while the whole job waits in the queue or for its
 * tool; in a job that runs, 'running' (the prop is on the tool now), 'resting' (begun, it waits for its next
 * step: the 3D generator takes every prop of a job through one step before the next) or 'waiting' (not
 * begun). fraction: how far this prop itself is, 0..1. message: the prop's own words, or the job's while it
 * is queued or its tool starts. null once the job is through with the prop (done, failed, cancelled),
 * whatever it still does for its other props.
 */
export function jobFor(stage, propId) {
  for (const job of store.jobs) {
    if (job.stage !== stage || !ACTIVE_JOB.has(job.status)) continue;
    const prop = job.props?.[propId];
    if (!prop || PROP_ENDED.has(prop.state)) continue;
    if (job.status !== 'running') {
      return { job, phase: job.status === 'queued' ? 'queued' : 'starting', fraction: 0, message: job.progress?.message || '' };
    }
    const phase = prop.state === 'working' ? 'running' : prop.state === 'resting' ? 'resting' : 'waiting';
    const fraction = phase === 'waiting' ? 0 : Math.min(1, Math.max(0, Number(prop.fraction) || 0));
    return { job, phase, fraction, message: prop.message || '' };
  }
  return null;
}

/**
 * "Loading the shape model…" while a job's tool reads something from disk (job.loading names it: the 3D
 * generator reads each model when the step that needs it begins), else ''.
 */
export const jobLoading = (job) => (job?.loading ? `Loading the ${job.loading}…` : '');

/**
 * The hub's words while a job is being stopped, else ''. Whether it is: job.stopping alone; what it says:
 * progress.message ("Stopping… (a model that is being saved is finished first)" for a 3D job, "Cancelling… (the
 * current prop may take a moment to stop)" for the others).
 */
export const jobStopping = (job) => (job?.stopping ? job.progress?.message || '' : '');
