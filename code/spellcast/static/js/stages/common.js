// Pieces the stage views share: going down to the next stage with every prop (the calls to
// action), a prop's job state (the props list's overlays), the tool gates (installing, can't run here),
// which prop a later stage shows, and small building blocks (busy layer, error line, empty states, the
// numbers of a result, Donate in the viewport's corner, the pictures of the window of what Qwen-Image was
// given). ("box" in the code is a prop: the props were drawn as boxes once.)

import { api, fileUrl } from '../api.js';
import { formatBytes, h, plural } from '../dom.js';
import { icon } from '../icons.js';
import {
  FIRST_START, PROP_ENDED, STAGE_TOOL, boxesOf, bustFor, capital, configChanged, configProblems, imageTool, jobFor,
  jobLoading, jobStopping, projectId, stageNoun, store, thumbOf, toolName, toolStatus,
} from '../store.js';
import { croppedPicture, propPicture } from '../cropped.js';
import { cancelJob, startJob } from '../actions.js';
import { attempt } from '../ui.js';
import { progressBar, spinner } from '../controls.js';
import { FASTER, fasterOn } from './faster.js';
import { carryPlan, planCount, planSummary, reaches, rowPlan } from './rules.js';

// A view that replaced another one fades in (the shell's motion, quicker with reduced motion).
export { fadeIn } from '../shell/anim.js';


/** URL of a file of a prop (rel as stored in project.json), cache-busted per stage result. */
export const boxFile = (box, rel, stage) => fileUrl(projectId(), box.id, rel, bustFor(stage, box.id));

/**
 * URL of the page's snapshot of a prop's `stage` result (thumbs.js), or null. A picture's thumbnail
 * ('images') only while it shows the prop's picture as it is now (store.thumbOf).
 */
export function thumbUrl(box, stage) {
  const t = thumbOf(box, stage);
  return t ? fileUrl(projectId(), box.id, t.path, t.v) : null;
}

export const findBox = (id) => (id ? boxesOf().find((b) => b.id === id) || null : null);

// ---- going down with every prop ----------------------------------------------------------------

/** A job of `stage` is queued or running for this prop. */
export const pending = (stage, box) => !!jobFor(stage, box.id);

/** What the rules (rules.js) need to know about the hub now: jobs, tools and Pixal3D's multi-view models. */
export const liveDeps = () => ({
  pending, open: (stage, tool) => gateOpen(toolGate(stage, tool)), multiview: !!store.status?.multiview?.available,
});

/**
 * Takes every prop down to the stage `upTo` ('mesh' | 'retopo' | 'projection'): queues the work every prop still
 * lacks on the way there (rules.carryPlan: the missing 3D models, or the bakes). A tool that can't run here
 * queues nothing.
 */
export async function carryDown(upTo) {
  for (const [stage, ids, tool] of carryPlan(upTo, boxesOf(), liveDeps())) await startJob(stage, ids, { tool });
}

/** Some prop has a result of `upTo`, or carryDown(upTo) would give it one. */
export const canCarry = (upTo) => reaches(upTo, boxesOf(), liveDeps());

/**
 * The stage's own button over its call to action (view.rowAction), which works on the props right in
 * this stage, without going on: every prop ready for `stage` that lacks a fresh result there
 * (rules.rowPlan). null when there is none, or when it is only the prop on the workbench, whose own
 * button does it. words: the start of its label ("Bake" -> "Bake 3 props"); icon: its icon.
 */
export function rowAction(stage, shownId, words, iconName) {
  const work = () => rowPlan(stage, boxesOf(), liveDeps());
  const plan = work();
  const count = planCount(plan, stage);
  if (!count || (count === 1 && plan[0][1][0] === shownId)) return null;
  return {
    label: `${words} ${plural(count, 'prop')}`,
    icon: iconName,
    hint: planSummary(plan, toolName),
    async run() {
      for (const [s, now, tool] of work()) await startJob(s, now, { tool });   // what is still missing when pressed
    },
  };
}

// ---- per-prop state for the props list's overlays ----------------------------------------------

/**
 * A tool's progress message without the percentage some tools end it with ("Baking the textures… 18%"):
 * the progress bars show the prop's own share of the job, and one number is enough.
 */
export const withoutPercent = (msg) => msg.replace(/\s*\(?\d+(?:\.\d+)?\s*%\)?\s*$/, '');

/**
 * {state, message, fraction} of this prop's queued / running job of `stage`, or its last error; else null.
 * 'processing': the tool works on it now; 'resting': the job has begun it and it waits for its next step, as
 * far as it got (the 3D generator takes every prop of a job through one step before the next); 'waiting':
 * not begun.
 */
export function jobState(stage, box) {
  const j = jobFor(stage, box.id);
  if (j) {
    const msg = j.message || '';
    const installing = /install/i.test(msg) ? msg : '';
    if (j.phase === 'running') return { state: 'processing', message: withoutPercent(msg) || 'Working…', fraction: j.fraction };
    if (j.phase === 'resting') return { state: 'resting', message: restingMessage(j.job), fraction: j.fraction };
    if (j.phase === 'starting') {
      return { state: 'processing', message: installing || `Loading ${stageNoun(stage, j.job.tool)}…`, fraction: null };
    }
    if (installing) return { state: 'waiting', message: installing };
    // Retopo runs one tool after the other: a prop of the next one waits for the whole group before it.
    const ahead = j.phase === 'queued' && stage === 'retopo' ? (store.jobs || []).find((o) => o !== j.job
      && o.stage === 'retopo' && ['running', 'starting_tool'].includes(o.status) && o.tool && o.tool !== j.job.tool) : null;
    if (ahead) return { state: 'waiting', message: `Waits for ${toolName(ahead.tool)} to finish` };
    return { state: 'waiting', message: j.phase === 'queued' ? 'Queued · its turn comes soon' : 'Waiting for its turn' };
  }
  const err = store.jobErrors?.[`${stage}:${box.id}`];
  return err ? { state: 'error', message: err } : null;
}

/**
 * The prop a running job works on now: {id, name, words, fraction} - its own words ("Refining fine details · step
 * 24 of 25") and how far it is, as its tile says; null while none works (a network loads, say).
 */
export function onCard(job) {
  const id = job?.status === 'running' ? job.current_prop : null;
  const prop = id ? job.props?.[id] : null;
  if (prop?.state !== 'working') return null;
  const name = (store.project?.props || []).find((p) => p.id === id)?.name || 'Another prop';
  return { id, name, words: withoutPercent(prop.message || ''), fraction: Math.min(1, Math.max(0, Number(prop.fraction) || 0)) };
}

const lowerFirst = (t) => t.charAt(0).toLowerCase() + t.slice(1);

/**
 * What a prop that waits for its next step is told. The 3D generator takes every prop of the job through one
 * step before the next, so a prop that is done with a step waits while the others have theirs - minutes in a
 * batch. The words say who is at work meanwhile and how far, on two lines (a line break the busy layer keeps:
 * white-space pre-line): "Waiting for the next step · Plant:" / "refining fine details · step 24 of 25" (onCard);
 * else the step (job.part); the plain words when the hub says no more.
 */
export function restingMessage(job, lead = 'Waiting for the next step') {
  const busy = onCard(job);
  if (busy?.words) return `${lead} · ${busy.name}:\n${lowerFirst(busy.words)}`;
  const part = job?.part;
  const label = part?.label ? lowerFirst(part.label) : '';
  if (!label) return lead;
  const name = (store.project?.props || []).find((p) => p.id === job.current_prop)?.name;
  const count = part.of > 1 && part.position ? ` (${part.position} of ${part.of})` : '';
  return `${lead}${name ? ` · ${name}:` : ''}\n${label}${count}`;
}

// ---- tools: installing, loading, can't run here ------------------------------------------------

/**
 * Whether `tool` (default: the tool of `stage`; in Retopo, a prop's method) can take work: {kind: 'ok' |
 * 'installing' | 'failed' | 'missing' | 'unsupported' | 'config' | 'unknown', text, detail?, fraction?}. An
 * installing tool still takes jobs (they wait for it); the others explain themselves in the centre.
 * 'config': Images has no image generator, because spellcast3d-config.ini has a mistake (its words say which line);
 * Omni-Texture has no Qwen-Image 2.1, because that file picks Forge Neo (or has a mistake).
 */
export function toolGate(stage, tool = STAGE_TOOL[stage]) {
  if (!tool && stage === 'omnitexture' && store.status?.tools) {
    return {
      kind: 'config',
      text: `Omni-Texture paints with Qwen-Image 2.1, and spellcast3d-config.ini ${imageTool() === 'forge'
        ? 'picks Forge Neo as the image generator' : 'has a mistake'}.`,
      detail: 'To use it, switch on the line backend = qwen there, save the file, then close Spellcast3D and start it again.',
    };
  }
  if (!tool && stage === 'images' && store.status?.tools) {
    return {
      kind: 'config',
      text: 'No image generator: spellcast3d-config.ini has a mistake.',
      detail: `${configProblems().join(' ') || 'It names no image generator: write backend = qwen or backend = forge.'} `
        + 'Fix it, save the file, then close Spellcast3D and start it again.',
    };
  }
  if (!tool) return { kind: 'ok', text: '' };
  const noun = stageNoun(stage, tool);   // "the retopology tool", "LATO.2"
  const st = toolStatus(tool);
  if (!st) return { kind: 'unknown', text: '' };
  const comp = store.status?.install?.components?.[tool];
  if (st.state === 'unsupported' || st.supported === false || comp?.status === 'unsupported') {
    return { kind: 'unsupported', text: st.reason || st.message || `${capital(noun)} can't run on this computer.` };
  }
  if (st.state === 'installing' || (st.installed === false && ['pending', 'running'].includes(comp?.status))) {
    const f = typeof comp?.fraction === 'number' ? comp.fraction : null;
    const bytes = comp?.bytes_total ? ` · ${formatBytes(comp.bytes_done || 0)} of ${formatBytes(comp.bytes_total)}` : '';
    return {
      kind: 'installing', fraction: f, bytes,
      text: `${capital(noun)} is installing…${f == null ? '' : ` ${Math.round(f * 100)}%`}`,
      detail: `${comp?.step || comp?.message || 'Spellcast3D installs it in the background.'}${bytes}`,
    };
  }
  if (st.state === 'not_installed' || st.installed === false) {
    if (comp?.status === 'failed') {
      return {
        kind: 'failed',
        text: `${capital(noun)} could not be installed${comp.message ? `: ${comp.message}` : '.'}`,
        detail: 'Close Spellcast3D and start it again to try once more. Settings > Debugging shows the details.',
      };
    }
    return {
      kind: 'missing',
      text: st.message || `${capital(noun)} is not installed yet. Close Spellcast3D and start it again to finish the installation.`,
    };
  }
  return { kind: 'ok', text: '' };
}

/** The tool can take work now, or will once it is installed. */
export const gateOpen = (gate) => ['ok', 'unknown', 'installing'].includes(gate.kind);

const LOADING_SAID_FROM = 5;   // seconds of a start before its lines say how long it has taken

/**
 * "Loading the image generator… (the first time takes about a minute)" while the tool starts, else ''. A start
 * that began a while ago (the hub loads the next stage's tool ahead) says how far it is: "...; 45 s so far)", so
 * it does not read as a load begun on arriving here.
 */
export function loadingText(stage, tool = STAGE_TOOL[stage]) {
  const st = tool ? toolStatus(tool) : null;
  if (st?.state !== 'starting') return '';
  const sofar = st.starting_seconds >= LOADING_SAID_FROM ? `${st.starting_seconds} s so far` : '';
  const said = [FIRST_START[tool], sofar].filter(Boolean).join('; ');
  return `Loading ${stageNoun(stage, tool)}…${said ? ` (${said})` : ''}`;
}

// ---- the prop a later stage shows --------------------------------------------------------------

/**
 * Which prop a stage view (3D, Retopo, Projection) shows: the selected one; else the prop a job that goes
 * from prop to prop is on right now; else the one shown before (`previousId`) while a running job of this
 * stage has it among its props, finished there or not; else the first prop such a job is not through with;
 * else the one shown before; else the first prop that `needsWork`; else the first prop.
 * A job that takes every prop through one step before the next (job.part: the 3D generator) is not
 * followed - its props advance together, so the workbench stays where it is, and a prop that finishes
 * there never changes which prop is shown: its 3D model loads only when it is the prop on the workbench.
 * Returns {box, auto} (auto: the artist did not pick it) or {box: null}.
 */
export function focusBox(ctx, stage, { needsWork, previousId = null } = {}) {
  const boxes = boxesOf();
  const chosen = findBox(ctx.selectedId);
  if (chosen) return { box: chosen, auto: false };
  const working = boxes.find((b) => {
    const j = jobFor(stage, b.id);
    return j?.phase === 'running' && !j.job.part;
  });
  if (working) return { box: working, auto: true };
  const before = findBox(previousId);
  const running = (store.jobs || []).filter((j) => j.stage === stage && j.status === 'running');
  if (before && running.some((j) => j.props?.[before.id])) return { box: before, auto: true };
  for (const job of running) {
    const open = (id) => !!job.props?.[id] && !PROP_ENDED.has(job.props[id].state);
    const next = (job.prop_ids || []).filter(open).map(findBox).find(Boolean);
    if (next) return { box: next, auto: true };
  }
  if (before) return { box: before, auto: true };
  return { box: boxes.find((b) => needsWork?.(b)) || boxes[0] || null, auto: true };
}

// ---- building blocks ---------------------------------------------------------------------------

/**
 * Options for modelPane's show() while the workbench moves to another stage (ctx.moving): the model waits
 * until it has stopped (loading it would stall the move), with the prop's snapshot `snapshot` shown meanwhile.
 */
export const afterMove = (ctx, snapshot) => (ctx?.moving ? { after: ctx.settled, snapshot } : {});

export function emptyState(iconName, title, text, ...actions) {
  return h('div', { class: 'sv-empty' },
    iconName ? icon(iconName, 'big') : null,
    h('h3', null, title),
    text ? h('p', null, text) : null,
    actions.length ? h('div', { class: 'sv-empty-actions' }, actions) : null);
}

const DONATE_URL = 'https://stableprojectorz.com/thanks';

/**
 * Donate, in the bottom-left corner of a stage's viewport (every stage box has one: the picture, the model,
 * the Instant Meshes viewer, the empty states), across from the navigation hint. A plain link: it opens in
 * the browser, whatever the hub is doing; not draggable, so a drag that starts on it does nothing.
 */
export const donateCorner = () => h('a', {
  class: 'btn donate', href: DONATE_URL, target: '_blank', rel: 'noopener', draggable: 'false',
  title: 'Support Spellcast3D (opens in a new tab)',
}, icon('heart'), 'Donate');

const DISCORD_URL = 'https://discord.gg/yg4dABJpyZ';

/**
 * "Join Discord": under Donate in the last two stages (Projection and Export, where the work is done), and under
 * "Video Lessons" where a prop begins (Images' chooser).
 */
export const discordCorner = () => h('a', {
  class: 'btn discord', href: DISCORD_URL, target: '_blank', rel: 'noopener', draggable: 'false',
  title: 'Join the Spellcast3D community on Discord (opens in a new tab)',
}, icon('chat'), 'Join Discord');

const LESSONS_URL = 'https://spellcast3d.com/lessons';

/** "Video Lessons", in the bottom-left corner where a prop begins (Images' chooser): the page of videos, in the browser. */
export const lessonsCorner = () => h('a', {
  class: 'btn lessons', href: LESSONS_URL, target: '_blank', rel: 'noopener', draggable: 'false',
  title: 'Short videos that show how to use Spellcast3D (opens in a new tab)',
}, icon('video'), 'Video Lessons');

/** "Show the file": File Explorer on Spellcast3D's folder with spellcast3d-config.ini selected. */
export const showConfigFile = () => attempt(() => api('POST', '/api/open-folder', { which: 'config' }));

/**
 * The note over a stage's main button while its tool installs (with how much is downloaded), loads, or
 * can't run here; sync(tool?) (Retopo: the prop's method). after: {<gate kind>: what the artist can do
 * meanwhile}, e.g. {installing: 'Import image works already.'}. In Images it also says, quietly, that
 * spellcast3d-config.ini was changed (it applies at the next start) or has a small mistake that did not
 * stop the image generator (an unknown model size: the default is used); a mistake in that file has a
 * link that shows it in File Explorer.
 */
export function gateLine(stage, after = {}) {
  const el = h('div', { class: 'sv-gate-line', hidden: true });
  el.sync = (tool = STAGE_TOOL[stage]) => {
    const gate = toolGate(stage, tool);
    const loading = loadingText(stage, tool);
    const blocked = !['ok', 'unknown'].includes(gate.kind);
    const problems = stage === 'images' ? configProblems() : [];
    const note = !blocked && !loading && stage === 'images' ? [configChanged(), ...problems].filter(Boolean).join(' ') : '';
    el.hidden = !blocked && !loading && !note;
    el.className = `sv-gate-line ${blocked ? gate.kind : note ? 'config' : 'loading'}`;
    const fileLink = (blocked ? gate.kind === 'config' : !!note && problems.length > 0)
      ? h('a', { href: '#', onclick: (e) => { e.preventDefault(); showConfigFile(); } }, 'Show the file') : null;
    const small = [blocked && gate.kind === 'config' ? gate.detail : '', blocked ? after[gate.kind] || '' : '']
      .filter(Boolean).join(' ');
    el.replaceChildren(icon((blocked && gate.kind !== 'installing') || note ? 'alert' : 'info'),
      h('span', null,
        blocked ? `${gate.text}${gate.kind === 'installing' ? gate.bytes || '' : ''}` : loading || note,
        small ? h('small', null, ` ${small}`) : null,
        fileLink ? ' ' : null, fileLink));
  };
  return el;
}

/**
 * The layer over a preview while this prop has a job of `stage` queued or running: what happens, the
 * progress and Stop. sync(box) shows / hides it. A prop at work shows its own words and how far it is; one that
 * waits (for its next step, or its turn) names the prop at work and follows its words and bar (onCard), so it is
 * never unclear whose progress shows. It says when the job is being stopped, and what the tool loads meanwhile
 * ("Loading the shape model…"). While the image generator draws with its shortcut (faster.js), a line over the
 * words says so, in every picture's progress: the artist chose it for all of them, once, somewhere else.
 */
export function busyLayer(stage) {
  const shortcut = h('div', { class: 'sv-busy-faster', hidden: true }, `${FASTER.images.title} is on`);
  const text = h('div', { class: 'sv-busy-text' });
  const bar = progressBar(null);
  let job = null;
  const stop = h('button', {
    class: 'btn ghost small', title: 'Stop making this',
    onclick: () => job && attempt(() => cancelJob(job)),
  }, icon('stop'), 'Stop');
  const el = h('div', { class: 'sv-busy', hidden: true }, spinner(), shortcut, text, bar, stop);
  el.sync = (box) => {
    const j = box ? jobFor(stage, box.id) : null;
    el.hidden = !j;
    job = j?.job || null;
    if (!j) return;
    const s = jobState(stage, box);
    const tool = j.job.tool || STAGE_TOOL[stage];
    shortcut.hidden = tool !== FASTER.images.tool || !fasterOn('images');
    const begun = j.phase === 'running' || j.phase === 'resting';
    // What the hub says of the job itself comes before the prop's own words: its tool starts, it is being
    // stopped (said by every prop it has begun), its tool reads a model while this prop waits.
    const stopping = begun ? jobStopping(j.job) : '';
    const loading = j.phase === 'running' ? '' : jobLoading(j.job);
    const words = j.phase === 'starting' ? (/install/i.test(j.message || '') ? '' : loadingText(stage, tool))
      : stopping || loading;
    const follows = (j.phase === 'resting' || j.phase === 'waiting') && !jobStopping(j.job);   // a stopped job: its own
    const other = follows ? onCard(j.job) : null;
    text.textContent = words || (other && j.phase === 'waiting' ? restingMessage(j.job, 'Waiting for its turn') : s.message);
    bar.set(other ? other.fraction : begun ? j.fraction : null);
    // Qwen-Image 2.1 finishes all the variations of a prop together at the end.
    stop.title = stage === 'images' && tool === 'qwen' ? 'Stop making these pictures. Pictures not finished yet are not kept.'
      : 'Stop making this';
  };
  return el;
}

/** The last error of this prop in `stage` (the job said why); sync(box). */
export function errorLine(stage) {
  const text = h('span');
  const el = h('div', { class: 'sv-error', role: 'alert', hidden: true }, icon('alert'), text);
  el.sync = (box) => {
    const err = box && !jobFor(stage, box.id) ? store.jobErrors?.[`${stage}:${box.id}`] : '';
    el.hidden = !err;
    text.textContent = err || '';
  };
  return el;
}

/** A picture's tooltip says this where Alt + click on it shows what Qwen-Image was given for it. */
export const INPUTS_KEY = 'Alt + click: what Qwen-Image was given for it.';

/**
 * A picture in the window of what Qwen-Image was given (omnitexture.js showInputs, given.js): itself small, a click
 * away from its full size.
 */
export const figure = (src, caption) => h('figure', { class: 'ot-input' },
  h('a', { href: src, target: '_blank', rel: 'noopener', title: 'Opens the picture at its full size' },
    h('img', { src, alt: caption, loading: 'lazy', draggable: 'false' })),
  h('figcaption', null, caption));

/** A list of a result's numbers ([label, value] pairs, empty values left out) and its warnings. */
export function statsBlock(rows, warnings = []) {
  const shown = rows.filter(([, v]) => v != null && v !== '');
  const notes = warnings.filter((w) => typeof w === 'string' && w);
  if (!shown.length && !notes.length) return '';
  return h('div', { class: 'sv-stats-wrap' },
    shown.length ? h('dl', { class: 'sv-stats' }, shown.map(([k, v]) => [h('dt', null, k), h('dd', null, v)])) : null,
    notes.map((w) => h('div', { class: 'sv-warning' }, icon('alert'), h('span', null, w))));
}

/**
 * The shown prop's name over a later stage's side panel, with its picture (or snapshot) as a small
 * image and a line under it; set(box, note, picture). `picture` is a URL: when it is the prop's own picture
 * (not a 3D snapshot), it shows as in the props list (cropped.propPicture): of a sheet sliced into views,
 * its Front slice. The picture before stays until the new one has loaded.
 */
export function boxHeading() {
  const pic = h('div', { class: 'sv-heading-pic', style: 'position: relative' });
  const name = h('h2', { class: 'sv-title' });
  const note = h('div', { class: 'sv-subtitle' });
  const el = h('div', { class: 'sv-heading' }, pic, h('div', { class: 'sv-heading-text' }, name, note));
  const shot = croppedPicture({ onError: () => pic.replaceChildren(icon('image')) });
  Object.assign(shot.el.style, { position: 'absolute', inset: '0' });
  let shownPic = null;
  el.set = (box, text = '', picture = null) => {
    name.textContent = box?.name || '';
    name.title = box?.name || '';
    note.textContent = text;
    note.hidden = !text;
    const own = box && picture ? propPicture(box) : null;
    const crop = own?.url === picture ? own.crop : null;
    const key = picture ? `${picture}|${crop || ''}` : null;
    if (key !== shownPic) {
      shownPic = key;
      shot.update(picture, crop);
      if (!picture) pic.replaceChildren(icon('image'));
      else if (shot.el.parentNode !== pic) pic.replaceChildren(shot.el);
    }
  };
  return el;
}
