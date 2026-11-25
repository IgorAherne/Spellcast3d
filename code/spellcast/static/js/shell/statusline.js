// The status line at the bottom of the props column: what Spellcast3D is doing right now, in plain
// words, with the time it has been at it - only while something loads, installs or works, or went wrong,
// and only what the workbench does not show already (the prop it shows being made, its stage's tool
// loading). The tools start and stop by themselves; this line is where the artist sees it happen.

import { formatBytes, formatElapsed, h } from '../dom.js';
import { icon } from '../icons.js';
import {
  ACTIVE_JOB, FIRST_START, PROP_ENDED, STAGE_TOOL, TOOL_NAMES, TOOL_ROLES, capital, getProp, jobLoading, jobStopping,
  stageNoun, stageTool, store, toolName, toolNoun, toolStatus,
} from '../store.js';
import { progressBar, spinner } from '../controls.js';
import { nameFromPrompt } from '../stages/rules.js';
import { withoutPercent } from '../stages/common.js';
import { showLog } from './logtail.js';

const VERB = {
  images: 'Making pictures', mesh: 'Making 3D models', omnitexture: 'Painting Omni-Textures', retopo: 'Retopologizing',
  projection: 'Baking textures',
};
const TOOLS = ['qwen', 'forge', 'pixal3d', 'imesh', 'lato2', 'triflow'];

const since = new Map();   // what is going on -> when it began, as far as this page knows
/**
 * Milliseconds since `key` was first seen (keys not asked for in a render are forgotten); `already`: the seconds it
 * had been going on by then (a tool's start the hub began earlier: its status' starting_seconds).
 */
function elapsed(key, seen, already = 0) {
  seen.add(key);
  if (!since.has(key)) since.set(key, Date.now() - (already || 0) * 1000);
  return Date.now() - since.get(key);
}

// The jobs that take every prop through one step before the next (job.part). The hub has no step for a
// moment while such a job starts its tool again, and a job that has shown one stays one: one clock, and
// one rule about what the workbench says, for the whole job.
const batches = new Set();

/** A job whose tool is starting: what it waits for, or "Loading LATO.2…" (Retopo runs one of several). */
function startingLine(job, seen) {
  const tool = job.tool || STAGE_TOOL[job.stage];
  const noun = stageNoun(job.stage, tool);
  const message = job.progress?.message || '';
  // The hub says what it waits for ("Waiting for the 3D generator to finish installing (45%)").
  if (toolStatus(tool)?.state === 'installing' || /install/i.test(message)) {
    return { kind: 'install', text: message || `Waiting for ${noun} to finish installing`, stage: job.stage };
  }
  const clock = `load:${tool}:${job.id}`;
  elapsed(clock, seen, toolStatus(tool)?.starting_seconds);
  return {
    kind: 'loading', clock, stage: job.stage,
    text: `Loading ${noun}…${FIRST_START[tool] ? ` (${FIRST_START[tool]})` : ''}`,
  };
}

/**
 * A running job says what it does, how far it is and for which prop: "Retopologizing with LATO.2 · 2 of 5 ·
 * Lantern"; the 3D generator, which takes every prop of a job through one step before the next, also says the
 * step, and counts the props in it: "Making 3D models · Sculpting the shape · 2 of 5 · Lantern" (job.part),
 * with one clock for the whole job.
 */
function runningLine(job, seen) {
  const tool = job.tool || STAGE_TOOL[job.stage];
  const p = job.progress || {};
  const batch = batches.has(job.id);   // its props take turns step by step
  const part = job.part || null;       // the step every prop of the job goes through now
  const states = Object.values(job.props || {}).map((prop) => prop?.state);
  // The props counted: those of the step, or of the job, one after another.
  const count = part ? part.of : states.length;
  const position = part ? part.position : Math.min(count, states.filter((s) => PROP_ENDED.has(s)).length + 1);
  // The job's clock when its props take turns step by step, else one for each prop.
  const clock = batch ? `run:${job.id}` : `run:${job.id}:${job.current_prop}`;
  elapsed(clock, seen);
  const line = (text) => ({ kind: 'busy', text, fraction: p.fraction ?? null, clock, stage: job.stage });
  // Retopo says which tool: "Retopologizing with LATO.2".
  const doing = `${VERB[job.stage] || 'Working'}${job.stage === 'retopo' && TOOL_NAMES[tool] ? ` with ${toolName(tool)}` : ''}`;
  const stopping = jobStopping(job);
  // Stop was pressed: short, so the words the artist is waiting for are not the ones the narrow column cuts off.
  if (batch && stopping) return line(`${doing} · ${stopping}`);
  const parts = [doing];
  if (part?.label) parts.push(part.label);
  if (count > 1 && position >= 1) parts.push(`${position} of ${count}`);
  const who = propName(getProp(job.current_prop));   // between two props the job has none
  if (who) parts.push(who);
  // What the tool reads from disk meanwhile, else the tool's own words, without the percentage some end
  // with ("Baking the textures… 42%": the bar shows how far it is). In a step, those words are the prop's
  // and the workbench's; between two steps they are the job's ("Starting Pixal3D again…").
  const words = jobLoading(job) || (batch && part ? '' : withoutPercent(p.message || ''));
  const verb = (s) => s.split(/\s/)[0].toLowerCase();
  if (words && verb(words) !== verb(doing) && !doing.toLowerCase().includes(words.toLowerCase())) parts.push(words);
  return line(parts.join(' · '));
}

/**
 * The line to show, or null: {kind: 'offline'|'busy'|'loading'|'install'|'error', text, fraction?,
 * clock?, stage?, tool?}. `clock` is a key whose elapsed time is shown. centre: {stage, prop}, the stage
 * on screen and the prop its workbench shows: the work on that prop and the loading of the tool that works
 * on it there (Retopo: its method) are left out, since the workbench shows them.
 */
export function headline(seen = new Set(), centre = {}) {
  // A running job's clock runs from when the page first saw it run, also while the workbench shows it
  // instead, or while the hub cannot be reached for a moment.
  const live = new Set();
  for (const j of store.jobs || []) {
    live.add(j.id);
    if (j.part) batches.add(j.id);
    if (j.status === 'running') elapsed(`run:${j.id}`, seen);
  }
  for (const id of batches) if (!live.has(id)) batches.delete(id);
  if (!store.connected) {
    return { kind: 'offline', text: 'Can’t reach Spellcast3D. Is its black console window still open? Trying again…' };
  }
  // The workbench shows a job by the prop on it: a job whose props take turns step by step says all of it
  // there, for any of its props; another job before it runs, for any of its props (its tool loads), and
  // while it runs from when it begins that prop until it is through with it - and for a prop it has not
  // begun while its tool reads a model, which the workbench says then too.
  const shown = (j) => {
    const prop = centre.prop && j.stage === centre.stage ? j.props?.[centre.prop] : null;
    if (!prop) return false;
    if (batches.has(j.id)) return true;
    if (PROP_ENDED.has(prop.state)) return false;
    return j.status !== 'running' || prop.state !== 'waiting' || !!jobLoading(j);
  };
  const jobs = (store.jobs || []).filter((j) => ACTIVE_JOB.has(j.status) && !shown(j));
  const job = jobs.find((j) => j.status === 'running') || jobs.find((j) => j.status === 'starting_tool');
  if (job) return job.status === 'starting_tool' ? startingLine(job, seen) : runningLine(job, seen);
  // The tool of the prop on screen (Retopo: its method) loads where the workbench says so already.
  const centreTool = stageTool(centre.stage, getProp(centre.prop) || null);
  for (const tool of TOOLS) {
    const st = toolStatus(tool);
    if (st?.state === 'starting' && tool !== centreTool) {
      const clock = `load:${tool}`;
      elapsed(clock, seen, st.starting_seconds);
      return { kind: 'loading', clock, text: `Loading ${toolNoun(tool)}…${FIRST_START[tool] ? ` (${FIRST_START[tool]})` : ''}` };
    }
  }
  for (const tool of TOOLS) {
    const st = toolStatus(tool);
    if (st?.state === 'error') {
      return { kind: 'error', tool, text: `${capital(toolNoun(tool))} stopped with an error${st.message ? `: ${st.message}` : '.'}` };
    }
  }
  const install = store.status?.install;
  const comps = install?.components || {};
  const failed = Object.entries(comps).find(([, c]) => c.status === 'failed');
  if (failed) {   // the install card (header) has "Try again"
    const [id, c] = failed;
    return { kind: 'error', tool: 'install', text: `${capital(toolNoun(id))} could not be installed${c.message ? `: ${c.message}` : '.'}` };
  }
  if (install?.running) {
    const id = install.current;
    const c = comps[id] || {};
    const pct = typeof c.fraction === 'number' ? ` ${Math.round(c.fraction * 100)}%` : '';
    const what = `Installing ${TOOL_ROLES[id] ? toolNoun(id) : 'the tools'}`;
    // Short: the column is narrow. With the download's size the bar says how far it is.
    return {
      kind: 'install', fraction: c.fraction ?? null,
      text: c.bytes_total ? `${what} · ${formatBytes(c.bytes_done || 0)} of ${formatBytes(c.bytes_total)}` : `${what}…${pct}`,
    };
  }
  // A job waits in the queue while the installer is busy with its tool (or in another window): the hub says why.
  const waiting = jobs.find((j) => j.status === 'queued' && /install/i.test(j.progress?.message || ''));
  if (waiting) return { kind: 'install', text: waiting.progress.message, stage: waiting.stage };
  return null;
}

/**
 * A prop's name in a sentence: its own, else - a new prop whose first pictures are being made has none yet -
 * the one its prompt will give it (as the name field suggests it), else "the new prop".
 */
export function propName(prop) {
  if (!prop) return '';
  return prop.name?.trim() || nameFromPrompt(prop.images?.settings?.prompt) || 'the new prop';
}

/** handlers: {go(stage)}. Returns {el, render(currentStage, shownPropId)}. */
export function createStatusLine(handlers) {
  const text = h('span', { class: 'sl-text', role: 'status', 'aria-live': 'polite' });   // not the ticking clock
  const clock = h('span', { class: 'sl-clock' });
  const bar = progressBar(null);
  const details = h('button', { type: 'button', class: 'sl-link', hidden: true }, 'Details');
  const lead = h('span', { class: 'sl-lead' });
  const el = h('div', { class: 'status-line', hidden: true },
    lead, h('span', { class: 'sl-body' }, text, bar), clock, details);
  let line = null;
  let timer = null;

  details.addEventListener('click', (e) => {
    e.stopPropagation();
    if (line?.tool) showLog(line.tool);
  });
  el.addEventListener('click', () => { if (line?.stage && line.stage !== current) handlers.go(line.stage); });
  let current = null;

  function tick() {
    clock.textContent = line?.clock && since.has(line.clock) ? formatElapsed(Date.now() - since.get(line.clock)) : '';
    clock.hidden = !clock.textContent;
  }

  function render(stage, prop = null) {
    current = stage;
    const seen = new Set();
    line = headline(seen, { stage, prop });
    for (const key of since.keys()) if (!seen.has(key)) since.delete(key);
    el.hidden = !line;
    clearInterval(timer);
    if (!line) return;
    const alarm = line.kind === 'error' || line.kind === 'offline';
    if (el.dataset.kind !== line.kind) lead.replaceChildren(alarm ? icon('alert') : spinner());   // a new spinner would restart
    el.dataset.kind = line.kind;
    text.textContent = line.text;
    el.title = line.text;
    bar.hidden = line.kind !== 'busy' && !(line.kind === 'install' && line.fraction != null);
    bar.set(line.fraction);
    details.hidden = !line.tool;
    el.classList.toggle('elsewhere', !!line.stage && line.stage !== stage);
    if (line.stage && line.stage !== stage) el.title = `${line.text}\nClick to go to that stage.`;
    tick();
    if (line.clock) timer = setInterval(tick, 1000);
  }

  return { el, render };
}
