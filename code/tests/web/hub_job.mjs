// Jobs as the hub sends them, for the fixtures of these tests. Every job of the hub says where each of its
// props stands (`props`: prop id -> {state, fraction, message}) and which step its props go through (`part`);
// the page reads a prop's state from nowhere else (store.jobFor). A fixture spells out what a job always had
// (status, prop_ids, progress, errors) and goes through hubJob(), which adds those keys the way the hub derives
// them for a job that makes its props one after another (jobs.py Job.to_dict; code/tests/test_web.py compares
// the two). Every job also says what its tool is reading from disk (`loading`: the label after "Loading the",
// '' for nothing) and whether a cancel has been asked for and it has not ended yet (`stopping`). The 3D generator's jobs, whose props take turns step by step, are spelled out with batchJob().

const OVER = new Set(['done', 'failed', 'cancelled']);

/**
 * `job` with `props` and `part` as the hub gives them to a job whose props run one after another, in the order
 * of prop_ids:
 *   queued, starting_tool    every prop 'waiting'
 *   running                  the props that ended 'done' ('failed' when job.errors has them), the next one
 *                            'working' with its own share of progress.fraction and the job's message, the rest
 *                            'waiting'
 *   done, error, cancelled   'done' / 'failed' for the props that ended; one the job never came to, or that a
 *                            cancel cut short: 'cancelled' ('failed' when job.errors has it: its tool could not
 *                            start, and every prop says why)
 * The props that ended are the first progress.done of them, less cutShort: the props a cancel stopped half-way
 * (the hub counts them in progress.done; they are 'cancelled', also while the job still winds down).
 */
export function hubJob(job, { cutShort = 0 } = {}) {
  const ids = job.prop_ids || [];
  const progress = job.progress || {};
  const done = progress.done || 0;
  const failed = (id) => Object.hasOwn(job.errors || {}, id);
  const ended = done - cutShort;
  const own = Math.min(1, Math.max(0, (progress.fraction || 0) * ids.length - done));
  const state = (id, i) => {
    if (job.status === 'queued' || job.status === 'starting_tool') return 'waiting';
    if (i < ended) return failed(id) ? 'failed' : 'done';
    if (job.status === 'running' && i >= done) return i === done ? 'working' : 'waiting';
    return failed(id) ? 'failed' : 'cancelled';
  };
  const props = Object.fromEntries(ids.map((id, i) => {
    const s = state(id, i);
    const working = s === 'working';
    // How far: a prop that ended is through, the one in work has its share of the job's progress.
    const fraction = working ? own : s !== 'waiting' && i < ended ? 1 : 0;
    return [id, { state: s, fraction, message: working ? progress.message || '' : '' }];
  }));
  return { loading: '', stopping: false, ...job, props, part: null };
}

// The steps the 3D generator takes every prop of a job through, as its launcher reports them: key -> [label, the
// prop's own fraction when the step is finished].
export const PARTS = {
  cutout: ['Removing the background', 0.03], camera: ['Estimating the camera angle', 0.05],
  features: ['Reading the picture', 0.06], ss: ['Blocking out the shape', 0.15], cond_lr: ['Preparing the shape', 0.16],
  lr: ['Sculpting the shape', 0.29], upsample: ['Sculpting the shape', 0.30], cond_hr: ['Preparing the details', 0.32],
  hr: ['Refining shape details', 0.55], tex: ['Painting the texture', 0.72], finish: ['Building the 3D mesh', 1],
};

/**
 * A running mesh job of the 3D generator, which takes all its props through one step before the next.
 * states: prop id -> 'waiting' | 'resting' | 'done' | 'failed' | 'cancelled', or [state, fraction, message];
 * the prop on the tool now: 'working', or ['working', fraction, message]. part: the step's key (PARTS), with
 * `position` of `of` props in it; null between the job's start and its first step. extra: any other key of
 * the job (status, errors, progress, loading, stopping...).
 */
export function batchJob(id, states, { part = null, position = 1, of = null, ...extra } = {}) {
  const ids = Object.keys(states);
  const props = Object.fromEntries(ids.map((prop) => {
    const [state, fraction, message] = [states[prop]].flat();
    const far = fraction ?? (OVER.has(state) ? 1 : state === 'resting' && part ? PARTS[part][1] : 0);
    const words = message ?? (state === 'resting' ? 'Waiting for the next step' : state === 'working' && part ? `${PARTS[part][0]} (3/12)` : '');
    return [prop, { state, fraction: far, message: words }];
  }));
  const working = ids.find((prop) => props[prop].state === 'working') ?? null;
  const ended = ids.filter((prop) => props[prop].state === 'done' || props[prop].state === 'failed');
  const mean = ids.reduce((sum, prop) => sum + props[prop].fraction, 0) / (ids.length || 1);
  return {
    id, stage: 'mesh', tool: 'pixal3d', project: 'Props', status: 'running', prop_ids: ids, current_prop: working,
    progress: { done: ended.length, total: ids.length, fraction: mean, message: working ? props[working].message : '' },
    errors: {}, warnings: {}, props, loading: '', stopping: false,
    part: part ? { key: part, label: PARTS[part][0], position, of: of ?? ids.length } : null,
    ...extra,
  };
}
