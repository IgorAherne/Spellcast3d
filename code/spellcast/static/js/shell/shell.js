// The application shell: the props column on the left (column.js, with the props list of proplist.js)
// and the workbench (bench.js) with the current stage's view (js/stages/*.js). It keeps where the artist
// is (the stage, the selected prop), hands every view a fresh context, and runs the moves between
// stages: the whole page (column and workbench; the header stays) swipes to the next stage, like
// scrolling down to the next section of a page (up for the stage before).

import { api } from '../api.js';
import { h, isTyping } from '../dom.js';
import { STAGES, boxesOf, draftOf, on, stageTool, store } from '../store.js';
import { updateUi, warmTool } from '../actions.js';
import { warmAsk } from '../stages/rules.js';
import { spinner } from '../controls.js';
import { ShellRefusal, dialogOpen, toast, toastError } from '../ui.js';
import { syncThumbs } from '../stages/thumbs.js';
import { loadViews, nextStage } from './stages.js';
import { createPropList } from './proplist.js';
import { createColumn } from './column.js';
import { createBench } from './bench.js';
import { createStatusLine } from './statusline.js';
import { initMotion, swipe } from './anim.js';
import { initTooltips } from './tooltips.js';
import { hideTip, menuOpen } from './popups.js';

const state = {
  views: null,
  stage: 'images',
  selectedId: null,
  mounted: null,     // the stage whose view is in the workbench
  moving: false,     // the page is swiping to another stage
  settled: Promise.resolve(),   // resolves once nothing moves on screen
};
let page = null;       // <main class="shell">: the column and the workbench, what moves between stages
let list = null;
let column = null;
let bench = null;
let statusLine = null;
let queued = null;
let warmDue = false;   // the artist has arrived at a stage whose tool has not been asked to start early yet
let motions = 0;       // moves on screen now
let stilled = null;    // resolves state.settled

/**
 * Something starts moving on screen: state.settled waits for it (and for anything else moving) until
 * the returned function is called. Heavy work - loading a 3D model - waits for it, not to stall the move.
 */
function inMotion() {
  if (!motions++) state.settled = new Promise((resolve) => { stilled = resolve; });
  let over = false;
  return () => {
    if (over) return;
    over = true;
    if (!--motions) stilled();
  };
}

// ---- context handed to the stage views ------------------------------------------------------------

function makeCtx(stage = state.stage) {
  return {
    store,
    api,
    project: store.project,
    draft: draftOf(),
    boxes: boxesOf(),
    selectedId: state.selectedId,
    stageId: stage,
    status: store.status,
    jobs: store.jobs,
    // While the page swipes, views show stills; 3D models (and snapshots) load once `settled`
    // resolves: loading one takes the page's attention for a moment, which would stall the move.
    moving: state.moving,
    settled: state.settled,
    select,
    goto: (s) => goto(s),
    refresh,
    toast: (message, kind) => toast(message, kind),
    /** The call to action asks for attention: the artist is done in this stage (Retopo's last pick). */
    nudgeCta: () => column.nudgeCta(),
  };
}

// ---- rendering -------------------------------------------------------------------------------------

let pendingRender = null;

/** Re-renders everything shortly (several calls in a row render once). Resolves after the render. */
function refresh() {
  pendingRender ||= new Promise((resolve) => setTimeout(() => {
    pendingRender = null;
    renderNow();
    resolve();
  }, 0));
  return pendingRender;
}

const guard = (fn) => {
  try {
    return fn();
  } catch (e) {
    console.error(e);
    return undefined;
  }
};

/** The prop the workbench shows (the view says; a later stage shows one even when none is selected). */
const shownId = () => (state.mounted === state.stage ? guard(() => state.views[state.stage].shownId?.()) ?? null : null);

function renderNow() {
  if (!store.project || !state.views) return;
  if (state.selectedId && !boxesOf().some((b) => b.id === state.selectedId)) state.selectedId = null;
  const ctx = makeCtx();
  const view = state.views[state.stage];
  if (state.mounted === state.stage) guard(() => view.update(ctx));
  const shown = shownId();
  list.render(ctx, view, shown);
  column.render(view, ctx, shown);
  statusLine.render(state.stage, shown);
  // The tiles' small pictures (and the views' snapshots) are made in the background.
  syncThumbs(state.stage, state.settled);
}

function mountView(stage) {
  bench.view.replaceChildren();
  bench.view.dataset.stage = stage;
  state.mounted = stage;
  guard(() => state.views[stage].mount(bench.view, makeCtx(stage)));
}

function unmountView() {
  if (!state.mounted) return;
  const view = state.views[state.mounted];
  state.mounted = null;
  guard(() => view.unmount());
  bench.view.replaceChildren();
}

function enterProject() {
  queued = null;
  unmountView();
  state.selectedId = null;
  const saved = store.project.ui?.stage;
  state.stage = STAGES.includes(saved) ? saved : 'images';
  mountView(state.stage);
  renderNow();
  arrive();
}

// ---- the stage's tool, started early ----------------------------------------------------------------

/** The artist has arrived at a stage (or the page opened on one): its tool is asked to start early, once. */
function arrive() {
  warmDue = true;
  warmStage();
}

/**
 * Asks the hub to start the tool of the stage on screen early, so its start is over by the time the artist presses
 * the stage's button. The hub starts it only when that displaces nothing, unless the stage's wish says otherwise
 * (actions.js warmTool, rules.js warmAsk). Before the hub's first status it waits for that (it runs again on every
 * status), then asks at most once per arrival.
 */
function warmStage() {
  const tools = store.status?.tools;
  if (!warmDue || !tools || !store.project || state.mounted !== state.stage) return;
  warmDue = false;
  const view = state.views[state.stage];
  // A view may say its wish itself (view.warmWish: Omni-Texture's only while a chosen prop waits for it, Images'
  // with the model picked, Retopo's the method of the props it would retopologize first).
  const wish = view.warmWish ? guard(() => view.warmWish(makeCtx())) ?? null : { tool: stageTool(state.stage) };
  const ask = warmAsk(tools, wish);
  if (ask) warmTool(ask.tool, ask.wish);
}

// ---- selection --------------------------------------------------------------------------------------

/** Selects a prop (null: none; in Images that is the new prop being prepared): the workbench shows it. */
function select(id) {
  if (id === state.selectedId) return;
  state.selectedId = id;
  refresh().then(() => { if (id) list.reveal(id, true); });
}

/** A click on a prop's tile: the prop is selected (clicked again, the workbench is drawn anew). */
function pick(id) {
  if (id === state.selectedId) refresh();
  else select(id);
}

/**
 * A click on the box beside a prop's tile (view.tileCheck): the view on screen ticks it or not (view.checkTile), and
 * the page is drawn again so the box shows it at once (a box that lags gets a second click, which undoes the first).
 */
function check(id) {
  const view = state.views?.[state.stage];
  if (!view?.checkTile || state.mounted !== state.stage || state.moving) return;
  guard(() => view.checkTile(id, makeCtx()));
  refresh();
}

/**
 * The artist typed in the boxes on a prop's tile (view.tileSizes): the view on screen keeps the numbers
 * (view.sizeTile), and the page is drawn again with them.
 */
function sizes(id, values) {
  const view = state.views?.[state.stage];
  if (!view?.sizeTile || state.mounted !== state.stage) return;
  guard(() => view.sizeTile(id, values, makeCtx()));
  refresh();
}

// ---- moving between stages ----------------------------------------------------------------------------

/**
 * Goes to stage `target`: the page swipes there (down for a later stage, up for an earlier one): a still
 * copy of the page being left moves out while the page, drawn at once for the new stage (its heading, the
 * tiles' status there, its view), moves in. opts: {select: prop id to select there, follow: drop the
 * selection so the workbench follows the work (after the call to action)}.
 */
async function goto(target, opts = {}) {
  if (!state.views || !store.project || !STAGES.includes(target)) return;
  if (state.moving) {
    queued = [target, opts];
    return;
  }
  if (target === state.stage) {
    if (opts.select !== undefined) select(opts.select);
    return;
  }
  state.moving = true;
  const arrived = inMotion();
  const down = STAGES.indexOf(target) > STAGES.indexOf(state.stage);
  const pid = store.project.id;
  hideTip();
  let frozen = null;
  try {
    frozen = await bench.freeze().catch((e) => {
      console.warn('The page could not be copied for the move:', e);
      return null;
    });
    // Another project opened while the page was copied: it shows its own stage, leave it be.
    if (store.project?.id !== pid) return;
    unmountView();
    state.stage = target;
    if (opts.follow) state.selectedId = null;
    if (opts.select !== undefined) state.selectedId = opts.select;
    updateUi({ stage: target });
    mountView(target);
    renderNow();
    arrive();
    if (state.selectedId) list.reveal(state.selectedId);
    await swipe(frozen, page, down);
  } catch (e) {
    console.error(e);
  } finally {
    frozen?.remove();
    state.moving = false;
    arrived();
  }
  renderNow();
  if (queued) {
    const [t, o] = queued;
    queued = null;
    goto(t, o);
  }
}

// ---- the "+", the stage's own button and the call to action ---------------------------------------------

const working = { cta: false, plus: false, row: false };

async function onCta() {
  const view = state.views?.[state.stage];
  if (!view || state.moving || working.cta) return;
  const ctx = makeCtx();
  const gate = guard(() => view.ctaEnabled(ctx)) || { enabled: false };
  if (!gate.enabled) {
    // Nothing to make here yet (no picture, no model...): the button still leads down, as the only way to
    // the next stage (its own "+" may import what this stage has not made). The last stage, and one that holds
    // the props back (gate.blocked), say why not.
    const down = nextStage(state.stage);
    if (down && !gate.blocked) await goto(down, { follow: true });
    else column.refuse('cta', gate.hint);
    return;
  }
  working.cta = true;
  column.busy('cta', true);
  try {
    await view.onCta(ctx);
  } catch (e) {
    if (e instanceof ShellRefusal) column.refuse('cta', e.message);
    else toastError(e);
    return;
  } finally {
    working.cta = false;
    column.busy('cta', false);
  }
  const next = nextStage(state.stage);
  if (next) await goto(next, { follow: true });
  else refresh();
}

/** The stage's own button: works on the props right in this stage (view.rowAction), without moving on. */
async function onRowAction() {
  const view = state.views?.[state.stage];
  if (!view?.rowAction || state.moving || working.row) return;
  const action = guard(() => view.rowAction(makeCtx()));
  if (!action) return;
  working.row = true;
  column.busy('row', true);
  try {
    await action.run();
  } catch (e) {
    toastError(e);
  } finally {
    working.row = false;
    column.busy('row', false);
  }
  refresh();
}

async function onPlus() {
  const view = state.views?.[state.stage];
  if (!view?.plus || state.moving || working.plus) return;
  working.plus = true;
  column.busy('plus', true);
  try {
    await view.onPlus(makeCtx());
  } catch (e) {
    if (e instanceof ShellRefusal) column.refuse('plus', e.message);
    else toastError(e);
  } finally {
    working.plus = false;
    column.busy('plus', false);
  }
  refresh();
}

// ---- keys -----------------------------------------------------------------------------------------------

function onKey(e) {
  if (!store.project || e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
  if (isTyping() || dialogOpen() || menuOpen()) return;
  if (e.key === 'Escape' && state.selectedId) {
    select(null);
    return;
  }
  if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
  const props = boxesOf();
  if (!props.length) return;
  e.preventDefault();
  const i = props.findIndex((b) => b.id === state.selectedId);
  const next = i < 0 ? (e.key === 'ArrowDown' ? 0 : props.length - 1)
    : Math.min(props.length - 1, Math.max(0, i + (e.key === 'ArrowDown' ? 1 : -1)));
  select(props[next].id);
  list.tile(props[next].id)?.focus({ preventScroll: true });
}

// ---- mounting ----------------------------------------------------------------------------------------------

/** Builds the props column and the workbench into `host`; the views fill in once a project is open. */
export async function mountShell(host) {
  initMotion();
  initTooltips();
  statusLine = createStatusLine({ go: (s) => goto(s) });
  list = createPropList({ select, pick, check, sizes });
  column = createColumn({ list, statusLine }, {
    go: (s) => goto(s), cta: onCta, plus: onPlus, row: onRowAction,
  });
  bench = createBench();
  page = h('main', { class: 'shell' }, column.el, bench.el);
  host.replaceChildren(page);
  bench.view.append(loadingNote());

  // A click on the empty space around the props lets go of the selected prop.
  list.el.addEventListener('click', (e) => {
    if ([list.list, list.el, list.scroller, list.tail].includes(e.target)) select(null);
  });
  document.addEventListener('keydown', onKey);

  state.views = await loadViews();
  on('project', ({ opened } = {}) => (opened ? enterProject() : refresh()));
  on('prop', refresh);
  on('jobs', refresh);
  on('status', refresh);
  on('status', warmStage);
  on('models', refresh);
  // A stage changed what a prop is (Images: its picture picked, or a picture made a prop): its tile rings.
  on('pulse-prop', ({ id } = {}) => {
    list.reveal(id, true);
    list.pulse(id);
  });
  // Asked for from outside the stages (the settings' Omni-Texture test): go to a stage, a prop selected there.
  on('goto', ({ stage, select: id } = {}) => goto(stage, id === undefined ? {} : { select: id }));
  if (store.project) enterProject();
}

const loadingNote = () => h('div', { class: 'centre-note' }, spinner(), h('h3', null, 'Opening your workbench…'));
