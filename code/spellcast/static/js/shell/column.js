// The props column, always on the left: the way back up ("↑ 3D", an outline button as big as the call to
// action, goes back up a stage) over the stage's heading (what it does to every prop: "All Retopo",
// plain words on one line), the props list (proplist.js) with the "+" right under
// its last prop, scrolling with it (Images: a new prop; 3D: import the artist's own model), and pinned at the bottom the
// stage's own button when its view offers one ("Make 3D for 3 props"), the status line, and the call to
// action, the one big warm button that goes down to the next stage with every prop ("Generate 3D ↓").
// Those two buttons are the way between the stages. Hovering or focusing the call to action explains what
// it does in a small popover at its right; when it can't act, pressing it shakes it and the popover says why.

import { h } from '../dom.js';
import { icon } from '../icons.js';
import { spinner } from '../controls.js';
import { fadeIn, nudge, shake } from './anim.js';
import { STAGE_INFO, nextStage, prevStage } from './stages.js';
import { hideTip, showTip, tipShown } from './popups.js';

const PLUS = {
  new: { label: 'New prop', title: 'Start a new prop: import or generate its picture' },
  import: { label: 'Import 3D model', title: 'Use your own 3D model (FBX, OBJ, glTF, or a .zip with its textures) as a new prop' },
};
const HOVER_DELAY = 150;   // ms over the call to action before it explains itself
const REFUSAL_MS = 4000;   // how long a refusal is shown

/**
 * parts: {list (proplist.js), statusLine}; handlers: {go(stage), cta(), plus(), row()}. Returns
 * {el, render(view, ctx, shownId), refuse(which, message), busy(which, on), nudgeCta()}.
 */
export function createColumn(parts, handlers) {
  const upText = h('span', { class: 'col-up-text' });
  const up = h('button', {
    type: 'button', class: 'btn big col-up', hidden: true, onclick: () => upTo && handlers.go(upTo),
  }, icon('up'), upText);
  const heading = h('h2', { class: 'col-heading' });
  const head = h('header', { class: 'col-head' }, up, heading);

  // Right under the last prop, in the scrolling list: never next to the call to action.
  const plusText = h('span');
  const plusBtn = h('button', { type: 'button', class: 'plus-btn', onclick: () => handlers.plus() }, icon('plus'), plusText);
  parts.list.tail.append(plusBtn);
  const rowIcon = h('span', { class: 'row-action-icon' });
  const rowText = h('span', { class: 'row-action-text' });
  const rowBtn = h('button', { type: 'button', class: 'btn ghost row-action', hidden: true, onclick: () => handlers.row() }, rowIcon, rowText);
  const ctaText = h('span', { class: 'cta-text' });
  const ctaIcon = h('span', { class: 'cta-icon' });
  const about = h('span', { class: 'sr-only', id: 'cta-about' });
  const cta = h('button', {
    type: 'button', class: 'btn big cta', 'aria-describedby': 'cta-about', onclick: () => handlers.cta(),
  }, ctaText, ctaIcon);
  const ring = h('span', { class: 'cta-ring', 'aria-hidden': 'true' });
  const ctaSlot = h('div', { class: 'cta-slot' }, cta, ring);
  const alert = h('span', { class: 'sr-only', role: 'alert' });
  const foot = h('footer', { class: 'col-foot' }, rowBtn, parts.statusLine.el, ctaSlot, about, alert);

  const el = h('aside', { class: 'column', 'aria-label': 'Stage and props' }, head, parts.list.el, foot);

  let stage = null;       // the stage the column is drawn for
  let upTo = null;
  let aboutText = '';     // what the call to action does (view.ctaHover)
  let ctaGlyph = '';      // the icon after its words
  let refusing = null;    // the button showing a refusal now
  let refusalTimer = null;
  let hoverTimer = null;

  // ---- the heading ----

  function renderHead(ctx) {
    if (ctx.stageId === stage) return;
    const first = stage === null;
    stage = ctx.stageId;
    upTo = prevStage(stage);
    // The first stage has no way back up: its heading is the top of the column (no empty place is kept:
    // between stages the whole page moves, so nothing is seen to jump).
    up.hidden = !upTo;
    if (upTo) {
      upText.textContent = STAGE_INFO[upTo].title;
      up.title = `Back up to ${STAGE_INFO[upTo].title}`;
      up.setAttribute('aria-label', `Back up to ${STAGE_INFO[upTo].title}`);
    }
    heading.textContent = STAGE_INFO[stage]?.heading || stage;
    heading.title = STAGE_INFO[stage]?.tip || '';
    // During a move the whole page swipes, heading and all.
    if (!first && !ctx.moving) fadeIn(head, 220);
  }

  // ---- the call to action's popover ----

  /** Explains the call to action next to it (unless it is saying why it can't act). */
  function showAbout() {
    if (refusing === cta || ctaSlot.hidden) return;
    if (aboutText) showTip(cta, aboutText, 'right', 'note');
    else hideTip(cta);
  }
  const hideAbout = () => {
    clearTimeout(hoverTimer);
    if (refusing !== cta) hideTip(cta);
  };
  cta.addEventListener('mouseenter', () => {
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(showAbout, HOVER_DELAY);
  });
  cta.addEventListener('mouseleave', hideAbout);
  cta.addEventListener('focus', () => { if (cta.matches(':focus-visible')) showAbout(); });
  cta.addEventListener('blur', hideAbout);
  cta.addEventListener('pointerdown', () => clearTimeout(hoverTimer));

  /** "No": the button shakes, and the popover next to it says why for a few seconds. */
  function refuse(which, message) {
    const btn = which === 'plus' ? plusBtn : cta;
    shake(btn);
    if (!message) return;
    clearTimeout(hoverTimer);
    clearTimeout(refusalTimer);
    refusing = btn;
    showTip(btn, message, 'right', 'refusal');
    alert.textContent = '';
    alert.textContent = message;
    refusalTimer = setTimeout(() => {
      refusing = null;
      if (!tipShown(btn)) return;
      if (btn === cta && (cta.matches(':hover') || cta.matches(':focus-visible'))) showAbout();
      else hideTip(btn);
    }, REFUSAL_MS);
  }

  // ---- rendering ----

  /** The heading, the "+", the stage's own button and the call to action of `view` for this context. */
  function render(view, ctx, shownId = null) {
    renderHead(ctx);
    const plus = PLUS[view.plus] || null;
    plusBtn.hidden = !plus;
    if (plus) {
      plusText.textContent = plus.label;
      plusBtn.title = plus.title;
      // The workbench shows the new prop being prepared (Images' draft): the "+" is where the artist is.
      const current = plus === PLUS.new && !ctx.selectedId && !!ctx.draft && shownId === ctx.draft.id;
      plusBtn.classList.toggle('current', current);
      plusBtn.setAttribute('aria-current', String(current));
    }
    let gate = { enabled: false };
    let row = null;
    let words = '';
    try {
      gate = view.ctaEnabled?.(ctx) || gate;
      row = view.rowAction?.(ctx) || null;
      words = typeof view.ctaHover === 'function' ? view.ctaHover(ctx) : view.ctaHover;
    } catch (e) {
      console.error(e);
    }
    rowBtn.hidden = !row;
    if (row) {
      rowText.textContent = row.label;
      rowBtn.title = row.hint || row.label;
      if (rowIcon.dataset.icon !== (row.icon || '')) {
        rowIcon.dataset.icon = row.icon || '';
        rowIcon.replaceChildren(row.icon ? icon(row.icon) : '');
      }
    }
    ctaSlot.hidden = !view.ctaLabel;   // the last stage has none: only the way back up
    if (view.ctaLabel) {
      ctaText.textContent = view.ctaLabel;
      paintCtaIcon(nextStage(ctx.stageId) ? 'down' : view.ctaIcon || '');
      // With nothing to make yet the button still leads down to the next stage (shell.onCta); only a last
      // stage's button, or one whose view holds the props back (gate.blocked: Retopo, until every prop has its
      // clean mesh picked), can refuse: the press reaches the handler, which shakes it and says why.
      const stuck = !gate.enabled && (gate.blocked || !nextStage(ctx.stageId));
      cta.setAttribute('aria-disabled', String(stuck));
      cta.classList.toggle('disabled', stuck);
    } else if (refusing === cta || tipShown(cta)) {
      refusing = null;
      hideTip(cta);
    }
    const text = (words || '').trim();
    if (text !== aboutText) {
      aboutText = text;
      about.textContent = text;
      if (tipShown(cta) && refusing !== cta) showAbout();
    }
  }

  /** The arrow (or the view's icon) after the call to action's words; a spinner while it works. */
  function paintCtaIcon(glyph) {
    ctaGlyph = glyph;
    const shown = cta.classList.contains('working') ? 'working' : glyph;
    if (ctaIcon.dataset.icon === shown) return;
    ctaIcon.dataset.icon = shown;
    ctaIcon.replaceChildren(shown === 'working' ? spinner() : glyph ? icon(glyph) : '');
  }

  function busy(which, on) {
    const btn = { plus: plusBtn, cta, row: rowBtn }[which];
    btn.classList.toggle('working', on);
    btn.disabled = on;
    if (btn === cta) {
      if (on) hideAbout();
      paintCtaIcon(ctaGlyph);
    }
  }

  return {
    el,
    render,
    refuse,
    busy,
    /** The call to action asks for attention (the artist is done here). */
    nudgeCta: () => (ctaSlot.hidden ? Promise.resolve() : nudge(cta, ring)),
  };
}
