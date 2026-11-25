// The "Edge fade" slider under an Omni-Texture's painted views: how far the edge of the repaint picked fades into
// what is around it. Moved, the hub puts that repaint onto the model again - its worker alone, no Qwen-Image - and
// answers with the prop and the models' new names (adapters/omnitexture.py refade). The hub's worker is readied
// when a repaint is pointed at or picked, so that the first change is as quick as the ones after it.

import { api, propApi } from '../api.js';
import { spinner } from '../controls.js';
import { h } from '../dom.js';
import { projectId } from '../store.js';
import { toastError } from '../ui.js';
import { FADE_STEP, MAX_FADE } from './rules.js';

/**
 * The slider of the prop `propId`: {row (its node), show(made, {any, locked}), ready(made), fading()}.
 * made: the repaint picked (rules.omniRepaints) or null; any: the model holds repaints at all; locked: a
 * Omni-Texture job of the prop waits or runs. Hooks: before() just before the hub is asked (it is asked once a
 * promise before() returns has settled: the hand paint goes into the files first), faded(project, answer)
 * with its answer {prop, renamed}, changed() whenever the slider's state changed (the caller shows it again).
 */
export function createFadeSlider(propId, { before, faded, changed }) {
  const range = h('input', {
    type: 'range', class: 'range ot-fade-range', min: 1, max: MAX_FADE, step: FADE_STEP, value: 1, disabled: true,
    'aria-label': 'Edge fade of the repaint picked',
  });
  const value = h('span', { class: 'ot-fade-value' });
  const busy = h('span', { class: 'ot-fade-busy', hidden: true }, spinner());
  const row = h('div', { class: 'ot-fade', hidden: true },
    h('span', { class: 'ot-fade-label' }, 'Edge fade'), range, value, busy);
  let picked = null;        // the repaint the slider is for
  let fading = false;
  let holding = false;      // the slider is held: the page's updates (any job's progress) leave its value alone
  let shown = null;         // the repaint and fade the slider was last set to
  let readied = null;       // the repaint the hub's worker was last readied for

  const showValue = () => {
    value.textContent = `\u00d7${String(+range.value)}`;
    range.style.setProperty('--fill', `${((range.value - 1) / (MAX_FADE - 1)) * 100}%`);
  };

  function show(made, { any, locked }) {
    picked = made;
    row.hidden = !any;
    range.disabled = !made || !made.fadable || fading || locked;
    const now = made ? `${made.glb}|${made.fade}` : null;
    if (!fading && !holding && now !== shown) {
      shown = now;
      range.value = made ? made.fade : 1;
    }
    showValue();
    busy.hidden = !fading;
    row.title = !made ? 'Click a repaint above (framed in magenta) to set how far its edge fades.'
      : !made.fadable ? 'This repaint cannot be faded again: the model it was painted onto is no longer kept.'
        : 'How far the edge of the repaint picked fades into what is around it. It is put onto the model again, '
          + 'which takes a few seconds.';
  }

  async function apply(made, fade) {
    const project = projectId();
    fading = true;
    changed();
    try {
      await before();
      faded(project, await api('POST', `${propApi(project, propId)}/omnitexture/fade`, { glb: made.glb, fade }));
    } catch (e) {
      shown = null;                 // (back to the fade the repaint has)
      toastError(e);
    } finally {
      fading = false;
      changed();
    }
  }

  /** The hub readies its worker for a repaint (it holds the model before it until a while after the last change). */
  function ready(made) {
    if (!made.fadable || readied === made.glb || fading) return;
    readied = made.glb;
    api('POST', `${propApi(projectId(), propId)}/omnitexture/fade`, { glb: made.glb, prepare: true })
      .catch(() => { readied = null; });
  }

  range.addEventListener('input', showValue);
  range.addEventListener('pointerdown', () => {
    holding = true;
    const release = () => {
      holding = false;
      document.removeEventListener('pointerup', release, true);
      document.removeEventListener('pointercancel', release, true);
    };
    document.addEventListener('pointerup', release, true);
    document.addEventListener('pointercancel', release, true);
  });
  range.addEventListener('change', () => {
    range.blur();
    if (picked && +range.value !== picked.fade) apply(picked, +range.value);
  });

  return { row, show, ready, fading: () => fading };
}
