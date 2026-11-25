// The calibration notice: a bar under the header while the background installer measures the graphics card for
// Pixal3D (install/components.py Pixal3D._calibrate; the step's text begins with CALIBRATING_STEP). It says what
// the measurement needs (heavy programs closed), what it does now and how far it is - the console says the same,
// which an artist who is not looking at it would miss - and that 3D jobs wait meanwhile. The × hides it for this
// calibration; it goes by itself when the step ends.

import { h } from '../dom.js';
import { icon } from '../icons.js';
import { on, store } from '../store.js';
import { progressBar, spinner } from '../controls.js';

// install/progress.py CALIBRATING_STEP: the hub's step text begins with it (tests/test_web.py keeps the two the same).
export const CALIBRATING_STEP = 'Calibrating Pixal3D for your graphics card';
const TITLE = 'Calibrating for performance: Spellcast3D is measuring your graphics card for the 3D step (once, a few minutes).';
const ASK = 'Please close heavy programs (Blender, games, videos, Photoshop, a browser with many tabs) until it is done. '
  + '3D jobs wait meanwhile.';

/** The calibration going on now, from the hub's status: {detail, fraction} or null. */
export function calibrationNow(status) {
  const entry = status?.install?.components?.pixal3d;
  if (!entry || entry.status !== 'running' || !String(entry.step || '').startsWith(CALIBRATING_STEP)) return null;
  const detail = String(entry.step).slice(CALIBRATING_STEP.length).replace(/^\s*·\s*/, '').trim();
  const fraction = typeof entry.step_fraction === 'number' ? entry.step_fraction : null;
  return { detail, fraction };
}

export function mountCalibrationNotice(host) {
  const detail = h('span', { class: 'cb-detail' });
  const bar = progressBar(null);
  const close = h('button', {
    type: 'button', class: 'icon-btn small', title: 'Hide', 'aria-label': 'Hide',
    onclick: () => { hidden = true; render(); },
  }, icon('x'));
  const el = h('div', { class: 'calib-banner', role: 'status', hidden: true },
    h('span', { class: 'cb-lead' }, spinner()),
    h('div', { class: 'cb-text' },
      h('b', null, TITLE),
      h('span', null, ASK),
      h('span', { class: 'cb-now' }, icon('clock'), detail),
      bar),
    close);
  host.append(el);
  let hidden = false;
  let seen = false;   // a calibration was on screen (the × hides that one; the next shows again)

  function render() {
    const now = calibrationNow(store.status);
    if (!now) {
      el.hidden = true;
      hidden = false;
      seen = false;
      return;
    }
    seen = true;
    el.hidden = hidden;
    detail.textContent = now.detail || 'starting…';
    bar.set(now.fraction);
  }
  on('status', render);
  render();
  return { el, render, get shown() { return seen && !hidden; } };
}
