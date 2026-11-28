// The install card: while Spellcast3D still installs its tools in the background (the hub runs the
// installer after the page is open), a small pill in the header says what is being installed and how far
// it got; it opens into one line per tool. It says so once when everything is ready, then goes away.

import { api } from '../api.js';
import { formatBytes, h } from '../dom.js';
import { icon } from '../icons.js';
import { configProblems, downloading, on, stepTime, store } from '../store.js';
import { pollStatusSoon } from '../actions.js';
import { progressBar, spinner } from '../controls.js';
import { attempt } from '../ui.js';
import { licenseChip } from '../licenses.js';
import { showLog } from './logtail.js';

export const NAMES = {
  hub: 'Spellcast3D',
  qwen: 'Image generator (Qwen-Image 2.1)',
  forge: 'Image generator (Forge Neo)',
  pixal3d: '3D generator',
  imesh: 'Retopology (Instant Meshes) and texture baking',
  lato2: 'AI retopology (LATO.2)',
  triflow: 'AI retopology (TriFlow)',
};
// The order the installer works in (install/components.py COMPONENTS, the hub first; the image generator
// is the one spellcast3d-config.ini picks).
export const ORDER = ['hub', 'imesh', 'qwen', 'forge', 'pixal3d', 'lato2', 'triflow'];
const WORDS = {
  pending: 'Waiting its turn', running: 'Installing…', done: 'Ready', failed: 'Could not be installed',
  unsupported: 'Can’t run on this computer', skipped: 'Skipped',
};
const STATE_ICON = { pending: 'clock', done: 'check', failed: 'alert', unsupported: 'x', skipped: 'x' };

export function mountInstallCard(host) {
  const title = h('span', { class: 'ic-title' });
  const lead = h('span', { class: 'ic-lead' });
  const toggle = h('button', { type: 'button', class: 'ic-head', 'aria-expanded': 'false' }, lead, title, icon('chevron', 'ic-chevron'));
  const items = h('div', { class: 'ic-items' });
  const retry = h('button', { type: 'button', class: 'btn primary small', onclick: () => tryAgain() }, icon('refresh'), 'Try again');
  const details = h('button', { type: 'button', class: 'btn ghost small', onclick: () => showLog('install') }, icon('terminal'), 'Details');
  const help = h('div', { class: 'ic-help' });
  // A mistake in spellcast3d-config.ini: the image generator it should name was not installed (or, for an
  // unknown model size, the default one was).
  const configText = h('span');
  const configRow = h('div', { class: 'ic-item ic-config', hidden: true }, h('span', { class: 'ic-icon' }, icon('alert')),
    h('div', { class: 'ic-text' }, h('span', { class: 'ic-name' }, h('b', null, 'Image generator')), configText));
  const list = h('div', { class: 'ic-list', hidden: true }, items, configRow, help);
  const close = h('button', { type: 'button', class: 'icon-btn small', title: 'Hide', 'aria-label': 'Hide', onclick: () => hide() }, icon('x'));
  const card = h('div', { class: 'install-card', hidden: true }, h('div', { class: 'ic-top' }, toggle, close), list);
  host.append(card);
  const rows = new Map();   // component id -> its line in the list
  let dismissed = '';       // the state the artist hid the card in (it comes back when that changes)
  let sawWork = false;      // this page saw an install running (so "all ready" is news)
  let doneTimer = null;

  const unfold = (open) => {
    list.hidden = !open;
    toggle.setAttribute('aria-expanded', String(open));
    card.classList.toggle('open', open);
  };
  toggle.addEventListener('click', () => unfold(list.hidden));
  document.addEventListener('mousedown', (e) => { if (!list.hidden && !card.contains(e.target)) unfold(false); });

  async function tryAgain() {
    retry.disabled = true;
    await attempt(() => api('POST', '/api/install/start'));
    retry.disabled = false;
    pollStatusSoon(300);
  }
  function hide() {
    dismissed = signature();
    card.hidden = true;
  }
  const install = () => store.status?.install || {};
  const components = () => Object.entries(install().components || {})
    .sort(([a], [b]) => ORDER.indexOf(a) - ORDER.indexOf(b));
  const signature = () => `${components().map(([id, c]) => `${id}:${c.status}`).join('|')}|${install().error || ''}`;

  /** One line per tool, updated in place (a rebuilt spinner or bar would jump at every status poll). */
  function renderItems(comps) {
    for (const [id, row] of rows) if (!comps.some(([cid]) => cid === id)) { row.el.remove(); rows.delete(id); }
    for (const [id, c] of comps) {
      let row = rows.get(id);
      if (!row) {
        row = { icon: h('span', { class: 'ic-icon' }), text: h('span'), bar: progressBar(null), bytes: h('small'), status: null };
        row.el = h('div', { class: 'ic-item' }, row.icon, h('div', { class: 'ic-text' },
          h('span', { class: 'ic-name' }, h('b', null, NAMES[id] || id), licenseChip(id)), row.text, row.bar, row.bytes));
        rows.set(id, row);
      }
      items.append(row.el);   // keeps the order
      if (row.status !== c.status) {
        row.status = c.status;
        row.el.className = `ic-item ${c.status}`;
        row.icon.replaceChildren(c.status === 'running' ? spinner() : icon(STATE_ICON[c.status] || 'clock'));
      }
      row.text.textContent = c.status === 'failed' && c.message ? c.message : c.step || c.message || WORDS[c.status] || c.status;
      row.bar.hidden = c.status !== 'running';
      row.bar.set(typeof c.fraction === 'number' ? c.fraction : null);
      row.bytes.textContent = c.status !== 'running' ? ''
        : downloading(c) ? `${formatBytes(c.bytes_done || 0)} of ${formatBytes(c.bytes_total)}` : stepTime(id, c);
      row.bytes.hidden = !row.bytes.textContent;
    }
  }

  function render() {
    const inst = install();
    const comps = components();
    const failed = comps.filter(([, c]) => c.status === 'failed');
    const working = !!inst.running || comps.some(([, c]) => c.status === 'running' || c.status === 'pending');
    const trouble = failed.length > 0 || (!working && !!inst.error);
    if (working) sawWork = true;
    const allDone = !working && !trouble && sawWork;
    if ((!working && !trouble && !allDone) || signature() === dismissed) {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    const problems = configProblems();
    const noImages = problems.length > 0 && store.status?.config?.image_backend == null;
    // Installed, but without an image generator (spellcast3d-config.ini has a mistake): no success check.
    const kind = trouble ? 'failed' : working ? 'working' : noImages ? 'warn' : 'done';
    if (card.dataset.kind !== kind) {
      card.dataset.kind = kind;
      lead.replaceChildren(kind === 'failed' || kind === 'warn' ? icon('alert') : kind === 'working' ? spinner() : icon('check'));
    }
    configRow.hidden = !problems.length;
    configRow.classList.toggle('failed', noImages);
    configText.textContent = problems.join(' ');
    const left = comps.filter(([, c]) => c.status === 'running' || c.status === 'pending').length;
    const current = comps.find(([id]) => id === inst.current)?.[1];
    toggle.title = '';
    if (trouble) {
      title.textContent = failed.length ? `${NAMES[failed[0][0]] || failed[0][0]}: could not be installed` : 'The installation stopped';
    } else if (working) {
      const pct = typeof current?.fraction === 'number' ? ` · ${Math.round(current.fraction * 100)}%` : '';
      title.textContent = left ? `Installing ${left === 1 ? '1 tool' : `${left} tools`}${pct}` : inst.message || 'Checking what still needs installing…';
      // On hover, what the tool at work does right now: the pill itself stays short.
      toggle.title = current ? `${NAMES[inst.current] || inst.current}: ${current.step || current.message || WORDS.running}` : '';
    } else {
      title.textContent = noImages ? 'Installed, without an image generator' : 'Everything is installed';
      doneTimer ||= setTimeout(() => { sawWork = false; doneTimer = null; render(); }, 6000);
    }
    renderItems(comps);
    help.replaceChildren(...(trouble
      ? [h('p', { class: 'hint' }, !failed.length && inst.error ? inst.error
        : 'It continues where it stopped: what is already installed stays.'), h('div', { class: 'ic-actions' }, retry, details)]
      : [h('p', { class: 'hint' }, 'You can work meanwhile: anything that needs a tool still being installed waits for it.')]));
  }

  on('status', render);
  render();
}
