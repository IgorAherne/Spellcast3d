// Spellcast3D web UI entry point: the header and the shell are drawn at once, then the project opened
// last (or a fresh untitled one) is opened, so the artist can start working without any question.

import { api } from './js/api.js';
import { h } from './js/dom.js';
import { icon } from './js/icons.js';
import { RUNNING, emit, imageTool, on, store } from './js/store.js';
import { loadModels, openLastProject, pollStatusSoon } from './js/actions.js';
import { mountHeader } from './js/shell/header.js';
import { mountCalibrationNotice } from './js/shell/calibration.js';
import { mountShell } from './js/shell/shell.js';

// A file dropped outside a drop zone would make the browser navigate away from the app.
for (const evt of ['dragover', 'drop']) window.addEventListener(evt, (e) => e.preventDefault());

// Forge's model list follows Forge's state, while Forge Neo is the image generator (spellcast3d-config.ini).
on('status', (before) => {
  if (imageTool() !== 'forge') return;
  const was = before?.tools?.forge?.state;
  const now = store.status?.tools?.forge?.state;
  if (was === now) return;
  if (now === 'ready' || now === 'busy') {
    if (store.models.state !== 'ready') loadModels();
  } else if (!RUNNING.has(now)) {
    store.models = { state: 'offline', data: null, message: '' };
    emit('models');
  }
});

async function boot() {
  const app = document.getElementById('app');
  mountHeader(document.getElementById('topbar'));
  mountCalibrationNotice(document.getElementById('notices'));
  const shell = mountShell(app);
  pollStatusSoon(0);
  api('GET', '/api/settings').then((s) => { store.settings = s; }).catch(() => {});
  try {
    await shell;
    await openLastProject();
  } catch (e) {
    console.error(e);
    showBootError(app, e);
  }
}

function showBootError(app, err) {
  app.replaceChildren(h('div', { class: 'boot-error' },
    h('div', { class: 'centre-note' },
      icon('alert', 'big'),
      h('h3', null, 'Spellcast3D could not open your work'),
      h('p', null, err.message),
      h('button', { class: 'btn primary big', onclick: () => location.reload() }, icon('refresh'), 'Try again'))));
}

boot();
