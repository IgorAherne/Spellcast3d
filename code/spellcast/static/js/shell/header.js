// The header: the logo, the project menu on the left (a folder and the open project's name; "Save as" to
// name an untitled one, rename, new project, recent projects, its folder), what Spellcast3D is for in one
// quiet line, and on the right the install progress while tools are still being installed and the settings
// gear. (Donate is in the bottom-left corner of every stage's viewport: stages/common.js donateCorner.)
// There is no Save: everything is saved as it happens; naming a project is all "Save as" does.

import { api } from '../api.js';
import { h, plural, timeAgo } from '../dom.js';
import { icon } from '../icons.js';
import { on, projectId, store } from '../store.js';
import { createProject, openProject, renameProject } from '../actions.js';
import { attempt } from '../ui.js';
import { popupMenu } from './popups.js';
import { openSettings } from './settings.js';
import { mountInstallCard } from './install.js';

const RECENT = 8;
const TAGLINE = 'Generate assets efficiently, reloading heavy neural nets once, only between stages.';

const isUntitled = () => !!(store.project?.untitled
  ?? store.projects.find((p) => p.id === projectId())?.untitled);

export function mountHeader(host) {
  const name = h('span', { class: 'pm-name' });
  const dot = h('span', { class: 'pm-dot', title: 'Not named yet. Your work is saved all the same; “Save as…” gives the project a name.' });
  const trigger = h('button', {
    type: 'button', class: 'pm-button', 'aria-haspopup': 'menu', 'aria-expanded': 'false',
    title: 'Project: name it, start a new one or open another',
    onclick: () => openMenu(),
  }, icon('folder', 'pm-folder'), h('span', { class: 'pm-label' }, name, dot), icon('chevron', 'pm-chevron'));
  const menuWrap = h('div', { class: 'project-menu' }, trigger);
  // A dot on it while Settings has a note (spellcast3d-config.ini changed: start again; an extra model folder
  // that is not there), with the note on hover.
  const gearDot = h('span', { class: 'gear-dot', hidden: true });
  const gear = h('button', { type: 'button', class: 'icon-btn header-gear', title: 'Settings', 'aria-label': 'Settings', onclick: () => openSettings() },
    icon('gear'), gearDot);
  const syncGear = () => {
    const notes = (store.status?.settings_warnings || []).filter((w) => typeof w === 'string' && w);
    gearDot.hidden = !notes.length;
    gear.title = notes.length ? ['Settings', ...notes].join('\n') : 'Settings';
  };
  const installSlot = h('div', { class: 'install-slot' });
  host.append(
    h('div', { class: 'brand' },
      h('img', { src: '/static/favicon.svg', alt: '', width: 26, height: 26 }),
      h('span', null, 'Spellcast', h('b', null, '3D'))),
    menuWrap,
    h('span', { class: 'tagline', title: TAGLINE }, TAGLINE),   // the title: when it is cut short
    h('div', { class: 'spacer' }),
    installSlot,
    gear);
  mountInstallCard(installSlot);

  function render() {
    const p = store.project;
    menuWrap.hidden = !p;
    if (!p) return;
    if (!menuWrap.querySelector('.pm-edit')) name.textContent = p.name;
    trigger.setAttribute('aria-label', `Project ${p.name}`);
    dot.hidden = !isUntitled();
    document.title = `${p.name} · Spellcast3D`;
  }

  /** Turns the name in the header into a text field ("Save as…", "Rename…"). */
  function editName() {
    if (!store.project || menuWrap.querySelector('.pm-edit')) return;
    const untitled = isUntitled();
    const input = h('input', {
      class: 'input pm-edit', type: 'text', maxlength: 80, 'aria-label': 'Project name',
      value: untitled ? '' : store.project.name, placeholder: 'Name your project, e.g. Pirate tavern props',
    });
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const value = input.value.trim();
      input.replaceWith(trigger);
      if (save && value && value !== store.project?.name) {
        name.textContent = value;
        const ok = await attempt(async () => { await renameProject(value); return true; });
        if (!ok) render();
      }
      trigger.focus();
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); finish(true); }
      if (e.key === 'Escape') { e.preventDefault(); finish(false); }
    });
    input.addEventListener('blur', () => finish(true));
    trigger.replaceWith(input);
    input.focus();
    input.select();
  }

  function openMenu() {
    const untitled = isUntitled();
    const others = store.projects
      .filter((p) => p.id !== projectId())
      .sort((a, b) => String(b.updated || '').localeCompare(String(a.updated || '')))
      .slice(0, RECENT);
    popupMenu(trigger, [
      untitled
        ? { label: 'Save as…', icon: 'save', hint: 'Give this project a name', onClick: editName }
        : { label: 'Rename…', icon: 'pencil', onClick: editName },
      { label: 'New project', icon: 'plus', hint: 'Starts empty; name it whenever you like', onClick: () => attempt(() => createProject()) },
      { label: 'Open project folder', icon: 'folder', onClick: () => attempt(() => api('POST', '/api/open-folder', { which: 'project', project: projectId() })) },
      ...(others.length ? ['sep', { heading: 'Recent projects' }] : []),
      ...others.map((p) => ({
        label: p.name, icon: 'cube',
        hint: [plural(p.prop_count || 0, 'prop'), timeAgo(p.updated)].filter(Boolean).join(' · '),
        onClick: () => attempt(() => openProject(p.id)),
      })),
    ], { className: 'project-popup' });
  }

  on('project', render);
  on('projects', render);
  on('status', () => {
    host.classList.toggle('offline', !store.connected);
    syncGear();
  });
  render();
}
