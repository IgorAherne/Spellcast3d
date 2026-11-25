// A live log: the tail of a tool's code/logs/<tool>.log, or of install.log ('install'), followed while it
// is on screen (Settings > Debugging, the status line's and the install card's "Details").

import { api } from '../api.js';
import { h } from '../dom.js';
import { TOOL_ROLES, toolName } from '../store.js';
import { attempt, modal } from '../ui.js';

/** <pre> that follows GET /api/tools/{tool}/log (or /api/install/log) while it is in the page. */
export function logTail(tool, lines = 60) {
  const path = tool === 'install' ? '/api/install/log' : `/api/tools/${tool}/log`;
  const pre = h('pre', { class: 'log' }, 'Loading the log…');
  let wasConnected = false;
  let waits = 0;   // never put on the page: give up after a while
  const tick = async () => {
    if (pre.isConnected) {
      wasConnected = true;
      if (!document.hidden) {   // nobody can read it in a hidden tab
        try {
          const data = await api('GET', `${path}?tail=${lines}`);
          const stick = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24;
          pre.textContent = (data?.lines || []).join('\n') || 'No output yet.';
          if (stick) pre.scrollTop = pre.scrollHeight;
        } catch (e) {
          pre.textContent = e.message;
        }
      }
    } else if (wasConnected || ++waits > 20) {
      return;   // removed from the page
    }
    setTimeout(tick, 1500);
  };
  tick();
  return pre;
}

export const openLogsFolder = () => attempt(() => api('POST', '/api/open-folder', { which: 'logs' }));

/** A dialog with a live log ("Details"): a tool's, or the installation's ('install'). */
export function showLog(tool) {
  const install = tool === 'install';
  modal({
    title: install ? 'Installation log' : `${toolName(tool)} · ${TOOL_ROLES[tool]}`,
    wide: true,
    body: h('div', { class: 'stack' },
      h('p', { class: 'modal-text' }, `The last lines ${install ? 'the installer' : 'this tool'} wrote. They help when you ask `
        + 'for help or report a problem.'),
      logTail(tool, 120)),
    actions: [
      { label: 'Open the logs folder', onclick: () => openLogsFolder() },
      { label: 'Close', kind: 'primary', value: true },
    ],
  });
}
