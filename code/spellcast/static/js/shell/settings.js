// Settings: Advanced (Forge Neo's extra model folders, GPU sharing; GET/PATCH /api/settings - and the animations,
// the tooltips and the 3D view's anti-aliasing, which this browser keeps) and Debugging (each tool's state with
// Start / Stop, its live log, the video memory in use). The tools start and stop by themselves; Debugging is where
// to look when something seems stuck. Last, folded away, the Omni-Texture test (omnilab.js): a 3D model from a
// file, OmniTextured without Images and 3D first.

import { api } from '../api.js';
import { h } from '../dom.js';
import { icon } from '../icons.js';
import { RUNNING, TOOL_ETA, TOOL_ROLES, imageTool, on, store, toolName, toolStatus } from '../store.js';
import { restartTool, toggleTool } from '../actions.js';
import { field, segmented, toggle } from '../controls.js';
import { attempt, confirmDialog, modal, toast, toastError } from '../ui.js';
import { licenseChip } from '../licenses.js';
import { logTail, openLogsFolder } from './logtail.js';
import { motionPref, setMotionPref } from './anim.js';
import { plainEdges, setPlainEdges } from './graphics.js';
import { setTooltips, tooltipsOn } from './tooltips.js';
import { omniTextureLab } from './omnilab.js';

export const TOOLS = ['qwen', 'forge', 'pixal3d', 'imesh', 'lato2', 'triflow'];
const MOTION = [
  { value: 'full', label: 'Full', title: 'Every animation: the whole page slides from one stage to the next' },
  { value: 'reduced', label: 'Reduced', title: 'Short, plain moves: the stages cross-fade' },
];
const STATE_LABEL = {
  stopped: 'Off', starting: 'Starting…', ready: 'Ready', busy: 'Working', stopping: 'Stopping…', installing: 'Installing…',
  error: 'Error', not_installed: 'Not installed', unsupported: 'Can’t run here',
};

/** Opens the settings; section 'diagnostics' opens with Debugging unfolded. */
export async function openSettings(section = 'advanced') {
  let s;
  try {
    s = await api('GET', '/api/settings');
  } catch (e) {
    toastError(e);
    return;
  }
  store.settings = s;
  let gpuExclusive = s.gpu_exclusive !== false;
  let motion = motionPref();
  let tooltips = tooltipsOn();
  let plain = plainEdges();
  const dirs = h('textarea', {
    class: 'input mono', rows: 3, spellcheck: 'false',
    placeholder: 'D:\\AI\\models\\Stable-diffusion',
    value: (s.forge?.ckpt_dirs || []).join('\n'),
  });
  const warnings = store.status?.settings_warnings || [];
  const diag = diagnostics();
  let closeDialog = () => {};

  const body = h('div', { class: 'stack settings' },
    warnings.length ? h('div', { class: 'banner warn inline' }, icon('alert'), h('div', null, warnings.map((w) => h('div', null, w)))) : null,
    h('details', { class: 'fold settings-fold', open: section === 'advanced' },
      h('summary', null, 'Advanced'),
      h('div', { class: 'stack' },
        // Forge Neo's own models (Qwen-Image 2.1 brings its own): only while it is the image generator.
        imageTool() === 'forge' ? field('Extra model folders', dirs,
          'Models you keep elsewhere (for example another AI program’s folder), one folder per line. They are used as '
          + 'they are, nothing is copied. Your own go into the models folder inside Spellcast3D.') : null,
        h('button', { class: 'btn ghost small', onclick: () => attempt(() => api('POST', '/api/open-folder', { which: 'models' })) },
          icon('folder'), 'Open the models folder'),
        toggle(gpuExclusive, (v) => { gpuExclusive = v; }, 'Share the GPU: one AI tool at a time (recommended)',
          'Before an AI tool starts (the image generator, the 3D generator, LATO.2, TriFlow), the one on the graphics '
          + 'card is stopped and its video memory freed. Turn this off only with 24 GB of video memory or more.'),
        field('Animations', segmented(MOTION, motion, (v) => { motion = v; }),
          'Full slides the whole page up or down from one stage to the next. Reduced keeps every move short and '
          + 'plain: the stages cross-fade. Only this browser remembers the choice.'),
        toggle(tooltips, (v) => { tooltips = v; }, 'Show tooltips',
          'The explanations that appear when the pointer rests on a button or a picture, also why a button '
          + 'cannot be pressed. Errors are still shown. Only this browser remembers the choice.'),
        toggle(plain, (v) => { plain = v; }, 'Faster 3D view: no anti-aliasing',
          'The 3D model is drawn without smoothing its edges, so they look a little jagged: about a third less work '
          + 'for the graphics chip that draws this page, and less of its memory. Worth it when the 3D view feels '
          + 'slow - a high-resolution screen driven by the processor’s built-in graphics. The page loads again '
          + 'when you save. Only this browser remembers the choice.'))),
    h('details', { class: 'fold settings-fold', open: section === 'diagnostics' },
      h('summary', null, 'Debugging'),
      diag.el),
    h('details', { class: 'fold settings-fold', open: section === 'omnitexture' },
      h('summary', null, 'Omni-Texture test'),
      omniTextureLab({ close: () => closeDialog() })));

  const { done, close } = modal({
    title: 'Settings',
    body,
    wide: true,
    className: 'settings-dialog',
    actions: [
      { label: 'Cancel', value: false },
      {
        label: 'Save', kind: 'primary',
        onclick: async (close) => {
          const ckpt = dirs.value.split(/\r?\n/).map((l) => l.trim().replace(/^"(.*)"$/, '$1')).filter(Boolean);
          // What this browser keeps: applied at once, no restart.
          const moved = motion !== motionPref() || tooltips !== tooltipsOn() || plain !== plainEdges();
          if (motion !== motionPref()) setMotionPref(motion);
          if (tooltips !== tooltipsOn()) setTooltips(tooltips);
          // But for the 3D view's anti-aliasing, which a view's canvas is made with: kept once everything else is
          // saved, and the page then loads again.
          const redrawn = () => {
            if (plain === plainEdges()) return false;
            setPlainEdges(plain);
            location.reload();
            return true;
          };
          // Only what the artist changed is written, so settings.json keeps following the defaults otherwise.
          const patch = {};
          const restart = [];   // running tools that only see the change after a restart
          if (gpuExclusive !== (s.gpu_exclusive !== false)) patch.gpu_exclusive = gpuExclusive;
          if (JSON.stringify(ckpt) !== JSON.stringify(s.forge?.ckpt_dirs || [])) {
            patch.forge = { ckpt_dirs: ckpt };
            restart.push('forge');
          }
          if (!Object.keys(patch).length) {
            close(moved);
            if (!redrawn() && moved) toast('Settings saved.', 'success');
            return;
          }
          try {
            store.settings = await api('PATCH', '/api/settings', patch);
          } catch (e) {
            toastError(e);
            return;
          }
          close(true);
          await offerRestart(restart);
          redrawn();
        },
      },
    ],
  });
  closeDialog = () => close(false);
  done.then(() => diag.stop());
}

/** Tools read these settings only when they start: offer to restart the ones running with the old values. */
async function offerRestart(tools) {
  const running = tools.filter((t) => ['starting', 'ready'].includes(toolStatus(t)?.state));
  if (!running.length) {
    toast(tools.some((t) => toolStatus(t)?.state === 'busy')
      ? 'Settings saved. They apply the next time the tool starts.' : 'Settings saved.', 'success');
    return;
  }
  const names = running.map((t) => TOOL_ROLES[t]).join(' and ');
  const ok = await confirmDialog({
    title: `Restart the ${names}?`,
    message: `The ${names} ${running.length === 1 ? 'is' : 'are'} running with the old settings. Restart now so the new ones apply`
      + `${running.includes('forge') ? ' (the image generator looks at its model folders only when it starts)' : ''}?`
      + (running.length === 1 ? ` Starting again takes ${TOOL_ETA[running[0]] || 'a moment'}.` : ''),
    ok: 'Restart', cancel: 'Later',
  });
  if (ok) for (const t of running) await attempt(() => restartTool(t));
  else toast('Settings saved. They apply the next time the tool starts.', 'success');
}

// ---- Debugging (the code's "diagnostics") ---------------------------------------------------------

function diagnostics() {
  const vramFill = h('span', { class: 'vram-fill' });
  const vramText = h('span', { class: 'vram-text' });
  const vram = h('div', { class: 'diag-vram' },
    h('div', { class: 'diag-vram-head' }, h('b', null, 'Video memory'), vramText),
    h('div', { class: 'vram-bar' }, vramFill));
  const rows = Object.fromEntries(TOOLS.map((tool) => {
    const state = h('span', { class: 'state-chip' });
    const message = h('div', { class: 'hint diag-msg' });
    const logBox = h('div', { class: 'diag-log', hidden: true });
    const action = h('button', { type: 'button', class: 'btn ghost small', onclick: () => attempt(() => toggleTool(tool)) });
    const logBtn = h('button', {
      type: 'button', class: 'btn ghost small',
      onclick: () => {
        logBox.hidden = !logBox.hidden;
        logBox.replaceChildren(logBox.hidden ? '' : logTail(tool));
        logBtn.classList.toggle('on', !logBox.hidden);
      },
    }, icon('terminal'), 'Log');
    const el = h('div', { class: 'diag-tool' },
      h('div', { class: 'diag-tool-head' },
        h('span', { class: 'dot' }),
        h('div', { class: 'diag-tool-name' },
          h('span', { class: 'diag-tool-title' }, h('b', null, toolName(tool)), licenseChip(tool)),
          h('span', { class: 'hint' }, TOOL_ROLES[tool])),
        state, action, logBtn),
      message, logBox);
    return [tool, { el, state, message, action }];
  }));
  // The graphics card measured again for the 3D step (install/components.py: the installer did it once at the
  // install; it never measures by itself afterwards). A few minutes, with heavy programs closed; 3D jobs wait.
  const measure = h('button', {
    type: 'button', class: 'btn ghost small',
    onclick: async () => {
      const ok = await confirmDialog({
        title: 'Measure the graphics card again?',
        message: 'Spellcast3D measures how the 3D generator runs fastest on this graphics card: a few minutes, during '
          + 'which 3D jobs wait. Close heavy programs (Blender, games, videos, Photoshop, a browser with many tabs) '
          + 'first: a program using the card while it measures makes the measurement wrong.',
        ok: 'Measure now',
      });
      if (!ok) return;
      const done = await attempt(() => api('POST', '/api/install/calibrate'));
      if (done) toast('Measuring the graphics card: the bar under the header shows how far it is.', 'info', 6000);
    },
  }, icon('gauge'), 'Measure the graphics card again');
  const measureRow = h('div', { class: 'diag-measure', hidden: true }, measure,
    h('span', { class: 'hint' }, 'For the 3D step; done once at the install, never by itself afterwards.'));
  const el = h('div', { class: 'stack diag' },
    vram,
    h('div', { class: 'diag-tools' }, TOOLS.map((t) => rows[t].el)),
    measureRow,
    h('p', { class: 'hint' },
      'The tools start by themselves when something needs them. Stop one to free its memory for another program; '
      + 'it starts again when it is needed. ',
      h('a', { href: '#', onclick: (e) => { e.preventDefault(); openLogsFolder(); } }, 'Open the logs folder')));

  function render() {
    const gpu = store.status?.gpu;
    if (gpu?.vram_total_mb) {
      const f = gpu.vram_used_mb / gpu.vram_total_mb;
      vramFill.style.width = `${Math.round(f * 100)}%`;
      vram.dataset.level = f > 0.92 ? 'high' : f > 0.8 ? 'mid' : 'low';
      vramText.textContent = `${(gpu.vram_used_mb / 1024).toFixed(1)} of ${(gpu.vram_total_mb / 1024).toFixed(0)} GB in use by all programs · ${gpu.name}`;
    } else {
      vramFill.style.width = '0';
      vramText.textContent = store.status ? 'No NVIDIA graphics card was found.' : '';
    }
    for (const tool of TOOLS) {
      const st = toolStatus(tool);
      const r = rows[tool];
      // A tool this Spellcast3D does not have (an older hub) has no row.
      r.el.hidden = !!store.status?.tools && !st;
      const s = st?.state || 'unknown';
      r.el.dataset.state = s;
      r.el.querySelector('.diag-tool-head').dataset.state = s;
      r.state.textContent = STATE_LABEL[s] || s;
      r.state.dataset.state = s;
      r.message.textContent = s === 'error' || s === 'unsupported' || s === 'not_installed' ? st?.reason || st?.message || '' : '';
      r.message.hidden = !r.message.textContent;
      const canRun = st && st.supported !== false && st.installed !== false && !['unsupported', 'not_installed', 'installing', 'stopping'].includes(s);
      r.action.hidden = !canRun;
      r.action.replaceChildren(...(RUNNING.has(s) ? [icon('stop'), 'Stop · free memory'] : [icon('power'), 'Start']));
    }
    const p3d = toolStatus('pixal3d');
    const installing = !!store.status?.install?.running;
    measureRow.hidden = !(p3d && p3d.installed && p3d.supported !== false);
    measure.disabled = installing || !!store.jobs?.some((j) => ['queued', 'starting_tool', 'running'].includes(j.status));
  }
  const off = on('status', render);
  render();
  return { el, stop: off };
}
