// The "faster" switches, each in its stage's panel: a shortcut that saves time but no longer draws what the exact
// path draws. Not a prop's setting but the hub's (GET / PATCH /api/settings): it holds for every prop and project of
// its stage, so a switch asks before it changes. A tool picks its arithmetic when it starts: a running one is
// restarted for the switch's stage (`stage`: the hub starts it anew whenever a job of the other stage wants other
// arithmetic), and while it works the switch waits. What the window says comes in short paragraphs (on / off: their
// words - what it changes on a line of its own, then what that does; the restart is a paragraph of its own).
//   images       "Faster image generation" in Images: Qwen-Image's int8 matmuls for its pictures (qwen.int8_matmul;
//                adapters/qwen.py matmul_arg)
//   omnitexture  the same for Omni-Texture's repaints (qwen.int8_matmul_omnitexture)
//   mesh         "Faster 3D generation": Pixal3D's conditioning and coarse stages on their fast paths too
//                (pixal3d.coarse_exact false; the fork's modules/exact.py), which changes something only on RTX 30xx
//                and newer: older cards cannot compute the original's numbers either way

import { api } from '../api.js';
import { restartTool } from '../actions.js';
import { h } from '../dom.js';
import { toggle } from '../controls.js';
import { TOOL_ETA, imageTool, store, toolStatus } from '../store.js';
import { attempt, confirmDialog, toast, toastError } from '../ui.js';

const MODELS = 'all 3D generation: every 3D model made in the 3D stage, for every prop and project';

/** A paragraph of two lines: what the switch changes, then what turning it off does. */
const twoLines = (changes, then) => h('span', null, `This changes ${changes}.`, h('br'), then);

/**
 * Qwen-Image's int8 matmuls for one stage's pictures: `what` they are there, `section` the stage as the window's
 * title names it, `key` its hub setting under qwen.
 */
function imageSwitch(stage, { what, section, changes, hint, key }) {
  return {
    kind: stage,
    stage,
    tool: 'qwen',
    title: 'Faster image generation',
    name: `faster image generation ${section}`,
    hint,
    on: () => [
      `This changes ${changes}, until you turn it off.`,
      h('span', null, `${what} take about 20 to 40 % less time, with a shortcut in the image generator’s arithmetic: `
        + 'small details (an eye, a buckle, a roof’s edge) can come out differently. ',
      h('b', null, 'Some instructions may be ignored, for example a character sheet of four views.')),
    ],
    off: () => [twoLines(changes, `${what} come out exactly as the model draws them again, and take longer.`)],
    restarts: 'The image generator restarts now',
    busy: 'The image generator is drawing right now. Change this when it is done.',
    applies: () => imageTool() === 'qwen',
    isOn: (settings) => settings?.qwen?.[key] === true,
    patch: (on) => ({ qwen: { [key]: on } }),
  };
}

export const FASTER = {
  images: imageSwitch('images', {
    what: 'Pictures',
    section: 'in Images',
    changes: 'every picture made in Images, for every prop and project (Omni-Texture has a switch of its own)',
    hint: 'For all pictures in Images, not only this prop. Small details can come out differently.',
    key: 'int8_matmul',
  }),
  omnitexture: imageSwitch('omnitexture', {
    what: 'Repaints',
    section: 'for Omni-Texture',
    changes: 'every Omni-Texture repaint, for every prop and project (Images has a switch of its own)',
    hint: 'For all Omni-Texture repaints, not only this prop. Small details can come out differently.',
    key: 'int8_matmul_omnitexture',
  }),
  mesh: {
    kind: 'mesh',
    tool: 'pixal3d',
    title: 'Faster 3D generation',
    name: 'faster 3D generation',
    hint: 'For all 3D models, not only this prop. A seed then gives another model than the original Pixal3D.',
    on: () => [
      `This changes ${MODELS}, until you turn it off.`,
      'Each model takes a few seconds less, but a seed no longer gives the model the original Pixal3D makes: its '
        + 'rough shape comes out a little differently, and the model that grows from it is another one - sometimes '
        + 'better, sometimes worse.',
    ],
    off: () => [twoLines(MODELS, 'A seed gives the model the original Pixal3D makes again, and each model takes a few '
      + 'seconds longer.')],
    restarts: 'Pixal3D restarts now',
    busy: 'Pixal3D is making 3D models right now. Change this when it is done.',
    applies: () => (store.status?.gpu?.compute_cap ?? 0) >= 8,
    isOn: (settings) => settings?.pixal3d?.coarse_exact === false,
    patch: (on) => ({ pixal3d: { coarse_exact: !on } }),
  },
};

let reading = null;   // the hub's settings while they are being asked for: two switches ask once

function readSettings() {
  reading ??= api('GET', '/api/settings')
    .then((s) => { store.settings = s; })
    .catch(() => {})
    .finally(() => { reading = null; });
  return reading;
}

/**
 * FASTER[kind]'s shortcut is on, as far as the page knows the hub's settings: it asks for them when it does not
 * (the answer shows at the caller's next look).
 */
export function fasterOn(kind) {
  const sw = FASTER[kind];
  if (!sw.applies()) return false;
  if (!store.settings) readSettings();
  return sw.isOn(store.settings);
}

/**
 * The switch of FASTER[kind], for its stage's panel: it shows the hub's setting (asked for once while it applies
 * to this PC) and changes it once the artist has agreed. `el.applies()` says whether it changes anything here;
 * `el.sync()` follows the hub's settings as the page has them.
 */
export function fasterSwitch(kind) {
  const sw = FASTER[kind];
  const el = toggle(sw.isOn(store.settings), (wanted) => change(sw, el.input, wanted), sw.title, sw.hint);
  el.applies = sw.applies;
  el.sync = () => {
    if (store.settings) el.input.checked = sw.isOn(store.settings);
    else if (sw.applies()) readSettings().then(() => { el.input.checked = sw.isOn(store.settings); });
  };
  el.sync();
  return el;
}

async function change(sw, input, wanted) {
  input.checked = !wanted;                      // as it is, until the artist agrees and the hub has it
  const state = toolStatus(sw.tool)?.state;
  if (state === 'busy') {
    toast(sw.busy);
    return;
  }
  const running = state === 'starting' || state === 'ready';
  const ok = await confirmDialog({
    title: `${wanted ? 'Turn on' : 'Turn off'} ${sw.name}?`,
    message: [...(wanted ? sw.on() : sw.off()), ...(running ? [`${sw.restarts} (${TOOL_ETA[sw.tool]}).`] : [])],
    ok: wanted ? 'Turn on' : 'Turn off',
  });
  if (!ok) return;
  try {
    store.settings = await api('PATCH', '/api/settings', sw.patch(wanted));
  } catch (e) {
    toastError(e);
    return;
  }
  input.checked = wanted;
  if (running) await attempt(() => restartTool(sw.tool, sw.stage ? { stage: sw.stage } : null));
}
