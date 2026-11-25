// The artist's prompt library: short named prompts kept for reuse, one list for every project (the hub's
// prompt-library.json, promptlib.py). Under the words, inside their field: a chip per prompt ticked "Show", then
// "+prompt", which opens the "Prompt Library" panel over the viewport, where prompts are added (+ Add), named (10
// characters at most), written and ticked; a click on a chip puts its text where the text cursor is in the words.

import { api } from '../api.js';
import { h } from '../dom.js';
import { icon } from '../icons.js';
import { emit, on, store } from '../store.js';
import { attempt } from '../ui.js';

export const NAME_MAX = 10;
const SAVE_MS = 500;

// The list as the page has it: store.promptLibrary ([{id, name, text, shown}], null until the hub has answered).
let loading = null;
function loadLibrary() {
  loading ||= api('GET', '/api/prompt-library')
    .then((answer) => {
      store.promptLibrary = answer.prompts || [];
      emit('prompts');
    })
    .catch((e) => {
      loading = null;   // asked again by the next words field
      console.warn('Could not read the prompt library:', e);
    });
  return loading;
}

let saveTimer = null;
const save = () => attempt(() => api('PUT', '/api/prompt-library', { prompts: store.promptLibrary || [] }));
/** The list changed here: the chips follow at once, the hub shortly after (savePending: at once). */
function changed() {
  emit('prompts');
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, SAVE_MS);
}
function savePending() {
  if (!saveTimer) return;
  clearTimeout(saveTimer);
  saveTimer = null;
  save();
}

/**
 * `text` into the field `field` where its text cursor is (or over what is selected), a space before it when it would
 * stick to a word; then the field hears it as typed (its input event) and keeps the focus, the cursor after it.
 */
export function insertAt(field, text) {
  const start = field.selectionStart ?? field.value.length;
  const end = field.selectionEnd ?? start;
  const before = field.value.slice(0, start);
  const gap = before && !/\s$/.test(before) && !/^\s/.test(text) ? ' ' : '';
  field.value = before + gap + text + field.value.slice(end);
  field.selectionStart = field.selectionEnd = start + gap.length + text.length;
  field.focus();
  field.dispatchEvent(new Event('input', { bubbles: true }));
}

/**
 * The chips under the words field `field`: one per shown prompt, then "+prompt" (`open`: the library), on as many
 * lines as they need. A chip takes no focus, so the text cursor stays where it was. destroy() when the field goes.
 */
export function promptBar(field, open) {
  const el = h('div', { class: 'pl-bar' });
  const keepCursor = (e) => e.preventDefault();
  function render() {
    const shown = (store.promptLibrary || []).filter((p) => p.shown && p.name);
    el.replaceChildren(
      ...shown.map((p) => h('button', {
        type: 'button', class: 'pl-chip', title: p.text || '(no words yet)',
        onmousedown: keepCursor, onclick: () => p.text && insertAt(field, p.text),
      }, p.name)),
      h('button', {
        type: 'button', class: 'pl-chip pl-open', title: 'Your prompt library: prompts kept for every project',
        onmousedown: keepCursor, onclick: open,
      }, '+prompt'));
  }
  const off = on('prompts', render);
  render();
  loadLibrary();
  el.destroy = off;
  return el;
}

/**
 * The "Prompt Library" panel, over the viewport while open: a × in either top corner goes back to the picture, + Add
 * puts a new prompt at the end of the list (scrolled to it), and the prompts lie below, one long row each - the name,
 * the words, then "Show" over the bin - scrolling down when they are many. Edits save as they are made. open() /
 * close(); destroy() when the workbench goes.
 */
export function libraryPanel() {
  const list = h('div', { class: 'pl-list' });
  const empty = h('p', { class: 'pl-empty' },
    'No prompts yet. Press + to add one: a short name for its button, and the words it puts into your prompt.');
  const closer = () => h('button', {
    type: 'button', class: 'icon-btn pl-close', title: 'Back to the picture (Esc)', 'aria-label': 'Close the prompt library',
    onclick: () => panel.close(),
  }, icon('x'));
  const el = h('div', { class: 'pl-panel', hidden: true, role: 'dialog', 'aria-label': 'Prompt Library' },
    h('div', { class: 'pl-head' },
      closer(),
      h('h3', null, 'Prompt Library'),
      h('button', { type: 'button', class: 'btn small pl-add', title: 'Add a prompt', onclick: add }, icon('plus'), 'Add'),
      closer()),
    empty, list);

  function row(p) {
    const show = h('input', { type: 'checkbox', checked: !!p.shown, 'aria-label': `Show “${p.name}” under the words` });
    show.addEventListener('change', () => {
      p.shown = show.checked;
      changed();
    });
    const name = h('input', {
      class: 'input pl-name', value: p.name, maxlength: NAME_MAX, placeholder: 'Name', spellcheck: 'false',
      'aria-label': `Its name (${NAME_MAX} characters at most)`,
    });
    name.addEventListener('input', () => {
      p.name = name.value.slice(0, NAME_MAX);
      changed();
    });
    const text = h('textarea', {
      class: 'input pl-text', rows: 2, value: p.text, spellcheck: 'true', placeholder: 'The words it puts into your prompt',
      'aria-label': 'Its words',
    });
    text.addEventListener('input', () => {
      p.text = text.value;
      changed();
    });
    const bin = h('button', {
      type: 'button', class: 'icon-btn small pl-bin', title: 'Delete this prompt', 'aria-label': 'Delete this prompt',
      onclick: () => {
        store.promptLibrary = (store.promptLibrary || []).filter((q) => q !== p);
        changed();
        render();
      },
    }, icon('trash'));
    const el = h('div', { class: 'pl-row' }, name, text,
      h('div', { class: 'pl-side' }, h('label', { class: 'pl-show', title: 'Show it under the words' }, show, 'Show'), bin));
    el.name = name;
    return el;
  }

  function render() {
    const prompts = store.promptLibrary || [];
    empty.hidden = prompts.length > 0;
    list.replaceChildren(...prompts.map(row));
  }

  function add() {
    const p = { id: `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, name: '', text: '', shown: true };
    store.promptLibrary = [...(store.promptLibrary || []), p];
    changed();
    render();
    const last = list.lastElementChild;
    last?.name.focus({ preventScroll: true });
    list.scrollTop = list.scrollHeight;
    last?.scrollIntoView?.({ block: 'nearest' });
  }

  const onKey = (e) => {
    if (e.key !== 'Escape' || el.hidden) return;
    e.preventDefault();
    e.stopPropagation();
    panel.close();
  };
  document.addEventListener('keydown', onKey, true);

  const panel = {
    el,
    open() {
      el.hidden = false;
      loadLibrary().then(() => { if (!el.hidden) render(); });
      render();
    },
    close() {
      el.hidden = true;
      savePending();
    },
    destroy() {
      document.removeEventListener('keydown', onKey, true);
      savePending();
    },
  };
  return panel;
}
