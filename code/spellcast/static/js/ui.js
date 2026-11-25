// Toasts and modal dialogs.

import { h } from './dom.js';
import { icon } from './icons.js';

const TOAST_ICON = { info: 'info', success: 'check', error: 'alert', warn: 'alert' };

export function toast(message, kind = 'info', timeout = kind === 'error' ? 10000 : 4500) {
  const host = document.getElementById('toasts');
  const el = h('div', { class: `toast ${kind}`, role: kind === 'error' ? 'alert' : 'status' },
    icon(TOAST_ICON[kind] || 'info'),
    h('div', { class: 'toast-text' }, message),
    h('button', { class: 'icon-btn small', title: 'Dismiss', onclick: () => dismiss() }, icon('x')));
  const dismiss = () => {
    el.classList.add('leaving');
    setTimeout(() => el.remove(), 200);
  };
  host.append(el);
  while (host.children.length > 5) host.firstChild.remove();
  if (timeout) setTimeout(dismiss, timeout);
}

export const toastError = (err) => toast(err?.message || String(err), 'error');

/**
 * Thrown by a stage view's onPlus / onCta when the action cannot happen yet ("No image yet - import or
 * generate one first."): the props column shakes the button and shows the message next to it instead of
 * an error toast (an empty message: it only shakes).
 */
export class ShellRefusal extends Error {
  constructor(message) {
    super(message);
    this.name = 'ShellRefusal';
  }
}

/** Runs fn; on failure shows the error as a toast and resolves to undefined. */
export async function attempt(fn) {
  try {
    return await fn();
  } catch (err) {
    console.warn(err);
    toastError(err);
    return undefined;
  }
}

export const dialogOpen = () => !!document.querySelector('dialog[open]');

/**
 * Opens a modal. actions: [{label, kind: 'primary'|'danger'|'ghost', value, onclick?}] - an action
 * without onclick closes the dialog with its value. Resolves with the value (undefined on Esc / backdrop).
 */
export function modal({ title, body, actions = [], wide = false, className = '' }) {
  const dlg = h('dialog', { class: `modal ${wide ? 'wide' : ''} ${className}` });
  let result;
  const close = (value) => { result = value; dlg.close(); };
  const buttons = actions.map((a) => h('button', {
    type: 'button',
    class: `btn ${a.kind || 'ghost'}`,
    onclick: () => (a.onclick ? a.onclick(close) : close(a.value)),
  }, a.label));
  dlg.append(h('div', { class: 'modal-inner' },
    h('div', { class: 'modal-head' },
      h('h2', null, title),
      h('button', { type: 'button', class: 'icon-btn', title: 'Close', onclick: () => close(undefined) }, icon('x'))),
    h('div', { class: 'modal-body' }, body),
    buttons.length ? h('div', { class: 'modal-actions' }, buttons) : null));
  // A press on the backdrop (outside the inner panel) closes the dialog.
  dlg.addEventListener('mousedown', (e) => { if (e.target === dlg) close(undefined); });
  document.body.append(dlg);
  dlg.showModal();
  const done = new Promise((resolve) => dlg.addEventListener('close', () => { dlg.remove(); resolve(result); }));
  return { dialog: dlg, close, done };
}

/** Resolves true when confirmed. `message`: its words, or a list of paragraphs. With cancel: null it is a plain message with one button. */
export async function confirmDialog({ title, message, ok = 'OK', danger = false, cancel = 'Cancel' }) {
  const m = modal({
    title,
    body: [message].flat().map((words) => h('p', { class: 'modal-text' }, words)),
    actions: [
      ...(cancel === null ? [] : [{ label: cancel, value: false }]),
      { label: ok, kind: danger ? 'danger' : 'primary', value: true },
    ],
  });
  const buttons = m.dialog.querySelectorAll('.modal-actions .btn');
  // A destructive question starts on Cancel (an extra Enter must not delete or stop anything); others on OK.
  (danger ? buttons[0] : buttons[buttons.length - 1]).focus();
  return (await m.done) === true;
}
