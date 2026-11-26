// Two small floating helpers of the shell: a popup menu anchored to a button (a prop's menu, the
// project menu) and a quick tooltip (a prop's status, the call to action's explanation and its
// refusals), both drawn above everything so no scrolling list clips them.

import { h } from '../dom.js';
import { icon } from '../icons.js';
import { tipShows } from './tooltips.js';

let openMenu = null;   // {el, anchor, close}

/**
 * Opens a menu under `anchor`. items: [{label, icon?, lead? (an element in the icon's place), hint?,
 * danger?, disabled?, current? (the one in use: marked, and focused first), onClick}] or 'sep' or
 * {heading}. `head` is an optional element shown above the items. Resolves when the menu closes.
 */
export function popupMenu(anchor, items, { head = null, align = 'left', className = '' } = {}) {
  if (openMenu?.anchor === anchor) {   // a second click on the same button closes it
    openMenu.close();
    return Promise.resolve();
  }
  openMenu?.close();
  return new Promise((resolve) => {
    const buttons = [];
    let first = null;
    const el = h('div', { class: `popup-menu ${className}`, role: 'menu' },
      head,
      items.map((it) => {
        if (it === 'sep') return h('div', { class: 'menu-sep', role: 'separator' });
        if (it.heading) return h('div', { class: 'menu-heading' }, it.heading);
        const b = h('button', {
          type: 'button', role: 'menuitem', class: `menu-item${it.danger ? ' danger' : ''}${it.current ? ' current' : ''}`,
          disabled: !!it.disabled, title: it.title || null, 'aria-current': it.current ? 'true' : null,
          onclick: () => { close(); it.onClick(); },
        }, it.lead || (it.icon ? icon(it.icon) : h('span', { class: 'menu-noicon' })),
        h('span', { class: 'menu-label' }, it.label, it.hint ? h('small', null, it.hint) : null),
        it.current ? icon('check', 'menu-current') : null);
        buttons.push(b);
        if (it.current && !it.disabled) first = b;
        return b;
      }));
    document.body.append(el);
    const r = anchor.getBoundingClientRect();
    const w = el.offsetWidth;
    const left = align === 'right' ? r.right - w : r.left;
    el.style.left = `${Math.max(8, Math.min(left, innerWidth - w - 8))}px`;
    const below = r.bottom + 6;
    el.style.top = `${below + el.offsetHeight > innerHeight - 8 ? Math.max(8, r.top - el.offsetHeight - 6) : below}px`;
    anchor.setAttribute('aria-expanded', 'true');
    anchor.classList.add('menu-open');

    const onDown = (e) => { if (!el.contains(e.target) && !anchor.contains(e.target)) close(); };
    const onKey = (e) => {
      if (e.key === 'Escape') { e.preventDefault(); close(); anchor.focus(); }
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        const live = buttons.filter((b) => !b.disabled);
        const i = live.indexOf(document.activeElement);
        live[(i + (e.key === 'ArrowDown' ? 1 : -1) + live.length) % live.length]?.focus();
      }
    };
    function close() {
      if (openMenu?.el !== el) return;
      openMenu = null;
      el.remove();
      anchor.setAttribute('aria-expanded', 'false');
      anchor.classList.remove('menu-open');
      document.removeEventListener('mousedown', onDown, true);
      document.removeEventListener('keydown', onKey, true);
      window.removeEventListener('resize', close);
      resolve();
    }
    document.addEventListener('mousedown', onDown, true);
    document.addEventListener('keydown', onKey, true);
    window.addEventListener('resize', close);
    openMenu = { el, anchor, close };
    el.animate([{ opacity: 0, transform: 'translateY(-4px)' }, { opacity: 1, transform: 'none' }], { duration: 120, easing: 'ease-out' });
    (first || buttons.find((b) => !b.disabled))?.focus({ preventScroll: true });
  });
}

export const menuOpen = () => !!openMenu;

/** Closes the open menu when its button is inside `el` (a part of the page being drawn anew). */
export function closeMenuIn(el) {
  if (openMenu && el.contains(openMenu.anchor)) openMenu.close();
}

// ---- tooltip ------------------------------------------------------------------------------------

let tip = null;
let tipFor = null;

/**
 * Shows `text` next to `anchor` (right of it, or above with side: 'top') until hideTip(anchor).
 * kind: '' (a short tooltip), 'note' (a sentence explaining a button), 'refusal' (why it said no), 'problem'
 * (what went wrong; it looks like a short tooltip). The first two show only while the viewer keeps tooltips on
 * (tooltips.js).
 */
export function showTip(anchor, text, side = 'right', kind = '') {
  if (!text || !tipShows(kind)) return hideTip(anchor);
  tip ||= h('div', { class: 'tip-pop', role: 'tooltip', id: 'tip-pop' });
  tip.textContent = text;
  tip.dataset.side = side;
  tip.dataset.kind = kind;
  document.body.append(tip);
  tipFor = anchor;
  const r = anchor.getBoundingClientRect();
  const w = tip.offsetWidth;
  const hh = tip.offsetHeight;
  let left = side === 'right' ? r.right + 10 : r.left + r.width / 2 - w / 2;
  let top = side === 'right' ? r.top + r.height / 2 - hh / 2 : r.top - hh - 8;
  if (side === 'top' && top < 8) top = r.bottom + 8;
  left = Math.max(8, Math.min(left, innerWidth - w - 8));
  top = Math.max(8, Math.min(top, innerHeight - hh - 8));
  tip.style.left = `${left}px`;
  tip.style.top = `${top}px`;
  // A sentence next to a button points at it.
  tip.style.setProperty('--caret-y', `${Math.round(Math.min(hh - 10, Math.max(10, r.top + r.height / 2 - top)))}px`);
  return undefined;
}

/** The tooltip shows for `anchor` now. */
export const tipShown = (anchor) => !!tip && tipFor === anchor && tip.isConnected;

/** New words for the tooltip of `anchor` if it is the one showing (its status changed under the pointer). */
export function retip(anchor, text) {
  if (tipShown(anchor)) showTip(anchor, text, tip.dataset.side, tip.dataset.kind);
}

export function hideTip(anchor = null) {
  if (!tip || (anchor && tipFor !== anchor)) return;
  tip.remove();
  tipFor = null;
}
