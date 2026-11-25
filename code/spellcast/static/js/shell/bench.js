// The workbench, right of the props column: the stage view of the selected prop (or the one the stage
// follows). It has all the height: the call to action
// lives in the props column. Going to another stage, the whole page swipes: a still copy of the page
// being left (the props column and the workbench) swipes out while the page drawn for the new stage
// swipes in (anim.js swipe).

import { h } from '../dom.js';

/** Pairs each element matching `sel` in `a` with its counterpart in `b`, a copy of `a`. */
function counterparts(a, b, sel) {
  const copies = b.querySelectorAll(sel);
  return [...a.querySelectorAll(sel)].map((el, i) => [el, copies[i]]);
}

export function createBench() {
  const view = h('div', { class: 'centre-view' });
  const el = h('section', { class: 'bench', 'aria-label': 'Workbench' }, view);

  /**
   * A still copy of the whole page as it is now (the shell the workbench is in: the props column and the
   * workbench), laid over it: the page that leaves during a move, so the heavy view is not live
   * meanwhile. A copy loses what is typed into its fields, what its canvases show (a 3D preview) and
   * where its panels are scrolled (the props list): those are carried over. An iframe (Instant Meshes'
   * viewer) is left blank, never loaded a second time. The copy is only a picture: inert, hidden from
   * screen readers, without ids or live regions. null when there is nothing to copy.
   */
  async function freeze() {
    const page = el.parentElement;
    if (!page || !view.firstElementChild) return null;
    const copy = page.cloneNode(true);
    copy.classList.add('shell-frozen');
    copy.inert = true;
    copy.setAttribute('aria-hidden', 'true');
    for (const b of copy.querySelectorAll('[id]')) b.removeAttribute('id');   // never two of one id in the page
    for (const b of copy.querySelectorAll('[data-stage]')) b.removeAttribute('data-stage');
    for (const b of copy.querySelectorAll('[role="alert"], [role="status"], [aria-live]')) {
      b.removeAttribute('role');
      b.removeAttribute('aria-live');
    }
    for (const [a, b] of counterparts(page, copy, 'input, textarea, select')) {
      b.value = a.value;
      b.checked = a.checked;
    }
    for (const [, b] of counterparts(page, copy, 'iframe')) b.replaceWith(h('div', { class: b.className }));
    const canvases = counterparts(page, copy, 'canvas');
    if (canvases.length) {
      const { viewer } = await import('../viewer3d.js');
      for (const [a, b] of canvases) b.replaceWith(viewer.still(a));
    }
    // Scrolled panels keep their place in the copy.
    for (const [a, b] of counterparts(page, copy, '*')) {
      if (a.scrollTop || a.scrollLeft) {
        b.dataset.scrollTop = a.scrollTop;
        b.dataset.scrollLeft = a.scrollLeft;
      }
    }
    page.after(copy);
    for (const b of copy.querySelectorAll('[data-scroll-top]')) {
      b.scrollTop = +b.dataset.scrollTop;
      b.scrollLeft = +b.dataset.scrollLeft;
    }
    return copy;
  }

  return { el, view, freeze };
}
