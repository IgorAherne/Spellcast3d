// A sideways row of tiles under a viewport (the Images workbench's pictures), packed against its right end: a mouse
// drag scrolls it (the click that ends a drag picks nothing), the wheel scrolls it sideways, and while the pointer is
// on it the row holds still: a removed tile's room is kept at the row's last end, so the tiles after it slide one place
// left and the next one's × lands under the pointer - the way a browser's tabs wait for the pointer to leave before
// they settle. When it leaves, the row settles against its right end again, the tiles gliding to their new places.
// The row's CSS gives "held" its meaning: its last end (::after) grows by --tail, the room of the tiles removed
// (stages-images.css). With `carry`, a drag that leaves the row (above or below it) lifts the pressed tile's picture
// out instead of scrolling: the row goes back to where it was, the picture follows the pointer, and where
// carry.over(x, y) says yes a release gives it to carry.drop(tile).

import { fullMotion } from '../shell/anim.js';

const DRAG_PX = 5;          // a press that moves less than this is a click
const CARRY_PX = 12;        // this far above or below the row, a drag lifts the picture out
const SETTLE_MS = 220;

/**
 * The row `el` (its children are the tiles): {removing(tile), reveal(tile, {fromEnd})}. removing(tile) just before a
 * tile leaves (the row keeps its room while the pointer is on it); reveal(tile) scrolls it into view unless the artist
 * is working on the row (a row not laid out yet tries again on the next frames), fromEnd: from the row's right end.
 * carry: {over(x, y) -> whether a release there takes the picture, drop(tile), end()} (null: no carrying).
 */
export function createTileRow(el, { carry = null } = {}) {
  let inside = false;       // the pointer is on the row
  let held = false;
  let tail = 0;             // the room of the tiles removed while held
  let pan = null;           // {id, x, left, moved, tile, ghost} from a press until its release
  let swallow = false;      // the click that ends a drag

  /** The row lets go: the tiles glide from where they were to where they settle. */
  function release() {
    if (!held) return;
    const before = new Map([...el.children].map((t) => [t, t.getBoundingClientRect().left]));
    held = false;
    tail = 0;
    el.classList.remove('held');
    el.style.removeProperty('--tail');
    if (!fullMotion()) return;
    for (const [t, x] of before) {
      const dx = x - t.getBoundingClientRect().left;
      if (Math.abs(dx) > 0.5) {
        t.animate([{ transform: `translateX(${dx}px)` }, { transform: 'none' }],
          { duration: SETTLE_MS, easing: 'cubic-bezier(.2, .8, .2, 1)' });
      }
    }
  }

  el.addEventListener('pointerenter', () => { inside = true; });
  el.addEventListener('pointerleave', () => {
    inside = false;
    if (!pan) release();
  });

  // ---- a drag scrolls (a mouse or a pen: a finger scrolls the row by itself) ----------------------------
  el.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'touch' || e.button !== 0) return;
    const tile = [...el.children].find((t) => t.contains?.(e.target)) || null;
    pan = { id: e.pointerId, x: e.clientX, left: el.scrollLeft, moved: false, tile, ghost: null };
    // Followed on the window: a quick drag can be off the row before the row holds the pointer.
    globalThis.addEventListener('pointermove', follow, true);
    globalThis.addEventListener('pointerup', endPan, true);
    globalThis.addEventListener('pointercancel', endPan, true);
  });
  function follow(e) {
    if (pan?.id !== e.pointerId) return;
    const dx = e.clientX - pan.x;
    const out = !!carry && !!pan.tile && !pan.ghost && outside(e.clientY);
    if (!pan.moved) {
      if (!out && Math.abs(dx) < DRAG_PX) return;
      pan.moved = true;
      try { el.setPointerCapture(e.pointerId); } catch { /* the pointer is gone already */ }
      el.classList.add('panning');
    }
    if (out) {
      el.scrollLeft = pan.left;                    // the picture is carried, the row is not scrolled
      el.classList.remove('panning');
      pan.ghost = lifted(pan.tile);
    }
    if (!pan.ghost) {
      el.scrollLeft = pan.left - dx;
      return;
    }
    pan.ghost.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
    pan.ghost.classList.toggle('over', !!carry.over(e.clientX, e.clientY));
  }

  function outside(y) {
    const r = el.getBoundingClientRect();
    return y < r.top - CARRY_PX || y > r.bottom + CARRY_PX;
  }

  /** The picture of `tile`, lifted: a copy that follows the pointer (fixed, over everything). */
  function lifted(tile) {
    const ghost = document.createElement('img');
    ghost.className = 'tr-ghost';
    ghost.alt = '';
    ghost.src = tile.querySelector('img')?.src || '';
    document.body.append(ghost);
    return ghost;
  }
  function endPan(e) {
    if (pan?.id !== e.pointerId) return;
    const { moved, ghost, tile } = pan;
    pan = null;
    globalThis.removeEventListener('pointermove', follow, true);
    globalThis.removeEventListener('pointerup', endPan, true);
    globalThis.removeEventListener('pointercancel', endPan, true);
    el.classList.remove('panning');
    if (ghost) {
      ghost.remove();
      const taken = e.type === 'pointerup' && carry.over(e.clientX, e.clientY);
      carry.end?.();
      if (taken) carry.drop(tile);
    }
    if (moved) {
      swallow = true;
      setTimeout(() => { swallow = false; }, 0);   // a release outside any tile makes no click
    }
    if (!inside) release();
  }
  el.addEventListener('click', (e) => {
    if (!swallow) return;
    swallow = false;
    e.preventDefault();
    e.stopPropagation();
  }, true);

  // ---- the wheel scrolls sideways (the row has no other way down) ----------------------------------------
  el.addEventListener('wheel', (e) => {
    if (e.deltaX || el.scrollWidth <= el.clientWidth) return;
    e.preventDefault();
    el.scrollLeft += e.deltaY;
  }, { passive: false });

  const row = {
    removing(tile) {
      if (!inside || !tile) return;
      held = true;
      el.classList.add('held');
      tail += tile.offsetWidth;
      el.style.setProperty('--tail', `${tail}px`);
    },

    reveal(tile, { fromEnd = false, tries = 3 } = {}) {
      if (!tile || held || inside || pan) return;
      if (!el.isConnected || !el.clientWidth) {
        if (tries > 0) globalThis.requestAnimationFrame?.(() => row.reveal(tile, { fromEnd, tries: tries - 1 }));
        return;
      }
      if (!tile.isConnected) return;   // the row was drawn again meanwhile
      if (fromEnd) el.scrollLeft = el.scrollWidth;
      const left = tile.offsetLeft;
      const right = left + tile.offsetWidth;
      if (left >= el.scrollLeft && right <= el.scrollLeft + el.clientWidth) return;
      el.scrollLeft = left - (el.clientWidth - tile.offsetWidth) / 2;
    },
  };
  return row;
}
