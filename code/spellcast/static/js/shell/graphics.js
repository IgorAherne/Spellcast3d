// How this browser draws the 3D view, as far as the artist chooses (Settings > Advanced): kept in this browser, as
// the animations and the tooltips are.
//
// "Faster 3D view" draws the model without anti-aliasing: its edges are a little jagged, and the graphics chip that
// draws the page has about a third less to do per picture and holds less memory - what a processor's built-in
// graphics behind a high-density screen needs. Off unless chosen.

const PREF_KEY = 'spellcast3d.plainEdges';

let plain = null;

function readPref() {
  try {
    return localStorage.getItem(PREF_KEY) === 'on';
  } catch {
    return false;
  }
}

/** The 3D view is drawn without anti-aliasing. */
export const plainEdges = () => (plain ??= readPref());

/**
 * Keeps the choice (for this browser; for this page only when the browser keeps nothing). A 3D view follows it
 * from the next page load on: its canvas is made with anti-aliasing or without.
 */
export function setPlainEdges(on) {
  plain = !!on;
  try {
    if (plain) localStorage.setItem(PREF_KEY, 'on');
    else localStorage.removeItem(PREF_KEY);
  } catch { /* this page keeps it */ }
}
