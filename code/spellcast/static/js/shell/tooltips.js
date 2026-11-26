// The small explanations that appear when the pointer rests on something: the browser's own tooltip of an
// element's title, and the shell's quick ones (popups.js showTip). An artist who knows the page turns them off
// in Settings > Advanced > "Show tooltips"; the choice is kept in this browser, like the animations'.
//
// A browser shows a title whatever the page wishes. So while they are off every title is taken off its element
// as it appears and kept in data-title-off, and put back when they are turned on again. Words that say
// something went wrong (why a button said no, what failed for a prop) are no tooltip: they always show.
// A button of an icon alone is named by its title: meanwhile the words are its aria-label, for a screen reader.
// A browser also shows a long title as one strip across the screen: every title is broken into lines of about
// WRAP characters as it appears (the shell's quick ones wrap by their own width: shell.css .tip-pop).

const PREF_KEY = 'spellcast3d.tooltips';
const KEPT = 'data-title-off';
const NAMED = 'data-named-by-title';
const ALWAYS = new Set(['refusal', 'problem']);     // kinds of showTip that are answers, not explanations
const WRAP = 60;

let on = null;
let watcher = null;

/** `words` in lines of WRAP characters at the most, broken at spaces (a longer word keeps its line; lines stay). */
export function wrapped(words) {
  return words.split('\n').map((line) => {
    const rows = [];
    for (const word of line.split(' ')) {
      const row = rows.at(-1);
      if (row === undefined || row.length + 1 + word.length > WRAP) rows.push(word);
      else rows[rows.length - 1] = `${row} ${word}`;
    }
    return rows.join('\n');
  }).join('\n');
}

/** The title of `el` in lines (set only when that changes it: the watcher hears of every change). */
function wrap(el) {
  const words = el.getAttribute('title');
  if (words && wrapped(words) !== words) el.setAttribute('title', wrapped(words));
}

// Only "off" is stored.
function readPref() {
  try {
    return localStorage.getItem(PREF_KEY) !== 'off';
  } catch {
    return true;
  }
}

/** Do tooltips show: the viewer's choice, yes by default. */
export const tooltipsOn = () => (on ??= readPref());

/** Does a quick tip of this kind show now (showTip's kinds; a refusal and a problem's words always do). */
export const tipShows = (kind) => tooltipsOn() || ALWAYS.has(kind);

/** The title off `el`, its words kept; a title the page emptied empties what was kept. */
function takeOff(el) {
  const words = el.getAttribute('title');
  if (words == null) return;
  el.removeAttribute('title');
  if (words) el.setAttribute(KEPT, words);
  else el.removeAttribute(KEPT);
  if (el.hasAttribute(NAMED) || !(el.hasAttribute('aria-label') || (el.textContent || '').trim())) name(el, words);
}

/** `words` as the name of an element that has no other (none: that name taken away again). */
function name(el, words) {
  for (const attr of ['aria-label', NAMED]) {
    if (words) el.setAttribute(attr, attr === NAMED ? '' : words);
    else el.removeAttribute(attr);
  }
}

function putBack(el) {
  const words = el.getAttribute(KEPT);
  if (words == null) return;
  el.removeAttribute(KEPT);
  if (el.hasAttribute(NAMED)) name(el, '');
  if (!el.hasAttribute('title')) el.setAttribute('title', words);
}

/** `root` and everything in it, brought to the choice: its titles shown, in lines, or taken off. */
function sweep(root) {
  if (root.nodeType !== 1) return;
  const each = (attr, act) => {
    if (root.hasAttribute(attr)) act(root);
    for (const el of root.querySelectorAll(`[${attr}]`)) act(el);
  };
  if (tooltipsOn()) {
    each(KEPT, putBack);
    each('title', wrap);
  } else {
    each('title', takeOff);
  }
}

/** Whatever enters the page or gets a title from now on follows the choice. */
function watch() {
  const observer = new MutationObserver((changes) => {
    for (const change of changes) {
      if (change.type === 'childList') change.addedNodes.forEach(sweep);
      else if (tooltipsOn()) wrap(change.target);
      else takeOff(change.target);
    }
  });
  observer.observe(document.documentElement, { subtree: true, childList: true, attributes: true, attributeFilter: ['title'] });
  return observer;
}

function apply() {
  watcher ||= watch();
  sweep(document.documentElement);
}

/** Keeps the viewer's choice (for this browser; for this page only when the browser keeps nothing). */
export function setTooltips(show) {
  on = !!show;
  try {
    if (on) localStorage.removeItem(PREF_KEY);
    else localStorage.setItem(PREF_KEY, 'off');
  } catch { /* this page keeps it */ }
  apply();
}

/** Brings the page to the stored choice. Call once, at start. */
export function initTooltips() {
  apply();
}
