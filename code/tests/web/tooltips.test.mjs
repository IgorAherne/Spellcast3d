// The tooltips' switch of the web UI (static/js/shell/tooltips.js): shown by default; turned off in
// Settings, every title leaves its element as it appears and comes back when they are turned on again, the
// shell's quick tips stay away, and what says that something went wrong still shows.
//   node --test code/tests/web/tooltips.test.mjs
// (code/tests/test_web.py runs every code/tests/web/*.test.mjs when Node.js is installed)
import assert from 'node:assert/strict';
import { test } from 'node:test';

// What a browser page has, as far as tooltips.js uses it: elements with attributes and children, the page's
// watcher (its changes are handed over by the tests, as a browser does after the page's code ran) and a store.
class El {
  constructor(attrs = {}, children = [], textContent = '') {
    this.nodeType = 1;
    this.attrs = new Map(Object.entries(attrs));
    this.children = children;
    this.textContent = textContent;
  }
  getAttribute(k) { return this.attrs.has(k) ? this.attrs.get(k) : null; }
  hasAttribute(k) { return this.attrs.has(k); }
  setAttribute(k, v) { this.attrs.set(k, String(v)); }
  removeAttribute(k) { this.attrs.delete(k); }
  querySelectorAll(selector) {
    const attr = selector.slice(1, -1);
    const found = [];
    const walk = (el) => el.children.forEach((c) => { if (c.attrs.has(attr)) found.push(c); walk(c); });
    walk(this);
    return found;
  }
}
const text = { nodeType: 3 };
let watchers = [];
globalThis.MutationObserver = class {
  constructor(callback) { this.callback = callback; }
  observe(target, options) { watchers.push({ callback: this.callback, target, options }); }
};
const saved = new Map();
let storageWorks = true;
const storage = {
  getItem: (k) => { if (!storageWorks) throw new Error('blocked'); return saved.get(k) ?? null; },
  setItem: (k, v) => { if (!storageWorks) throw new Error('blocked'); saved.set(k, String(v)); },
  removeItem: (k) => { if (!storageWorks) throw new Error('blocked'); saved.delete(k); },
};
Object.defineProperty(globalThis, 'localStorage', { value: storage, configurable: true });

const TOOLTIPS = new URL('../../spellcast/static/js/shell/tooltips.js', import.meta.url);
let loads = 0;
/** A page loading afresh: its elements, and tooltips.js as it finds it (it reads the stored choice once). */
async function freshPage() {
  watchers = [];
  const gear = new El({ title: 'Settings', 'aria-label': 'Settings' });
  const plain = new El({ class: 'btn' });
  const render = new El({ title: 'Draws the picture' }, [new El({ title: 'Its icon' })], 'Render');
  const html = new El({}, [new El({}, [gear, plain]), render]);
  globalThis.document = { documentElement: html };
  const module = await import(`${TOOLTIPS.href}?page=${++loads}`);
  /** The browser telling the page's watcher what the page's code just did. */
  const tell = (changes) => watchers.forEach((w) => w.callback(changes));
  return { ...module, html, gear, plain, render, icon: render.children[0], tell };
}
const titled = (el) => [el, ...el.querySelectorAll('[title]')].filter((e) => e.hasAttribute('title')).length;

test('tooltips show by default: nothing is stored, and a short title is left as it is', async () => {
  saved.clear();
  const page = await freshPage();
  page.initTooltips();
  assert.equal(page.tooltipsOn(), true);
  assert.equal(titled(page.html), 3);
  assert.deepEqual([page.gear, page.render, page.icon].map((el) => el.getAttribute('title')), ['Settings', 'Draws the picture', 'Its icon']);
  assert.equal(saved.size, 0);
  assert.ok(['', 'note', 'refusal', 'problem'].every(page.tipShows));
});

test('a long title is broken into lines of about 60 characters, as it appears and whenever the page changes it', async () => {
  saved.clear();
  const page = await freshPage();
  const long = 'Inverse perspective: bends the view round the model, so that what turns away from you - the cheeks of a face - shows';
  const lines = ['Inverse perspective: bends the view round the model, so that', 'what turns away from you - the cheeks of a face - shows'];
  assert.equal(page.wrapped(long), lines.join('\n'));
  assert.ok(lines.every((line) => line.length <= 60));
  assert.equal(page.wrapped(page.wrapped(long)), page.wrapped(long), 'lines already made stay');
  assert.equal(page.wrapped('one\nsecond line of its own'), 'one\nsecond line of its own', 'the page\'s own line breaks are kept');
  assert.equal(page.wrapped(`${'x'.repeat(70)} y`), `${'x'.repeat(70)}\ny`, 'a word longer than a line keeps its line');
  // On the page: what is there at the start, what is drawn later, a title the page sets anew.
  page.gear.setAttribute('title', long);
  page.initTooltips();
  assert.equal(page.gear.getAttribute('title'), lines.join('\n'));
  const drawn = new El({}, [new El({ title: long })]);
  page.html.children.push(drawn);
  page.tell([{ type: 'childList', addedNodes: [drawn] }]);
  assert.equal(drawn.children[0].getAttribute('title'), lines.join('\n'));
  page.render.setAttribute('title', long);
  page.tell([{ type: 'attributes', target: page.render }]);
  assert.equal(page.render.getAttribute('title'), lines.join('\n'));
  // Turned off and on again, the words come back in their lines.
  page.setTooltips(false);
  assert.equal(page.render.hasAttribute('title'), false);
  page.setTooltips(true);
  assert.equal(page.render.getAttribute('title'), lines.join('\n'));
});

test('turned off, every title leaves its element, and this browser remembers', async () => {
  saved.clear();
  const page = await freshPage();
  page.initTooltips();
  page.setTooltips(false);
  assert.equal(page.tooltipsOn(), false);
  assert.equal(saved.get('spellcast3d.tooltips'), 'off');
  assert.equal(titled(page.html), 0, 'the browser has no title to show');
  assert.equal(page.gear.getAttribute('data-title-off'), 'Settings', 'the words are kept for later');
  assert.equal(page.gear.getAttribute('aria-label'), 'Settings', 'a name for screen readers is not a tooltip');
  assert.equal(page.icon.getAttribute('data-title-off'), 'Its icon');
  assert.equal(page.plain.hasAttribute('data-title-off'), false);
  const watch = watchers[0].options;
  assert.deepEqual([watch.subtree, watch.childList, watch.attributes, watch.attributeFilter],
    [true, true, true, ['title']]);
  assert.equal(watchers[0].target, page.html);

  const next = await freshPage();
  next.initTooltips();
  assert.equal(next.tooltipsOn(), false, 'the next page load keeps it');
  assert.equal(titled(next.html), 0);
});

test('while they are off, what the page draws or retitles later loses its title too', async () => {
  saved.clear();
  const page = await freshPage();
  page.setTooltips(false);
  // A button gets other words (its status changed), a new part of the page is drawn, a title is emptied.
  page.gear.setAttribute('title', 'Settings\nA model folder is missing');
  page.tell([{ type: 'attributes', target: page.gear }]);
  assert.equal(page.gear.hasAttribute('title'), false);
  assert.equal(page.gear.getAttribute('data-title-off'), 'Settings\nA model folder is missing');
  const drawn = new El({ title: 'A panel' }, [new El({ title: 'A slider' }), new El()]);
  page.html.children.push(drawn);
  page.tell([{ type: 'childList', addedNodes: [text, drawn] }]);
  assert.equal(titled(page.html), 0);
  assert.equal(drawn.children[0].getAttribute('data-title-off'), 'A slider');
  page.render.setAttribute('title', '');
  page.tell([{ type: 'attributes', target: page.render }]);
  assert.equal(page.render.hasAttribute('title'), false);
  assert.equal(page.render.hasAttribute('data-title-off'), false, 'no words are kept for a title the page took away');
  page.tell([{ type: 'attributes', target: page.plain }]);     // (the watcher hears its own removals too)
  assert.equal(page.plain.attrs.size, 1);
});

test('turned on again, the titles are back, also on a part of the page that was set aside meanwhile', async () => {
  saved.clear();
  const page = await freshPage();
  page.setTooltips(false);
  const aside = page.html.children.pop();      // the Render button's part leaves the page while they are off
  assert.equal(aside, page.render);
  page.setTooltips(true);
  assert.equal(saved.has('spellcast3d.tooltips'), false, 'shown is the default: nothing is stored');
  assert.equal(page.gear.getAttribute('title'), 'Settings');
  assert.equal(page.gear.hasAttribute('data-title-off'), false);
  assert.equal(page.render.hasAttribute('title'), false, 'not in the page: not reached yet');
  page.html.children.push(aside);
  page.tell([{ type: 'childList', addedNodes: [aside] }]);
  assert.equal(page.render.getAttribute('title'), 'Draws the picture');
  assert.equal(page.icon.getAttribute('title'), 'Its icon');
  // Words the page gave an element after they were turned on are not overwritten by the older ones.
  const stale = new El({ title: 'New words', 'data-title-off': 'Old words' });
  page.tell([{ type: 'childList', addedNodes: [stale] }]);
  assert.equal(stale.getAttribute('title'), 'New words');
  assert.equal(stale.hasAttribute('data-title-off'), false);
  page.gear.setAttribute('title', 'Settings again');
  page.tell([{ type: 'attributes', target: page.gear }]);
  assert.equal(page.gear.getAttribute('title'), 'Settings again', 'titles are left alone while they show');
});

test('with tooltips off the quick tips stay away, but a refusal and a problem\'s words still show', async () => {
  saved.clear();
  const page = await freshPage();
  page.setTooltips(false);
  assert.deepEqual(['', 'note', 'refusal', 'problem'].map(page.tipShows), [false, false, true, true]);
  page.setTooltips(true);
  assert.deepEqual(['', 'note', 'refusal', 'problem'].map(page.tipShows), [true, true, true, true]);
});

test('a browser that keeps nothing still gets the choice for this page', async () => {
  saved.clear();
  storageWorks = false;
  try {
    const page = await freshPage();
    page.initTooltips();
    assert.equal(page.tooltipsOn(), true);
    page.setTooltips(false);
    assert.equal(page.tooltipsOn(), false);
    assert.equal(titled(page.html), 0);
  } finally {
    storageWorks = true;
  }
});

test('a button of an icon alone keeps a name while its title is away, and gives it back with the title', async () => {
  saved.clear();
  const page = await freshPage();
  const close = new El({ title: 'Close' });                 // an icon, no words, no aria-label
  page.html.children.push(close);
  page.setTooltips(false);
  assert.deepEqual([close.hasAttribute('title'), close.getAttribute('aria-label')], [false, 'Close']);
  assert.equal(page.render.hasAttribute('aria-label'), false, 'a button with words of its own needs none');
  assert.equal(page.gear.getAttribute('aria-label'), 'Settings');
  // Its title changes meanwhile: the name follows; emptied, the name goes too.
  close.setAttribute('title', 'Close the window');
  page.tell([{ type: 'attributes', target: close }]);
  assert.deepEqual([close.hasAttribute('title'), close.getAttribute('aria-label')], [false, 'Close the window']);
  page.setTooltips(true);
  assert.deepEqual([close.getAttribute('title'), close.hasAttribute('aria-label')], ['Close the window', false]);
  assert.equal(page.gear.getAttribute('aria-label'), 'Settings', 'a name of its own is never taken');
  assert.equal(close.attrs.size, 1, 'nothing of the switch is left on it');
});
