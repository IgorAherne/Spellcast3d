// Tiny DOM helpers: h() builds elements, the rest are small shared utilities.

const PROPS = new Set(['value', 'checked', 'disabled', 'selected', 'hidden', 'multiple', 'readOnly', 'tabIndex']);

/** h('button', {class: 'btn', onclick: fn, disabled: true}, 'Label', childNode, [more]) */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'style') el.style.cssText = v;
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else if (PROPS.has(k)) el[k] = v;
      else el.setAttribute(k, v === true ? '' : String(v));
    }
  }
  return add(el, children);
}

export function add(el, children) {
  for (const c of [children].flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : String(c));
  }
  return el;
}

/** True while focus is in a field that uses arrow keys itself (text, slider, dropdown). */
export function isTyping() {
  const el = document.activeElement;
  if (!el) return false;
  if (el.isContentEditable) return true;
  if (el.tagName === 'TEXTAREA' || el.tagName === 'SELECT') return true;
  if (el.tagName === 'INPUT') return !['checkbox', 'radio', 'button', 'submit'].includes(el.type);
  return false;
}

/** True while focus is in a field the artist types text into (not a slider, a dropdown or a colour field). */
export function isTextField() {
  const el = document.activeElement;
  if (!el) return false;
  if (el.isContentEditable || el.tagName === 'TEXTAREA') return true;
  return el.tagName === 'INPUT' && !['checkbox', 'radio', 'button', 'submit', 'range', 'color', 'file'].includes(el.type);
}

export function plural(n, one, many = one + 's') {
  return `${n} ${n === 1 ? one : many}`;
}

export function formatCount(n) {
  if (n >= 1e6) return `${+(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${+(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}k`;
  return String(n);
}

/** 72_000 -> "1:12", 3_725_000 -> "1:02:05" */
export function formatElapsed(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const mm = String(Math.floor(s / 60) % 60).padStart(s >= 3600 ? 2 : 1, '0');
  const ss = String(s % 60).padStart(2, '0');
  return s >= 3600 ? `${Math.floor(s / 3600)}:${mm}:${ss}` : `${mm}:${ss}`;
}

/** 1_288_490_188 -> "1.2 GB" */
export function formatBytes(n) {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  if (n >= 1024 ** 2) return `${Math.round(n / 1024 ** 2)} MB`;
  return `${Math.max(1, Math.round(n / 1024))} KB`;
}

/** An ISO time as "just now", "5 min ago", "3 h ago", "yesterday" or a date. */
export function timeAgo(iso) {
  const t = Date.parse(iso || '');
  if (Number.isNaN(t)) return '';
  const min = Math.round((Date.now() - t) / 60000);
  if (min < 1) return 'just now';
  if (min < 60) return `${min} min ago`;
  if (min < 24 * 60) return `${Math.round(min / 60)} h ago`;
  if (min < 48 * 60) return 'yesterday';
  return new Date(t).toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
}

/** Resolves with the chosen File, or null when the user closes the picker. */
export async function pickFile(accept = 'image/*') {
  return (await pickFiles(accept, { multiple: false }))[0] || null;
}

/** Resolves with the Files chosen (several at once unless multiple is false), or [] when the user closes the picker. */
export function pickFiles(accept = 'image/*', { multiple = true } = {}) {
  return new Promise((resolve) => {
    const input = h('input', { type: 'file', accept, multiple, style: 'display:none' });
    input.addEventListener('change', () => { resolve([...input.files]); input.remove(); });
    input.addEventListener('cancel', () => { resolve([]); input.remove(); });
    document.body.append(input);
    input.click();
  });
}

/**
 * Makes `el` accept dropped image files. onFile(file) gets the first image dropped, or, with
 * {all: true}, onFile(files) gets every image dropped, in the order they came;
 * onReject() is called when the drop contained no image.
 */
export function dropTarget(el, onFile, onReject, { all = false } = {}) {
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
  el.addEventListener('dragenter', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth++;
    el.classList.add('drag-over');
  });
  el.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
  el.addEventListener('dragleave', () => { if (--depth <= 0) { depth = 0; el.classList.remove('drag-over'); } });
  el.addEventListener('drop', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.stopPropagation();
    depth = 0;
    el.classList.remove('drag-over');
    const images = [...e.dataTransfer.files].filter((f) => f.type.startsWith('image/'));
    if (!images.length) onReject?.();
    else onFile(all ? images : images[0]);
  });
}
