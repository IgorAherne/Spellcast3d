// Form controls shared by the stage views and the settings dialog (styled in stages.css).

import { h } from './dom.js';
import { icon } from './icons.js';

export function field(label, control, hint) {
  return h('div', { class: 'field' },
    label ? h('div', { class: 'field-label' }, label) : null,
    control,
    hint ? h('div', { class: 'hint' }, hint) : null);
}

/**
 * A setting on one line: its name, then its control beside it (under the name where the panel is too narrow for
 * both), its hint under them. A panel of these is half as tall as one of fields.
 */
export function rowField(label, control, hint) {
  return h('div', { class: 'field' },
    h('div', { class: 'field-row' }, h('div', { class: 'field-label' }, label), control),
    hint ? h('div', { class: 'hint' }, hint) : null);
}

/**
 * Slider with its value shown next to the label. For a non-linear scale pass toPos/fromPos
 * (value <-> 0..1000 slider position). row: on one line - the name, the slider, the value (rowField's look).
 */
export function sliderField(label, { min, max, step = 1, value, format = String, onInput, toPos, fromPos, hint, row = false }) {
  const mapped = !!toPos;
  const range = h('input', {
    type: 'range', class: 'range', 'aria-label': label,
    min: mapped ? 0 : min, max: mapped ? 1000 : max, step: mapped ? 1 : step,
  });
  const readout = h('span', { class: 'field-value' });
  const paint = () => {
    const pct = ((+range.value - +range.min) / (+range.max - +range.min)) * 100;
    range.style.setProperty('--fill', `${pct}%`);
  };
  // What screen readers announce: the value as shown, not the slider position (0..1000 when mapped).
  const show = (v) => {
    readout.textContent = format(v);
    range.setAttribute('aria-valuetext', format(v));
    paint();
  };
  const set = (v) => {
    range.value = mapped ? toPos(v) : v;
    show(v);
  };
  range.addEventListener('input', () => {
    const v = mapped ? fromPos(+range.value) : +range.value;
    show(v);
    onInput(v);
  });
  set(value);
  const el = row ? rowField(label, [range, readout], hint) : h('div', { class: 'field' },
    h('div', { class: 'field-label' }, h('span', null, label), readout),
    range,
    hint ? h('div', { class: 'hint' }, hint) : null);
  el.set = set;
  el.input = range;
  return el;
}

/** Row of mutually exclusive buttons. options: [{value, label, title?}] */
export function segmented(options, value, onChange) {
  const el = h('div', { class: 'seg', role: 'radiogroup' });
  const buttons = options.map((o) => h('button', {
    type: 'button', class: 'seg-btn', role: 'radio', title: o.title,
    onclick: () => { set(o.value); onChange(o.value); },
  }, o.label));
  el.append(...buttons);
  function set(v) {
    buttons.forEach((b, i) => {
      const on = options[i].value === v;
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', String(on));
    });
  }
  set(value);
  el.set = set;
  el.buttons = buttons;
  return el;
}

export function toggle(checked, onChange, text, hint) {
  const input = h('input', { type: 'checkbox', checked: !!checked });
  input.addEventListener('change', () => onChange(input.checked));
  const el = h('label', { class: 'switch' },
    input,
    h('span', { class: 'switch-track' }, h('span', { class: 'switch-thumb' })),
    h('span', { class: 'switch-text' },
      h('span', { class: 'switch-title' }, text),
      hint ? h('span', { class: 'hint' }, hint) : null));
  el.input = input;
  return el;
}

/**
 * Seed box: empty means random (-1). The dice switches back to random. max: the largest seed the tool takes
 * (a bigger number typed in becomes it).
 */
export function seedInput(value, onChange, { randomMeansMinusOne = true, max = Number.MAX_SAFE_INTEGER } = {}) {
  const input = h('input', {
    class: 'input', type: 'number', min: 0, max, step: 1, inputmode: 'numeric',
    placeholder: randomMeansMinusOne ? 'Random' : '0',
    value: value != null && value >= 0 ? String(value) : '',
  });
  input.addEventListener('change', () => {
    const t = input.value.trim();
    const n = parseInt(t, 10);
    if (t === '' || Number.isNaN(n)) {
      input.value = randomMeansMinusOne ? '' : '0';
      onChange(randomMeansMinusOne ? -1 : 0);
    } else {
      const seed = Math.min(max, Math.max(0, n));
      input.value = String(seed);
      onChange(seed);
    }
  });
  const dice = h('button', {
    type: 'button', class: 'icon-btn', title: randomMeansMinusOne ? 'Random seed every time' : 'Roll a new random seed',
    onclick: () => {
      if (randomMeansMinusOne) {
        input.value = '';
        onChange(-1);
      } else {
        const n = Math.floor(Math.random() * 2 ** 31);
        input.value = String(n);
        onChange(n);
      }
      dice.animate([{ transform: 'rotate(0)' }, { transform: 'rotate(180deg)' }], { duration: 300, easing: 'ease-out' });
    },
  }, icon('dice'));
  return h('div', { class: 'seed' }, input, dice);
}

export function progressBar(fraction) {
  const fill = h('span', { class: 'progress-fill' });
  const el = h('div', { class: 'progress' }, fill);
  el.set = (f) => {
    el.classList.toggle('indeterminate', f == null);
    fill.style.width = f == null ? '' : `${Math.round(Math.min(1, Math.max(0, f)) * 100)}%`;
  };
  el.set(fraction);
  return el;
}

export const spinner = () => h('span', { class: 'spinner', 'aria-hidden': 'true' });
