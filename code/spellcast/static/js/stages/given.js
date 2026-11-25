// What Qwen-Image was given for one of a prop's pictures and what came back (Alt + click on the picture's tile in the
// Images stage's row: images.js), in a window laid out as Omni-Texture's of a painted view (omnitexture.js showInputs,
// the same look: stages-models.css .ot-inputs): the words as it got them, the pictures in the order it got them, apart
// from them what it never sees as a picture (what it starts from or is held to, where it may draw), and its answer.
// The hub keeps the record beside the picture: for images/0012.png, images/.given/0012/given.json and the files it
// names. A picture made before such records were kept, an imported one and one of Forge Neo's have none.

import { field } from '../controls.js';
import { h } from '../dom.js';
import { projectId } from '../store.js';
import { modal, toast, toastError } from '../ui.js';
import { boxFile, figure } from './common.js';

const NO_RECORD = 'There is no record of what Qwen-Image was given for this picture: it was made before such records '
  + 'were kept, imported, or not made by Qwen-Image.';

// The models as the Images stage names them (images.js QWEN_MODELS; given.json "model").
const MODELS = { turbo: 'Qwen-Image 2.1 Turbo', full: 'Qwen-Image 2.1', flux3: 'FLUX.3 Image Klein' };

// How the picture was drawn (given.json "way").
const WAYS = {
  words: 'drawn from the words alone',
  examples: 'drawn from the words and the example pictures',
  start: 'started from the viewport’s picture',
  whole: 'a masked change: all of the picture drawn anew from the brushed one, then brought to its tone and its place',
  window: 'a masked change, drawn in the window and laid back into the picture',
  held: 'a masked change, drawn in the whole picture and held to it outside what you brushed',
};

// What a picture given is (given.json pictures[].is), after its number.
const IS = {
  window: ': the part of the picture it works on, what you brushed in magenta',
  brushed: ': the picture, what you brushed in magenta',
  start: ', an example and also what the drawing starts from',
  example: ': an example picture',
};

// What the way did to the model's own picture before it was saved (given.json answers).
const BEFORE = {
  window: 'before it was laid into the picture',
  whole: 'as it came back, before it was brought to your picture’s tone and - a large one, which comes back some '
    + 'pixels off yours and stretched a little - onto your picture',
  held: 'before it was held to the picture outside what you brushed',
};

const px = ([w, hh]) => `${w} × ${hh} px`;
const sized = (size) => Array.isArray(size) && size.length === 2;

/** The folder of the record of the picture `path`: "images/0012.png" -> "images/.given/0012". */
export function givenFolder(path) {
  const cut = path.lastIndexOf('/') + 1;
  return `${path.slice(0, cut)}.given/${path.slice(cut).replace(/\.[^.]+$/, '')}`;
}

/** The way of the record, and of a masked change drawn whole what was kept of it (given.json "patch": its box). */
function wayOf(record) {
  const patch = record.way === 'whole' && Array.isArray(record.patch) && record.patch;
  if (!patch) return WAYS[record.way];
  const part = px([patch[2] - patch[0], patch[3] - patch[1]]);
  return `${WAYS.whole}; a ${part} part of that, round what you brushed, laid into the picture`;
}

/** The record's hint line: the model, the seeds, the steps, the size asked of it and the way. */
function hintOf(record) {
  const size = !sized(record.size) ? null : record.way === 'window' && sized(record.picture)
    ? `a ${px(record.size)} window of a ${px(record.picture)} picture` : px(record.size);
  return [
    MODELS[record.model] || 'Qwen-Image 2.1',
    record.seed != null ? `seed ${record.seed}` : null,
    record.noise_seed != null ? `noise seed ${record.noise_seed}` : null,
    record.steps ? `${record.steps} steps` : null,
    size,
    wayOf(record),
  ].filter(Boolean).join(' · ');
}

/** What the drawing starts from (or is held to) and where it may draw: none of them a picture the model sees. */
function apartFrom(record, file) {
  const held = record.way === 'held' || record.way === 'window';
  const strength = typeof record.strength === 'number'
    ? `, strength ${record.strength}${record.strength < 1 ? ': below 1 keeps more of it' : ''}` : '';
  return [
    record.start && figure(file(record.start), `Not a picture it sees: ${held
      ? 'what its drawing is held to, outside where it may draw' : `what the drawing starts from${strength}`}`),
    record.mask && figure(file(record.mask), 'Not a picture it sees: where it may draw (white)'),
  ].filter(Boolean);
}

/**
 * What Qwen-Image was given for the picture `path` of `prop` (as images.candidates names it) and what came back, in
 * a window; a toast where the picture has no record.
 */
export async function showGiven(prop, path) {
  const folder = givenFolder(path);
  const file = (name) => boxFile(prop, `${folder}/${name}`, 'images');
  let record;
  try {
    const res = await fetch(file('given.json'));
    if (res.status === 404) {
      toast(NO_RECORD, 'info');
      return;
    }
    if (!res.ok) throw new Error(`The record of this picture could not be read (error ${res.status}).`);
    record = await res.json();
  } catch (e) {
    toastError(e);
    return;
  }
  const engine = record.model === 'flux3' ? MODELS.flux3 : 'Qwen-Image';
  const given = (Array.isArray(record.pictures) ? record.pictures : []).map((pic, i) => figure(file(pic.file),
    `Picture ${i + 1}${IS[pic.is] || ''}${sized(pic.seen) ? ` (${engine} sees it at ${px(pic.seen)})` : ''}`));
  const apart = apartFrom(record, file);
  const saved = boxFile(prop, path, 'images');
  const answers = Array.isArray(record.answers) ? record.answers : [];
  const own = answers[Number.isInteger(record.answer) ? record.answer : 0];
  const back = own
    ? [figure(file(own), `${engine}’s own picture, ${BEFORE[record.way] || 'before anything was kept or laid back'}`),
      figure(saved, 'The picture as it was saved')]
    : [figure(saved, `${engine}’s picture, as it was saved`)];
  const grid = (figures) => h('div', { class: 'ot-inputs-grid' }, figures);
  const words = (text) => h('p', { class: 'ot-words' }, text || '');
  modal({
    title: `What ${engine} was given for this picture`,
    wide: true,
    className: 'ot-inputs',
    body: h('div', { class: 'stack' },
      field('Words', words(record.words)),
      h('div', { class: 'hint' }, hintOf(record)),
      field('Pictures given, in this order', given.length ? grid(given) : h('div', { class: 'hint' },
        record.start ? 'None: it drew from the words and what it starts from, below.' : 'None: it drew from the words alone.')),
      apart.length ? field('Not given to it as pictures', grid(apart)) : null,
      field('What came back', grid(back)),
      h('div', { class: 'hint' }, `The files: projects/${projectId()}/props/${prop.id}/${folder}`)),
    actions: [{ label: 'Close', value: true }],
  });
}
