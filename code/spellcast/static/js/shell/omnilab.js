// Settings > Omni-Texture test: a 3D model loaded straight from a file is OmniTextured, without Images and 3D
// first, for working on the Omni-Texture stage itself. The model becomes a new prop of the open project (its pictures,
// when given, the pictures Qwen-Image 2.1 is shown), chosen for Omni-Texture; its job starts with the number of
// cameras and the words set here, and the page goes to the Omni-Texture stage, which shows the result.

import { h, plural } from '../dom.js';
import { icon } from '../icons.js';
import { emit, projectId, store } from '../store.js';
import { queueSettings, startJob, uploadImages } from '../actions.js';
import { field, segmented, spinner } from '../controls.js';
import { toastError } from '../ui.js';
import { NO_MODEL, importModelBox, modelDropTarget, pickModelFiles, usableModelFiles } from '../stages/modelimport.js';
import { importSides } from '../stages/slicer.js';

const MAX_CAMERAS = 8;       // the hub's omnitexture.MAX_VIEWS
const CAMERAS = 2;           // its VIEW_COUNT: what the stage itself paints from
const IMAGE_TYPES = 'image/png,image/jpeg,image/webp';
// What each view's first picture shows where it is to be painted (the hub's omnitexture.SHOWN; the stage: depth).
const SHOWN = [
  { value: 'depth', label: 'Depth map', title: 'The model’s depth from that camera: white near, almost black far' },
  { value: 'clay', label: 'Lit model', title: 'The model in plain grey, lit from slightly to the side, no shadows' },
  { value: 'normal', label: 'Normal map', title: 'Which way each bit of the surface faces, as colours: small shapes '
    + 'show far more clearly than in the depth map. The words then say “normal map”, and told so Qwen-Image has '
    + 'painted a normal map back so far' },
  { value: 'texture', label: 'Textured model', title: 'The model with the texture the 3D stage gave it' },
];
// How the views are painted (the hub's job params "together" and "sheet_side"): together in pictures up to
// 2048 px, as many views in each as keep their size alone (the stage), in smaller pictures that hold fewer, or a
// picture for every view.
const PAINTED = [
  { value: '', label: 'Together', title: 'In pictures up to 2048 × 2048 (no larger than the graphics card draws), '
    + 'as many views in each as fit at their full size: usually four or five (the stage does this)' },
  ...[1536, 1024].map((px) => ({
    value: px, label: `Up to ${px} px`, title: `In pictures up to ${px} × ${px}: fewer views in one, each followed `
      + 'more closely',
  })),
  { value: 'alone', label: 'One by one', title: 'A picture of 1024 × 1024 for every view' },
];
// The second picture, the example (the hub's job param "example": absent = the prop's own picture of its front,
// as the stage; "model" = the model from that view with the texture the 3D stage gave it, the stage's until
// 2026-11-23).
const EXAMPLE = [
  { value: '', label: 'Its front picture', title: 'The prop’s own picture of its front, the same for every view '
    + '(another side’s when it has no front; the textured model when it has no pictures): the stage does this' },
  { value: 'model', label: 'Textured model', title: 'The model from that view with the texture the 3D stage gave '
    + 'it: Qwen-Image then mostly copies that texture' },
];
// Whether the prop's other pictures are sent after the example, and how (the hub's job param "pictures":
// absent = not sent, as the stage; "full"; or the longest side they are blurred to, omnitexture.PICTURE_SIZES).
const SENT = [
  { value: '', label: 'Not sent' },
  { value: 'full', label: 'As they are' },
  ...[512, 256, 128].map((px) => ({ value: px, label: `${px} px`, title: `Shrunk to ${px} px, then enlarged back` })),
];

/** Resolves with the pictures chosen ([] when the picker is closed). */
function pickPictures() {
  return new Promise((resolve) => {
    const input = h('input', { type: 'file', multiple: true, accept: IMAGE_TYPES, style: 'display:none' });
    const done = (files) => { resolve(files); input.remove(); };
    input.addEventListener('change', () => done([...input.files]));
    input.addEventListener('cancel', () => done([]));
    document.body.append(input);
    input.click();
  });
}

/** The section's content; close(): closes the settings once the job has started. */
export function omniTextureLab({ close }) {
  let model = [];        // [{file, path}]: the model and whatever came with it
  let pictures = [];     // File[]
  let busy = false;
  let show = SHOWN[0].value;
  let painted = PAINTED[0].value;
  let example = EXAMPLE[0].value;
  let sent = SENT[0].value;

  const modelName = h('span', { class: 'hint' });
  const pictureNames = h('span', { class: 'hint' });
  const cameras = h('input', { class: 'input lab-number', type: 'number', min: 1, max: MAX_CAMERAS, step: 1, value: CAMERAS });
  const words = h('textarea', {
    class: 'input', rows: 3, spellcheck: 'false',
    placeholder: 'Empty: the stage’s own words. {view} becomes “viewed from the front”, “… the back” or “… the side”, '
      + '{map} what the first picture is (“depthmap”, “normal map”, …).',
  });
  const status = h('span', { class: 'hint lab-status' });
  const glyph = h('span', { class: 'ot-glyph' }, icon('sparkles'));
  const go = h('button', { type: 'button', class: 'btn primary', onclick: () => run() }, glyph, 'Omni-Texture it');

  const sync = () => {
    modelName.textContent = model.length ? model.map((f) => f.path).join(', ') : 'None chosen yet (.glb, .fbx, .obj…)';
    pictureNames.textContent = pictures.length ? pictures.map((f) => f.name).join(', ')
      : 'Optional: the pictures the model was made from (front, back, side)';
    go.disabled = busy || !model.length || !store.project;
    glyph.replaceChildren(busy ? spinner() : icon('sparkles'));
  };

  /** files: [{file, path}], chosen or dropped. */
  function takeModel(files) {
    if (!files.length || busy) return;
    const usable = usableModelFiles(files);
    if (!usable) {
      toastError(new Error(NO_MODEL));
      return;
    }
    model = usable;
    sync();
  }

  function takePictures(files) {
    const images = files.filter((f) => f.type.startsWith('image/'));
    if (images.length && !busy) pictures = images;
    sync();
  }

  async function run() {
    if (busy || !model.length) return;
    busy = true;
    sync();
    const pid = projectId();
    try {
      status.textContent = 'Reading the 3D model…';
      const prop = await importModelBox(model, (f) => {
        status.textContent = f != null && f < 1 ? `Uploading the 3D model… ${Math.round(f * 100)}%` : 'Reading the 3D model…';
      });
      if (projectId() !== pid) return;
      if (pictures.length) {
        status.textContent = `Adding ${plural(pictures.length, 'picture')}…`;
        const plan = importSides(pictures.map((f) => f.name || ''));
        await uploadImages(prop.id, plan.map((x) => pictures[x.index]), null, plan.map((x) => x.side));
      }
      status.textContent = 'Starting Omni-Texture…';
      queueSettings(prop.id, 'omnitexture', { chosen: true });
      const count = Math.max(1, Math.min(MAX_CAMERAS, Math.round(Number(cameras.value)) || CAMERAS));
      const params = { views: count, show };
      if (painted === 'alone') params.together = false;
      else if (painted) params.sheet_side = painted;
      if (example) params.example = example;
      if (sent) params.pictures = sent;
      if (words.value.trim()) params.prompt = words.value.trim();
      await startJob('omnitexture', [prop.id], { params });
      emit('goto', { stage: 'omnitexture', select: prop.id });
      close();
    } catch (e) {
      console.warn(e);
      toastError(e);
      status.textContent = '';
    } finally {
      busy = false;
      sync();
    }
  }

  // Each row also takes files dropped on it.
  const modelRow = h('div', { class: 'lab-row' },
    h('button', { type: 'button', class: 'btn ghost small', onclick: async () => takeModel(await pickModelFiles()) },
      icon('cube'), 'Choose the 3D model…'),
    modelName);
  modelDropTarget(modelRow, takeModel);
  const pictureRow = h('div', { class: 'lab-row' },
    h('button', { type: 'button', class: 'btn ghost small', onclick: async () => takePictures(await pickPictures()) },
      icon('image'), 'Choose pictures…'),
    pictureNames);
  modelDropTarget(pictureRow, (files) => takePictures(files.map((f) => f.file)));

  const el = h('div', { class: 'stack lab' },
    h('p', { class: 'hint' }, 'For working on the Omni-Texture stage itself: a 3D model from a file is painted without '
      + 'Images and 3D first. It becomes a new prop of this project, and the Omni-Texture stage shows the result.'),
    field('3D model', modelRow),
    field('Its pictures', pictureRow),
    field('Qwen-Image’s second picture', segmented(EXAMPLE, example, (v) => { example = v; }),
      'The example it takes the look from.'),
    field('Its other pictures sent to Qwen-Image', segmented(SENT, sent, (v) => { sent = v; }),
      'The stage sends none of them. Sent, they follow the example; blurred, they are shrunk to this on their '
      + 'longest side and enlarged back smoothly, so that their details blur away.'),
    field('Cameras', cameras, 'How many views are painted, in this order: front, back, front right, front left, '
      + 'back right, back left, top, bottom.'),
    field('The views are painted', segmented(PAINTED, painted, (v) => { painted = v; }),
      'Together, the views agree in colour: Qwen-Image paints them side by side in one picture, each as large as it '
      + 'would be alone. A larger picture holds more views, but Qwen-Image sees each of them smaller: it follows '
      + 'the model’s shape less closely and redraws faces and logos, and the picture takes more video memory.'),
    field('Qwen-Image is shown', segmented(SHOWN, show, (v) => { show = v; }),
      'Its first picture, the one a view painted alone starts from, shows this on the whole model.'),
    field('Words for Qwen-Image', words),
    h('div', { class: 'lab-row' }, go, status));
  sync();
  return el;
}
