// Stage 2, 3D: each prop's picture (or the views sliced from its pictures) becomes a textured 3D model
// (Pixal3D). The workbench shows the prop's model in the 3D viewer with its map bar (and a Wireframe toggle
// at its right end, for the model's triangles), its settings, "Make 3D again" and the [.glb] [.fbx] [.obj] buttons that save it into the project folder; an imported prop
// shows the texture maps its import found instead of the settings. "Import 3D model" (or a drop on the
// workbench) brings the artist's own 3D model in as a new prop. Several views need Pixal3D's multi-view
// models, a one-time download the panel offers. The settings start with the prop's Shape: Pixal3D's own or Hi3D (a
// new prop takes its project's, the Shape picked last in it, Pixal3D at first; Hi3D only where its code is), with
// "Powered by Hi3D 1.0" (a link to its site: Hi3D's licence asks for it) while Hi3D is picked or made the model
// shown. Hi3D's AI models are a one-time download of their own (/api/status "hi3d"): picking Hi3D while they are
// not on this PC asks first (agreeToHi3d: the download, slower, the front picture alone, the project's Shape from
// then on), then starts it; the note under Shape follows the download, or offers it to a prop set to Hi3D already.
// Under More: "Faster 3D generation" (faster.js), the hub's, on RTX 30xx and newer only.

import { formatBytes, formatCount, h, plural } from '../dom.js';
import { icon } from '../icons.js';
import { getProp, jobFor, store } from '../store.js';
import { downloadHi3d, downloadMultiview, queueSettings, startJob } from '../actions.js';
import { progressBar, rowField, segmented, seedInput, sliderField, spinner } from '../controls.js';
import { ShellRefusal, attempt, confirmDialog, modal, toast } from '../ui.js';
import {
  afterMove, boxFile, boxHeading, busyLayer, donateCorner, emptyState, errorLine, fadeIn, focusBox,
  gateLine, gateOpen, jobState, liveDeps, rowAction, stageJob, thumbUrl, toolGate,
} from './common.js';
import { asideHint, effectiveViews, frontOnly, madeWith, meshFresh, meshGeometry, smallViewsHint, tileable } from './rules.js';
import { fasterSwitch } from './faster.js';
import { exportButtons, modelPane, rememberWireframe, wireframeOn } from './preview.js';
import {
  NO_MODEL, detectedMaps, importModelBox, modelDropTarget, pickModelFiles, usableModelFiles,
} from './modelimport.js';

const DETAIL_MIN = 50_000;
const DETAIL_MAX = 1_000_000;
const NO_FRONT = 'Mark one of the slices as Front in Images: the 3D model is built facing it.';
// The hub's note on a model made from the front view alone (adapters/pixal3d.py FRONT_ONLY): out of date
// once the multi-view models are there.
const FRONT_ONLY = 'Used the front view only: the multi-view models aren’t downloaded yet.';
// Over a prop that has its 3D model while others are still being made: the model shows once all are made.
const HELD = 'Its 3D model shows once all are made';
const DEFAULT_QUALITY = 1536;     // High: the voxel grid a new prop is made at (projects.STAGE_DEFAULTS)

/** The number of views the prop's 3D model is made from now (rules.effectiveViews). */
const viewCount = (p) => effectiveViews(p.images).length;

/** The prop's 3D model is up to date (rules.meshFresh: with the multi-view models there, one made from the front view alone is not). */
const fresh = (p) => meshFresh(p, liveDeps());

/** Made from the front view alone, and now the multi-view models are here: made again, it uses every view. */
const againForViews = (p) => frontOnly(p) && !!store.status?.multiview?.available;

const QUALITY_HINT = 'High is sharper but takes longer and needs more memory.';
// The prop's Shape (mesh.settings.geometry; rules.GEOMETRIES): whose networks make it. The texture is Pixal3D's either way.
export const SHAPES = [
  { value: 'hi3d', label: 'Hi3D', title: 'Hi3D: finer carved detail, slower; from the front picture alone.' },
  { value: 'pixal3d', label: 'Pixal3D', title: 'Pixal3D: faster, follows the picture closely.' },
];
// Hi3D's licence (section 7) asks for these words, linked to its site, in the interface of a product that uses it.
export const POWERED_BY_HI3D = 'Powered by Hi3D 1.0';
export const HI3D_SITE = 'http://hi3d.ai/';
const HI3D_BYTES = 11_364_913_260;     // its AI models (install/models.py HI3D_BYTES), until the hub says
const gb = (bytes) => `${(bytes / 1e9).toFixed(1)} GB`;

/**
 * What the Shape note says while the prop is set to Hi3D and its AI models are not on this PC (/api/status "hi3d"),
 * or null: {text, button (its words, or null: none), failed, fraction (the download's, or null)}. The download is the
 * page's own (downloadHi3d; agreeToHi3d starts it when Hi3D is picked); a prop set to Hi3D already (a project from
 * another PC, say) gets the button; once the files are there the hub measures Hi3D's convolutions ("finishing").
 */
export function hi3dNote(prop, status) {
  if (meshGeometry(prop) !== 'hi3d') return null;
  const hi3d = status?.hi3d;
  if (!hi3d || (hi3d.available && !hi3d.finishing)) return null;
  // Without its code no download makes Hi3D work (this Spellcast3D does not have it yet).
  if (hi3d.code === false) {
    return { text: 'Hi3D is not part of this Spellcast3D yet. Pick Pixal3D.', button: null, failed: false, fraction: null };
  }
  if (hi3d.finishing) {
    return { text: 'Hi3D’s AI models are here. Measuring its convolutions on the graphics card (a few minutes; keep heavy programs closed)…',
      button: null, failed: false, fraction: null };
  }
  const size = gb(hi3d.total_bytes || HI3D_BYTES);
  if (hi3d.downloading) {
    const f = hi3d.total_bytes ? hi3d.done_bytes / hi3d.total_bytes : null;
    return { text: hi3d.waiting ? 'Hi3D’s AI models wait for the installer to finish…'
      : `Downloading Hi3D’s AI models (${size})…${f == null ? '' : ` ${Math.round(f * 100)}%`}`,
    button: null, failed: false, fraction: f };
  }
  if (hi3d.error) return { text: hi3d.error, button: 'Try again', failed: true, fraction: null };
  return { text: `Hi3D needs a one-time ${size} download of its AI models.`, button: `Download (${size})`,
    failed: false, fraction: null };
}

/**
 * The artist picks Hi3D for a prop. With its AI models on this PC (or their download running) nothing is asked.
 * Without them, a question first - the download, slower, the front picture alone, the project's Shape from then on -
 * and, agreed, the download starts. Resolves whether Hi3D is to be set.
 */
export async function agreeToHi3d() {
  const hi3d = store.status?.hi3d;
  if (!hi3d || hi3d.available || hi3d.downloading) return true;
  if (hi3d.code === false) {
    await confirmDialog({ title: 'Hi3D is not here', message: 'Hi3D is not part of this Spellcast3D yet. Pick Pixal3D.', cancel: null });
    return false;
  }
  const size = gb(hi3d.total_bytes || HI3D_BYTES);
  const m = modal({
    title: 'Use Hi3D for this prop?',
    body: h('div', { class: 'modal-text' },
      h('p', null, 'Hi3D carves finer detail than Pixal3D. Before it runs:'),
      h('ul', null,
        h('li', null, `Its AI models are a one-time ${size} download, into the models folder, in the background.`),
        h('li', null, 'It is slower than Pixal3D.'),
        h('li', null, 'It works from the front picture alone: a prop’s other views are not used.'),
        h('li', null, 'It becomes this project’s Shape for new props, until you pick Pixal3D again.'))),
    actions: [{ label: 'Cancel', value: false }, { label: `Download ${size} and use Hi3D`, kind: 'primary', value: true }],
  });
  if ((await m.done) !== true) return false;
  await attempt(() => downloadHi3d());
  return true;
}

let host = null;
let root = null;         // host > root: the prop view and the import progress over it
let ctx = null;
let view = null;         // the prop view on screen: {key, el, update(prop), destroy()}
let shownId = null;      // the prop shown last, kept while nobody picks another
let importing = null;    // {text, fraction} while the artist's model becomes a new prop
const importBar = progressBar(null);
const importText = h('span');
const importLine = h('div', { class: 'sv-importing', hidden: true, role: 'status' }, spinner(), importText, importBar);

const unsupported = () => toolGate('mesh').kind === 'unsupported';

/** 45 -> "45 s", 635 -> "10 min 35 s". */
const duration = (seconds) => {
  const s = Math.round(seconds);
  return s < 120 ? `${s} s` : `${Math.floor(s / 60)} min${s % 60 ? ` ${s % 60} s` : ''}`;
};

// ---- the artist's own model as a new prop ----------------------------------------------------------

function setImporting(state) {
  importing = state;
  importLine.hidden = !state;
  if (!state) return;
  importText.textContent = state.text;
  importBar.set(state.fraction);
}

/** Why an import can't start now (ShellRefusal), else nothing. Instant Meshes, a retopology tool, reads the models. */
function refuseImport() {
  if (importing) throw new ShellRefusal('Your model is still being imported. Wait a moment, then add the next one.');
  const gate = toolGate('retopo', 'imesh');   // whatever the props' default method is
  if (gate.kind === 'installing') {
    throw new ShellRefusal(`Your own models are read by the retopology tool, which is still installing${
      gate.fraction == null ? '' : ` (${Math.round(gate.fraction * 100)}%)`}. Try again once it is ready.`);
  }
  if (!gateOpen(gate)) throw new ShellRefusal(`Your own models are read by the retopology tool. ${gate.text}`);
}

/**
 * Uploads the model files as a new prop (the hub converts them before it answers); its tile joins the
 * props list and the workbench shows it.
 */
async function importNewProp(files) {
  refuseImport();
  const usable = usableModelFiles(files);
  if (!usable) throw new ShellRefusal(NO_MODEL);
  const count = plural(usable.length, 'file');
  setImporting({ text: `Uploading ${count}…`, fraction: 0 });
  try {
    const prop = await importModelBox(usable, (f) => setImporting(f == null || f >= 1
      ? { text: 'Converting your model: working out what each texture is…', fraction: null }
      : { text: `Uploading ${count}…`, fraction: f }));
    setImporting(null);
    ctx?.select(prop.id);
    const notes = prop.mesh?.imported?.warnings?.length || 0;
    toast(`Imported ${prop.name} as a new prop. Check the texture maps listed on the right.`
      + (notes ? ` ${plural(notes, 'note')} about it ${notes === 1 ? 'is' : 'are'} there too.` : ''), notes ? 'warn' : 'success');
  } catch (e) {
    toast(e.message, 'error', 12000);
  } finally {
    setImporting(null);
  }
}

/** A drop on the workbench: the same as "Import 3D model", with the refusal as a toast. */
async function dropped(files) {
  try {
    await importNewProp(files);
  } catch (e) {
    toast(e.message, 'warn', 8000);
  }
}

// ---- one prop -------------------------------------------------------------------------------------

function buildProp(prop) {
  const id = prop.id;
  const cur = () => getProp(id) || prop;
  const settings = () => cur().mesh?.settings || {};
  const set = (patch) => queueSettings(id, 'mesh', patch);
  const s = settings();

  // ---- left: the model -----------------------------------------------------------------------
  const pane = modelPane();
  pane.maps.offerWireframe(wireframeOn('mesh', false), (on) => rememberWireframe('mesh', on));
  const busy = busyLayer('mesh');
  const convertText = h('div', { class: 'sv-busy-text' });
  const converting = h('div', { class: 'sv-busy', hidden: true }, spinner(), convertText);
  pane.stage.append(busy, converting);
  const media = h('div', { class: 'sv-media' }, pane.el);

  let shownModel = null;
  function renderMedia(p) {
    const glb = p.mesh?.glb;
    const chosen = p.images?.chosen;
    const stale = !!glb && !fresh(p);
    pane.badges(
      p.mesh?.imported ? h('span', { class: 'chip', title: 'You imported this model. Its texture maps are listed on the right.' },
        icon('upload'), 'Your own model') : null,
      stale && againForViews(p)
        ? h('span', { class: 'chip warn', title: 'The multi-view models are here now: make it again to use every view.' },
          icon('alert'), 'Made from the front view only')
        : stale ? h('span', { class: 'chip warn', title: 'The picture or its views changed after this model was made. Make it again to update it.' },
          icon('alert'), 'Made from an older picture') : null);
    // While its 3D model is being made, the picture stays out of the view: a batch takes every prop through each
    // step in turn, so the busy words are about other props too, and the faint picture of this one misled. No 3D
    // model shows while any is being made: the progress stays in its place (HELD), the card is the job's.
    const working = !!jobFor('mesh', p.id);
    const held = !!glb && !!stageJob('mesh');
    const key = held ? 'held' : glb ? `glb|${boxFile(p, glb, 'mesh')}` : `none|${chosen}|${unsupported()}|${working}|${tileable(p)}`;
    if (key === shownModel) return;
    shownModel = key;
    if (held) {
      pane.message(h('div'));
      return;
    }
    if (glb) {
      pane.show(boxFile(p, glb, 'mesh'), 'glb', afterMove(ctx, thumbUrl(p, 'mesh')));
      return;
    }
    const picture = chosen && !working ? h('img', { class: 'sv-ghost-pic', src: boxFile(p, chosen, 'images'), alt: '', draggable: 'false' }) : null;
    pane.message(h('div', { class: 'sv-pic-note' }, picture,
      !chosen ? emptyState('image', 'No picture yet', 'This prop needs a picture before it can become 3D.',
        h('button', { class: 'btn ghost', onclick: () => ctx?.goto('images') }, icon('image'), 'Go to Images'))
        : tileable(p) ? emptyState('image', 'A tileable texture', 'It gets no 3D model: “Generate 3D” passes it by. '
          + 'To make one from this picture, untick Tileable in Images.',
        h('button', { class: 'btn ghost', onclick: () => ctx?.goto('images') }, icon('image'), 'Go to Images'))
        : unsupported() ? emptyState('alert', 'This computer can’t make 3D models from pictures', toolGate('mesh').text,
          h('p', { class: 'hint' }, 'You can still use 3D models of your own: press “Import 3D model” on the left.'))
          : emptyState('cube', 'Ready to become 3D', viewCount(p) >= 2
            ? `Press “Make 3D model” to turn its ${viewCount(p)} views into a textured model.`
            : 'Press “Make 3D model” to turn this picture into a textured model.')));
  }

  // ---- right: the prop, its settings or its imported maps ---------------------------------------
  const heading = boxHeading();
  // Shape: Hi3D or Pixal3D, with Hi3D's credit beside its name while it is picked or made the model shown.
  const shape = segmented(SHAPES, meshGeometry(prop), async (v) => {
    if (v === 'hi3d' && !(await agreeToHi3d())) {   // declined, or Hi3D is not here: the Shape stays as it was
      syncShape(cur());
      return;
    }
    set({ geometry: v });
    syncShape(cur());
  });
  shape.setAttribute('aria-label', 'Shape');
  const poweredBy = h('a', {
    class: 'chip hi3d', href: HI3D_SITE, target: '_blank', rel: 'noopener noreferrer',
    title: 'Hi3D’s website (opens in your browser)',
  }, POWERED_BY_HI3D);
  // Every setting on one line, its control beside its name (rowField): the panel is half as tall. Hi3D's credit,
  // while it shows, has the line under the Shape.
  const shapeField = rowField('Shape', shape);
  shapeField.classList.add('sv-shape');
  shapeField.append(poweredBy);
  const hi3dText = h('span');
  const hi3dBar = progressBar(0);
  const hi3dWords = h('span');
  const hi3dBtn = h('button', {
    class: 'btn ghost small', onclick: () => getHi3d(),
    title: 'Spellcast3D downloads them once, in the background, into its models folder. You can go on working meanwhile.',
  }, icon('download'), hi3dWords);
  const hi3dBox = h('div', { class: 'sv-mv-note', hidden: true }, icon('info'),
    h('div', { class: 'sv-mv-body' }, hi3dText, hi3dBar, hi3dBtn));
  let askingHi3d = false;   // until the hub has started the download: a double click asks once
  async function getHi3d() {
    if (askingHi3d) return;
    askingHi3d = hi3dBtn.disabled = true;
    await attempt(() => downloadHi3d());
    askingHi3d = false;
    sync(cur());
  }
  /** The Shape as the prop has it, Hi3D's credit while it is picked or made the model, and its download's state. */
  function syncShape(p) {
    shape.set(meshGeometry(p));
    poweredBy.hidden = meshGeometry(p) !== 'hi3d' && madeWith(p) !== 'hi3d';
    const note = unsupported() ? null : hi3dNote(p, store.status);
    hi3dBox.hidden = !note;
    if (!note) return;
    hi3dText.textContent = note.text;
    hi3dBtn.hidden = !note.button;
    hi3dBtn.disabled = askingHi3d;
    hi3dWords.textContent = note.button || '';
    hi3dBar.hidden = note.fraction == null;
    if (note.fraction != null) hi3dBar.set(note.fraction);
    hi3dBox.classList.toggle('failed', note.failed);
  }
  const quality = segmented([
    { value: 1024, label: 'Standard 1K', title: '1024 voxel grid' },
    { value: 1536, label: 'High 1.5K', title: '1536 voxel grid' },
  ], s.resolution ?? DEFAULT_QUALITY, (v) => set({ resolution: v }));
  const qualityField = rowField('Quality', quality, QUALITY_HINT);
  /** The quality the prop is set to (High for every prop, several views or one, until the artist picks Standard). */
  function syncQuality() {
    quality.set(settings().resolution ?? DEFAULT_QUALITY);
  }
  const detail = sliderField('High-poly detail', {
    value: s.decimation_target ?? 300_000,
    toPos: (v) => Math.round((Math.log(v / DETAIL_MIN) / Math.log(DETAIL_MAX / DETAIL_MIN)) * 1000),
    fromPos: (pos) => Math.round((DETAIL_MIN * (DETAIL_MAX / DETAIL_MIN) ** (pos / 1000)) / 10_000) * 10_000,
    format: (v) => `${formatCount(v)} tris`,
    onInput: (v) => set({ decimation_target: v }),
    hint: 'Fewer triangles make a lighter file. Retopo cleans it up afterwards anyway.',
    row: true,
  });
  const texture = segmented([
    { value: 1024, label: '1K' }, { value: 2048, label: '2K' }, { value: 4096, label: '4K' },
  ], s.texture_size ?? 2048, (v) => set({ texture_size: v }));
  const seed = seedInput(s.seed ?? 0, (v) => set({ seed: v }), { randomMeansMinusOne: false });
  const faster = fasterSwitch('mesh');     // the hub's, for every prop (faster.js)
  const more = h('details', { class: 'fold' }, h('summary', null, 'More'), h('div', { class: 'sv-group' }, faster));
  const settingsGroup = h('div', { class: 'sv-group' },
    shapeField,
    hi3dBox,
    qualityField,
    detail,
    rowField('Texture size', texture, 'Resolution of the colour texture.'),
    rowField('Seed', seed, 'Change it for a different take on the same picture.'),
    more);
  const maps = detectedMaps({
    onBusy: (text) => {
      converting.hidden = !text;
      convertText.textContent = text || '';
    },
  });

  // How the model was made, when it matters: the 3D generator's note about it, a view missing, views that
  // wait on another picture, and the multi-view models several views need (a one-time download).
  const madeNote = h('div', { class: 'sv-warning', hidden: true }, icon('alert'), h('span'));
  const frontNote = h('div', { class: 'sv-warning', hidden: true }, icon('alert'), h('span', null, NO_FRONT));
  const asideNote = h('div', { class: 'sv-mv-note', hidden: true }, icon('info'), h('span'));
  const smallNote = h('div', { class: 'sv-mv-note', hidden: true }, icon('info'), h('span'));
  const mvText = h('span');
  const mvBar = progressBar(0);
  const mvBtn = h('button', {
    class: 'btn ghost small', onclick: () => getMultiview(),
    title: 'Spellcast3D downloads them once, in the background, into its models folder. You can go on working meanwhile.',
  }, icon('download'), 'Download (22 GB)');
  const mvNote = h('div', { class: 'sv-mv-note', hidden: true }, icon('info'), h('div', { class: 'sv-mv-body' }, mvText, mvBar, mvBtn));
  let asking = false;   // until the hub has started the download: a double click asks once
  async function getMultiview() {
    if (asking) return;
    asking = mvBtn.disabled = true;
    await attempt(() => downloadMultiview());
    asking = false;
    sync(cur());
  }

  const makeLabel = h('span');
  const makeBtn = h('button', { class: 'btn primary big sv-wide', onclick: () => make() }, icon('cube'), makeLabel);
  const exports = exportButtons('mesh');
  const gate = gateLine('mesh', { unsupported: 'You can still import 3D models of your own with “Import 3D model”.' });
  const error = errorLine('mesh');
  const foot = h('div', { class: 'sv-foot' }, error, gate, exports, makeBtn);
  const side = h('div', { class: 'sv-side' },
    h('div', { class: 'sv-side-scroll' }, heading, madeNote, frontNote, asideNote, smallNote, mvNote, settingsGroup, maps),
    foot);
  const el = h('div', { class: 'sv sv-mesh' }, media, side);

  let starting = false;   // until the job is queued: a double click must not queue it twice
  async function make() {
    if (starting) return;
    starting = makeBtn.disabled = true;
    await attempt(() => startJob('mesh', [id]));
    starting = false;
    sync(cur());
  }

  /** "Made in 45 s · Seed 1234 · 4 views": the model's numbers on one line, under the prop's name. */
  function statsLine(p) {
    const st = p.mesh?.glb ? p.mesh.stats || {} : {};
    return [
      typeof st.seconds === 'number' ? `Made in ${duration(st.seconds)}` : '',
      !p.mesh?.imported && st.seed != null ? `Seed ${st.seed}` : '',
      !p.mesh?.imported && st.views >= 2 ? `${st.views} views` : '',
    ].filter(Boolean).join(' · ');
  }

  /**
   * The notes over the settings: the 3D generator's warning about the model (it may be stretched...); a
   * front view missing among several views; views that wait on another of its pictures, which it still has
   * (this one has none and is made into 3D alone: rules.asideHint); views that are small slices of a sheet
   * (a softer model: rules.smallViewsHint); and the multi-view models several
   * views need: a quiet offer to download them, their download's progress, or why it stopped. (A model made
   * from the front view alone says so on the model once they are here: made again, it uses every view.)
   */
  function syncNotes(p) {
    const views = effectiveViews(p.images);
    const several = views.length >= 2 && !unsupported();
    const mv = store.status?.multiview;
    const made = p.mesh?.glb && !p.mesh.imported ? p.mesh.stats || {} : null;
    mvNote.hidden = !several || !mv || mv.available;
    // "Used the front view only: the multi-view models aren't downloaded yet" says what the download note
    // says (or is out of date once they are here): said once.
    const warning = ((mv?.available || !mvNote.hidden ? made?.warning?.replace(FRONT_ONLY, '') : made?.warning) || '').trim();
    madeNote.hidden = !warning;
    madeNote.lastChild.textContent = warning;
    frontNote.hidden = !several || views.some((v) => v.view === 'front');
    // Only for views on pictures the prop still has (rules.viewsAside): never for a picture that just has none.
    const aside = asideHint(p.images);
    asideNote.hidden = !aside || unsupported();
    asideNote.lastChild.textContent = aside;
    // Small slices of a sheet make a softer model than full pictures (rules.smallViewsHint).
    const small = several ? smallViewsHint(p.images) : '';
    smallNote.hidden = !small;
    smallNote.lastChild.textContent = small;
    if (mvNote.hidden) return;
    const f = mv.total_bytes ? Math.min(1, (mv.done_bytes || 0) / mv.total_bytes) : null;
    mvText.textContent = mv.downloading
      ? `Downloading multi-view models…${f == null ? '' : ` ${Math.round(f * 100)}%`}`
      : mv.error ? `The multi-view models could not be downloaded: ${mv.error}`
        : 'Multi-view needs a one-time 22 GB download of extra Pixal3D models. Until then the front view is used.';
    mvNote.title = mv.downloading && mv.total_bytes
      ? `${formatBytes(mv.done_bytes || 0)} of ${formatBytes(mv.total_bytes)}. The front view is used until they are here.` : '';
    mvBar.hidden = !mv.downloading;
    mvBar.set(f);
    mvBtn.hidden = !!mv.downloading;
    mvBtn.disabled = asking;
    mvBtn.lastChild.textContent = mv.error ? 'Try again' : 'Download (22 GB)';
    mvNote.classList.toggle('failed', !mv.downloading && !!mv.error);
  }

  function sync(p) {
    const j = jobFor('mesh', id);
    // Its job has begun it: it is on the 3D generator now, or waits there for its next step.
    const begun = j?.phase === 'running' || j?.phase === 'resting';
    const canMake = !!p.images?.chosen && !unsupported() && !tileable(p);
    settingsGroup.hidden = !canMake;
    makeBtn.hidden = !canMake;
    makeBtn.disabled = starting || !!j || !gateOpen(toolGate('mesh'));
    makeBtn.classList.toggle('working', !!j);
    makeLabel.textContent = j ? (begun ? 'Making 3D…' : 'Waiting…')
      : p.mesh?.glb ? 'Make 3D again' : 'Make 3D model';
    // A fresh model needs nothing: making it again is there, quietly, under its export buttons.
    const quiet = !j && fresh(p);
    makeBtn.classList.toggle('primary', !quiet);
    makeBtn.classList.toggle('big', !quiet);
    makeBtn.classList.toggle('ghost', quiet);
    exports.sync(p.mesh?.glb ? p : null, !j);
    gate.sync();
    gate.hidden ||= !p.images?.chosen;   // an imported prop needs no 3D generator
    error.sync(p);
    busy.sync(p, p.mesh?.glb ? HELD : '');
    syncShape(p);
    syncQuality();
    more.hidden = !faster.applies();
    faster.sync();
    syncNotes(p);
    heading.set(p, p.mesh?.imported ? 'Your own 3D model' : statsLine(p),
      p.images?.chosen ? thumbUrl(p, 'images') || boxFile(p, p.images.chosen, 'images') : thumbUrl(p, 'mesh'));
  }

  const update = (p) => {
    renderMedia(p);
    maps.update(p);
    sync(p);
  };

  return { key: id, el, update, destroy: () => pane.destroy() };
}

// ---- no props yet ---------------------------------------------------------------------------------

function buildEmpty() {
  const choose = async (folder) => {
    const files = await pickModelFiles(folder);
    if (files.length) dropped(files);
  };
  const el = h('div', { class: 'sv sv-mesh sv-single' },
    h('div', { class: 'sv-stagebox sv-dropspace' }, emptyState('cube', 'No props here yet',
      'Make some props in Images and press “Generate 3D”. Or bring a 3D model of your own: press “Import 3D model”, '
      + 'or drop its files (FBX, OBJ, GLB or glTF with its textures, a folder or a .zip) here.',
      h('button', { class: 'btn ghost', onclick: () => ctx?.goto('images') }, icon('image'), 'Go to Images'),
      h('button', { class: 'btn ghost', onclick: () => choose(false) }, icon('upload'), 'Import a 3D model…'),
      h('button', { class: 'btn ghost', onclick: () => choose(true) }, icon('folder'), 'Choose a folder…')),
    donateCorner()));
  return { key: 'empty', el, update() {}, destroy() {} };
}

// ---- per prop, for the props list -----------------------------------------------------------------

function status(prop) {
  const job = jobState('mesh', prop);
  if (job) return job;
  if (prop.mesh?.glb) {
    return fresh(prop) ? { state: 'done', message: prop.mesh.imported ? 'Your own 3D model' : 'Has its 3D model' }
      : againForViews(prop) ? { state: 'stale', message: 'Made from the front view only: make it again to use every view' }
        : { state: 'stale', message: 'Made from an older picture: make it again' };
  }
  if (!prop.images?.chosen) {
    return jobFor('images', prop.id) ? { state: 'waiting', message: 'Waiting for its picture' }
      : { state: 'unreachable', message: 'No picture yet' };
  }
  if (tileable(prop)) return { state: 'unreachable', message: 'A tileable texture: it gets no 3D model' };
  if (unsupported()) return { state: 'unreachable', message: 'This computer can’t make 3D models from pictures' };
  return { state: 'idle', message: 'Not made into 3D yet' };
}

// ---- the stage view -------------------------------------------------------------------------------

function render() {
  if (!host) return;
  const { box: prop } = focusBox(ctx, 'mesh', { needsWork: (b) => !fresh(b) && !tileable(b), previousId: shownId });
  shownId = prop?.id ?? null;
  if (view?.key !== (prop ? prop.id : 'empty')) {
    view?.destroy();
    view = prop ? buildProp(prop) : buildEmpty();
    root.replaceChildren(view.el, importLine);
    fadeIn(view.el);
  }
  if (prop) view.update(prop);
}

export default {
  id: 'mesh',
  title: '3D',
  ctaLabel: 'Omni-Texture',
  ctaHover: 'Goes on to Omni-Texture, where you may choose props whose colour texture is painted anew. Nothing starts '
    + 'by itself there: props you do not choose keep their own texture.',
  plus: 'import',

  mount(el, c) {
    host = el;
    ctx = c;
    root = h('div', { class: 'sv-root' });
    modelDropTarget(root, dropped);
    host.replaceChildren(root);
    render();
  },

  update(c) {
    ctx = c;
    render();
  },

  unmount() {
    view?.destroy();
    view = null;
    host = root = null;
  },

  status,

  /** The prop the workbench shows. */
  shownId: () => shownId,

  /** Omni-Texture is optional: the call to action only goes there (its own call to action retopologizes). */
  ctaEnabled: () => ({ enabled: true }),

  async onCta() {},

  /** "Make 3D for 3 props": every prop with a picture and no fresh model, made right here. */
  rowAction: () => rowAction('mesh', shownId, 'Make 3D for', 'cube'),

  /** Imports the artist's own 3D model (its files, or a .zip) as a new prop. */
  async onPlus() {
    refuseImport();
    const files = await pickModelFiles();
    if (files.length) await importNewProp(files);
  },
};
