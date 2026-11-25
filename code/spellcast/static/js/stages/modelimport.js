// The artist's own 3D models: collecting the files (picked, or dropped with their folders), importing
// them as a new prop (POST .../props/import-model), and the "Detected maps" list of an imported prop,
// where a wrongly guessed texture role is corrected and the model converted again (POST .../mesh/reimport).

import { api, projectApi, propApi, uploadFiles } from '../api.js';
import { h } from '../dom.js';
import { icon } from '../icons.js';
import { bump, emit, getProp, projectId, store } from '../store.js';
import { replaceProp } from '../actions.js';
import { field } from '../controls.js';
import { toast } from '../ui.js';

// What an import takes; keep in sync with projects.MODEL_SUFFIXES / MODEL_COMPANION_SUFFIXES.
const MODEL_SUFFIXES = ['.fbx', '.glb', '.gltf', '.obj', '.dae', '.ply', '.stl', '.off'];
const IMPORT_SUFFIXES = new Set([...MODEL_SUFFIXES, '.zip', '.mtl', '.bin', '.png', '.jpg', '.jpeg', '.tga', '.tif',
  '.tiff', '.bmp', '.webp', '.dds', '.psd', '.exr']);
export const NO_MODEL = 'There is no 3D model among these files. Choose the .fbx, .obj, .glb or .gltf file together '
  + 'with its textures, or a .zip that contains them.';

const suffixOf = (name) => {
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(dot).toLowerCase() : '';
};

/** The files an import can use ([{file, path}]), or null when there is no model (or .zip) among them. */
export function usableModelFiles(files) {
  const usable = files.filter(({ path }) => IMPORT_SUFFIXES.has(suffixOf(path)));
  return usable.some(({ path }) => MODEL_SUFFIXES.includes(suffixOf(path)) || suffixOf(path) === '.zip') ? usable : null;
}

/** Every importable file under dropped folders / files, each with its path ("Crate/textures/a.png"). */
async function droppedFiles(dataTransfer) {
  const entries = [...dataTransfer.items].map((item) => item.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...dataTransfer.files].map((file) => ({ file, path: file.name }));
  const walk = async (entry) => {
    if (entry.isFile) {
      const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
      return [{ file, path: entry.fullPath.replace(/^\/+/, '') || file.name }];
    }
    const reader = entry.createReader();
    const found = [];
    for (;;) {   // readEntries answers in batches until it returns none
      const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
      if (!batch.length) return found;
      for (const child of batch) found.push(...await walk(child));
    }
  };
  return (await Promise.all(entries.map(walk))).flat();
}

/** Resolves with the chosen files ([{file, path}]); [] when the picker is closed. folder: pick a whole folder. */
export function pickModelFiles(folder = false) {
  return new Promise((resolve) => {
    const input = h('input', {
      type: 'file', multiple: true, style: 'display:none',
      accept: folder ? null : [...IMPORT_SUFFIXES].join(','), webkitdirectory: folder || null,
    });
    const done = (files) => { resolve(files); input.remove(); };
    input.addEventListener('change', () => done([...input.files].map((file) => ({ file, path: file.webkitRelativePath || file.name }))));
    input.addEventListener('cancel', () => done([]));
    document.body.append(input);
    input.click();
  });
}

/** Makes `el` take dropped model files and folders: onFiles([{file, path}]). */
export function modelDropTarget(el, onFiles) {
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
  el.addEventListener('drop', async (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.stopPropagation();
    depth = 0;
    el.classList.remove('drag-over');
    let files;
    try {
      files = await droppedFiles(e.dataTransfer);
    } catch (err) {
      toast(`These files could not be read: ${err.message || err}`, 'error');
      return;
    }
    onFiles(files);
  });
}

/**
 * The model files ([{file, path}], or one .zip) as a new prop, the last one of the props list; it starts
 * as a 3D model. onProgress(fraction | null) follows the upload; the hub converts the model before it answers.
 */
export async function importModelBox(files, onProgress) {
  const pid = projectId();
  const prop = await uploadFiles(`${projectApi(pid)}/props/import-model`, files, { onProgress });
  if (projectId() === pid && !store.project.props.some((p) => p.id === prop.id)) {
    store.project.props.push(prop);
    emit('project', { hard: true });
  }
  return prop;
}

// ---- detected maps -------------------------------------------------------------------------------

// What an imported texture can be (instant_meshes_brush.texroles.ROLES), in artist words.
const ROLE_LABELS = {
  basecolor: 'Base colour', normal: 'Normal', height: 'Height (bump)', roughness: 'Roughness',
  gloss: 'Gloss (smoothness)', metallic: 'Metallic', specular: 'Specular colour',
  spec_gloss: 'Specular + gloss in alpha', ao: 'Ambient occlusion', emissive: 'Emissive', opacity: 'Opacity',
  orm: 'ORM (AO, roughness, metallic)', metal_smooth: 'Metallic + smoothness in alpha',
  mask_hdrp: 'HDRP mask (metallic, AO, smoothness)', other: 'Other (not used)',
};
// Single-channel maps that some programs store the other way round.
const INVERTIBLE = new Set(['roughness', 'gloss', 'metallic', 'ao', 'opacity', 'height']);
const UNSURE_BELOW = 0.6;

/**
 * What the import of an imported prop found: its source file, each texture with its guessed role (the
 * artist may correct it and press Apply), warnings and the files left out. update(prop) re-renders
 * when the import changed. onBusy(text | null) follows the conversion Apply starts.
 */
export function detectedMaps({ onBusy } = {}) {
  const el = h('div', { class: 'sv-group' });
  let draft = {};     // the artist's corrections, sent with Apply (the full set replaces the stored one)
  let shown = null;
  let converting = false;

  function render(prop) {
    const imp = prop.mesh?.imported;
    draft = structuredClone(imp?.overrides || {});
    if (!imp) {
      el.replaceChildren();
      return;
    }
    const source = (prop.mesh.source || '').split('/').pop();
    const byKey = new Map();
    for (const material of imp.materials || []) {
      for (const map of material.maps || []) {
        const seen = byKey.get(map.key);
        if (seen) seen.materials.push(material.name);
        else byKey.set(map.key, { map, materials: [material.name] });
      }
    }
    const apply = h('button', { class: 'btn primary small', onclick: () => reimport(prop.id, draft) }, icon('check'), 'Apply changes');
    const undo = h('button', { class: 'btn ghost small', onclick: () => render(getProp(prop.id) || prop) }, 'Undo');
    const reset = h('button', {
      class: 'btn ghost small', title: 'Forget your corrections and use what Spellcast3D guessed',
      hidden: !Object.keys(imp.overrides || {}).length, onclick: () => reimport(prop.id, {}),
    }, icon('reset'), 'Back to the guesses');
    const syncActions = () => {
      apply.hidden = undo.hidden = JSON.stringify(draft) === JSON.stringify(imp.overrides || {});
    };
    syncActions();
    const multi = (imp.materials || []).length > 1;
    const rows = [...byKey.values()].map(({ map, materials }) => mapRow(map, multi ? materials : [], imp, syncActions));
    el.replaceChildren(
      h('p', { class: 'hint' }, 'Imported from ', h('b', null, source || 'your files'), '.'),
      rows.length
        ? field('Detected maps', h('div', { class: 'sv-maplist' }, rows),
          'What each texture was taken for. Correct a wrong guess, then press Apply: the model is converted again.')
        : h('p', { class: 'hint' }, 'No textures were found with this model.'),
      h('div', { class: 'sv-row-btns' }, apply, undo, reset),
      (imp.warnings || []).map((w) => h('div', { class: 'sv-warning' }, icon('alert'), h('span', null, w))),
      imp.skipped?.length ? h('p', { class: 'hint' }, `Left out (neither a model nor a texture): ${imp.skipped.join(', ')}`) : '');
    syncBusy();
  }

  function mapRow(map, materials, imp, changed) {
    const edit = (patch) => {
      const next = { ...draft[map.key], ...patch };
      for (const [k, v] of Object.entries(patch)) {
        // Back to what the model has now, and not a correction stored before: nothing to override.
        const now = k === 'role' ? map.role : false;
        if (v === now && imp.overrides?.[map.key]?.[k] === undefined) delete next[k];
      }
      if (Object.keys(next).length) draft[map.key] = next;
      else delete draft[map.key];
      paint();
      changed();
    };
    const role = () => draft[map.key]?.role ?? map.role;
    const labels = map.role in ROLE_LABELS ? ROLE_LABELS : { ...ROLE_LABELS, [map.role]: map.label || map.role };
    const select = h('select', { class: 'input small', 'aria-label': `What ${map.file} is` },
      Object.entries(labels).map(([value, label]) => h('option', { value }, label)));
    select.value = role();
    select.addEventListener('change', () => edit({ role: select.value }));
    const flag = (key, label, title) => h('button', {
      type: 'button', class: 'chip toggle', title, onclick: () => edit({ [key]: !draft[map.key]?.[key] }),
    }, label);
    const flip = flag('flip_green', 'Flip green', 'For a DirectX-style normal map: its bumps look pushed in and the light comes from the wrong side.');
    const invert = flag('invert', 'Invert', 'For a map stored the other way round, e.g. gloss read as roughness: what should shine looks matt.');
    const paint = () => {
      const r = role();
      flip.hidden = r !== 'normal';
      invert.hidden = !INVERTIBLE.has(r);
      for (const [b, key] of [[flip, 'flip_green'], [invert, 'invert']]) {
        const on = !!draft[map.key]?.[key];
        b.classList.toggle('on', on);
        b.setAttribute('aria-pressed', String(on));
      }
    };
    paint();
    const unsure = Number(map.confidence) < UNSURE_BELOW;
    const why = [`${map.file}: ${map.label || ROLE_LABELS[map.role] || map.role}, ${Math.round(Number(map.confidence) * 100)}% sure`,
      ...(map.evidence || []).slice(0, 4).map((e) => `- ${e}`)];
    if (map.role === 'normal' && map.y_convention) {
      why.push(`- read as ${map.y_convention === 'directx' ? 'DirectX (green down), turned into OpenGL' : 'OpenGL (green up)'}`
        + `${map.y_confident ? '' : ', a guess'}`);
    }
    return h('div', { class: 'sv-maprow' },
      h('div', { class: 'sv-mapfile', title: why.join('\n') },
        h('span', { class: 'sv-mapname' }, map.file),
        unsure ? h('span', { class: 'sv-unsure', title: 'Not sure about this one: please check it' }, '?') : null,
        materials.length ? h('span', { class: 'hint' }, materials.join(', ')) : null),
      h('div', { class: 'sv-mapedit' }, select, flip, invert));
  }

  async function reimport(propId, overrides) {
    converting = true;
    syncBusy();
    onBusy?.('Converting your model again with your corrections…');
    try {
      const updated = await api('POST', `${propApi(projectId(), propId)}/mesh/reimport`, { overrides });
      bump('mesh', propId);
      replaceProp(updated);
      toast(`Converted ${updated.name} again with your corrections.`, 'success');
    } catch (e) {
      toast(e.message, 'error', 12000);
    } finally {
      converting = false;
      syncBusy();
      onBusy?.(null);
    }
  }

  function syncBusy() {
    for (const b of el.querySelectorAll('button, select')) b.disabled = converting;
  }

  el.update = (prop) => {
    const key = JSON.stringify([prop.id, prop.mesh?.glb, prop.mesh?.imported]);
    if (key === shown) return;
    shown = key;
    render(prop);
  };
  return el;
}
