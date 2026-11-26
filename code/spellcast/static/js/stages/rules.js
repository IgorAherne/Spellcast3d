// The rules the stage views follow, as plain functions of the props (no page, no hub), so Node can test
// them (code/tests/web): what a prop's 3D model is made from (its picture, or the views sliced from its
// pictures while it shows one of theirs) and which of its results are fresh, what going down to a stage with every prop
// (or working on them in their own stage) queues, on which tool, and how to say so, and the names the Images workbench
// suggests. ("box" in the code is a prop: the props were drawn as boxes once.)

// ---- the tools a stage's work runs on ------------------------------------------------------------

/**
 * The tools that can run each stage, its default first (the same as paths.STAGE_TOOLS). Retopo has a
 * choice: each prop picks its method (retopo.settings.method). Images runs on the one image generator
 * spellcast3d-config.ini picks (the page asks the hub: store.imageTool).
 */
export const STAGE_TOOLS = {
  images: ['qwen', 'forge'], mesh: ['pixal3d'], omnitexture: ['qwen'], retopo: ['triflow', 'lato2', 'imesh'],
  projection: ['imesh'],
};

/**
 * The order "Retopo All"'s jobs run in, one job per method: the GPU tools first (TriFlow, the default, before
 * LATO.2), each loaded once for all its props; Instant Meshes (CPU) last.
 */
export const RETOPO_ORDER = ['triflow', 'lato2', 'imesh'];

/**
 * The retopology method the prop is set to (retopo.settings.method; TriFlow, the default, when unset: the
 * same as projects.DEFAULT_RETOPO_METHOD). A prop that picked a method keeps it.
 */
export const retopoMethod = (box) => box?.retopo?.settings?.method || STAGE_TOOLS.retopo[0];

/** The tool that does a prop's work in `stage`: its retopology method in Retopo, else the stage's default tool. */
export const toolOf = (stage, box) => (stage === 'retopo' ? retopoMethod(box) : STAGE_TOOLS[stage]?.[0] ?? null);

/**
 * What the page asks of the hub as the artist arrives at a stage, or picks something else there (POST .../warm).
 * `wish`: the stage's, {tool, model?, displace?} or null; `tools`: the hub's status of each. -> {tool, wish} to
 * send, or null: a tool that cannot run here is not asked, and one that runs already only when the wish names a
 * model (it may hold another: the hub then starts it anew for that one).
 */
export function warmAsk(tools, wish) {
  const { tool, ...rest } = wish || {};
  const st = tool ? tools?.[tool] : null;
  if (!st?.installed || !st.supported) return null;
  return st.state === 'stopped' || rest.model ? { tool, wish: rest } : null;
}

/**
 * Whose networks make a prop's 3D shape (mesh.settings.geometry, the 3D stage's "Shape"; projects.GEOMETRIES): Hi3D
 * (refining Pixal3D's coarse shape from the front picture) or Pixal3D's own; a new prop takes its project's Shape, the
 * one picked last in it (Pixal3D at first). Both run on Pixal3D's tool, which also makes the texture either way.
 */
export const GEOMETRIES = ['hi3d', 'pixal3d'];
/** A prop that names none (saved before there was a choice) keeps Pixal3D's: projects.mesh_geometry. */
export const meshGeometry = (box) => (GEOMETRIES.includes(box?.mesh?.settings?.geometry) ? box.mesh.settings.geometry
  : 'pixal3d');
/** The geometry that made the prop's 3D model (mesh.geometry), or null: imported, none yet, or made before the choice. */
export const madeWith = (box) => (box?.mesh?.glb && GEOMETRIES.includes(box.mesh.geometry) ? box.mesh.geometry : null);

// ---- views: the sides of the object a prop's pictures show ------------------------------------------

/**
 * The views a prop's pictures can be sliced into, in Pixal3D's orbit order (azimuth 0, 90, 180, 270):
 * the front, the side where the object faces image-left (its own left side), the back, and the side
 * where it faces image-right. images.views = [{view, src, rect: [x, y, w, h] | null}], one per kind.
 */
export const VIEW_ORDER = ['front', 'left', 'back', 'right'];

/**
 * The views in one line, the same whatever order they were saved in: "front:images/0003.png:0,0,512,512|
 * left:images/0003.png:*" (* = the whole picture; 512.0 is written 512). The same as projects.views_signature.
 */
export function viewsSignature(views) {
  return [...(views || [])]
    .sort((a, b) => VIEW_ORDER.indexOf(a.view) - VIEW_ORDER.indexOf(b.view))
    .map((v) => `${v.view}:${v.src}:${v.rect ? v.rect.join(',') : '*'}`)
    .join('|');
}

/**
 * The views a prop's 3D model is made from now: all of them while its chosen picture is one of theirs,
 * else none (the chosen picture alone; the views wait for one of theirs to be picked again). The same as
 * projects.effective_views.
 */
export function effectiveViews(images) {
  const views = (images?.views || []).filter((v) => v && typeof v === 'object');
  const chosen = images?.chosen;
  return chosen && views.some((v) => v.src === chosen) ? views : [];
}

/**
 * The views that wait on other pictures of the prop: the chosen picture is none of theirs, so it is made
 * into 3D alone. Only views on pictures the prop still has (images.candidates) count; [] while the views
 * are used (effectiveViews), and without a chosen picture.
 */
export function viewsAside(images) {
  if (!images?.chosen || effectiveViews(images).length) return [];
  const kept = Array.isArray(images.candidates) ? images.candidates : null;
  return (images.views || []).filter((v) => v && typeof v === 'object' && VIEW_ORDER.includes(v.view)
    && typeof v.src === 'string' && v.src !== images.chosen && (!kept || kept.includes(v.src)));
}

/**
 * The 3D stage's note for a prop whose views wait on other pictures (viewsAside), from what the artist
 * sees: this picture has none, so it is made alone, and where the views are. '' when none wait.
 */
export function asideHint(images) {
  const aside = viewsAside(images);
  if (!aside.length) return '';
  const count = `${aside.length} ${aside.length === 1 ? 'view' : 'views'}`;
  const use = aside.length === 1 ? 'use it' : 'use them';
  const where = new Set(aside.map((v) => v.src)).size === 1
    ? `Another picture of this prop has ${count}: choose that picture in Images to ${use}.`
    : `Other pictures of this prop have ${count}: choose one of them in Images to ${use}.`;
  return `This picture has no views marked, so it is made into 3D on its own. ${where}`;
}

/** A slice shorter than this on a side is enlarged a lot for the 3D generator, which reads its views at 1024 px. */
export const SMALL_VIEW_PX = 700;

/**
 * The 3D stage's note for a multi-view prop whose views are small slices of a sheet: the model gets softer
 * than from full pictures. '' when the views are big enough, whole pictures, or fewer than two.
 */
export function smallViewsHint(images) {
  const views = effectiveViews(images);
  const sides = views.filter((v) => Array.isArray(v.rect)).map((v) => Math.min(v.rect[2], v.rect[3]));
  const smallest = Math.min(...sides);
  if (views.length < 2 || !sides.length || !(smallest < SMALL_VIEW_PX)) return '';
  return `Its views are small slices (about ${Math.round(smallest)} px): the 3D model gets softer than from full `
    + 'pictures. One picture per side or higher image resolution gives more detail.';
}

/**
 * The part of the chosen picture that stands for the prop in small pictures (the props list, headings):
 * its Front slice [x, y, w, h] when the chosen picture is a sheet sliced into views - a whole turnaround
 * sheet made small is several half heads - else null (all of it). The same as projects.thumb_crop.
 */
export function thumbCrop(images) {
  const front = effectiveViews(images).find((v) => v.view === 'front');
  return front && front.src === images.chosen && Array.isArray(front.rect) ? front.rect : null;
}

/**
 * What a prop's 3D model is made from: its effective views when there are some, else its chosen picture.
 * The hub keeps it as mesh.source when it makes the model (projects.mesh_source). Nothing else of the prop
 * is in it: not the words its pictures were made from, not the Images tab it is on, not the 3D settings
 * (a new seed is asked for with "Make 3D again").
 */
export function meshSource(images) {
  const views = effectiveViews(images);
  return views.length ? `views:${viewsSignature(views)}` : images?.chosen ?? null;
}

/**
 * The prop is a tileable texture: its Tileable tick is on (Images), for a model that draws such pictures. A texture
 * is no object: no stage after Images works on it (carryPlan, retopoPlan, omniBatch) - it gets no 3D model, whatever
 * it had before the tick stays. The same as projects.tileable_prop.
 */
export const tileable = (box) => box?.images?.settings?.tileable === true && box.images.settings.qwen_model !== 'flux3';

/**
 * The prop's 3D model was made from its front view alone although its views are several: Pixal3D's
 * multi-view models were not downloaded then (adapters/pixal3d.py makes it so, and says it in
 * mesh.stats.warning).
 */
export const frontOnly = (box) => !!box.mesh?.glb && !box.mesh.imported && box.mesh.stats?.mode === 'single'
  && effectiveViews(box.images).length >= 2 && box.mesh.source === meshSource(box.images);

/**
 * The 3D model was made from the picture (or the views) the prop has now, or is the artist's own model.
 * Only that counts (mesh.source === meshSource): choosing the same picture again, another one and then the
 * first again, or saving the same views again leaves it fresh.
 * multiview: Pixal3D's multi-view models are there now - a model made from the front view alone (frontOnly)
 * is then out of date: made again, it uses every view. The same as the hub's projects.mesh_fresh.
 */
export const meshFresh = (box, { multiview = false } = {}) => !!box.mesh?.glb
  && (!!box.mesh.imported || !box.images?.chosen || !box.mesh.source || box.mesh.source === meshSource(box.images))
  && !(multiview && frontOnly(box));

/**
 * The clean meshes were made from the 3D model the prop has now, with the method it is set to now (a new
 * method makes them "to do" again; the old ones stay until new ones replace them). retopo.obj is the one the
 * prop uses. The same as the hub's projects.retopo_fresh.
 */
export const retopoFresh = (box) => !!box.retopo?.obj && !box.retopo.stale
  && (!box.retopo.source || box.retopo.source === box.mesh?.glb)
  && (box.retopo.method || 'imesh') === retopoMethod(box);

/**
 * A prop's sizes (retopo.settings.sizes, the boxes on its tile): thousands of vertices (0.25 is 250), one clean mesh
 * each, at most `count` of them, each from `min` to `max` with up to `decimals` decimals. The same as
 * projects.RETOPO_MAX_SIZES, RETOPO_MIN_SIZE, RETOPO_MAX_SIZE and RETOPO_SIZE_DECIMALS (tests/test_web.py compares
 * them).
 */
export const RETOPO_SIZES = { count: 3, min: 0.1, max: 99, decimals: 2 };

/** A size as the page and the hub write it (projects.retopo_size_name): 0.25 -> "0.25k", 2 -> "2k", 12.5 -> "12.5k". */
export const retopoSizeName = (size) => `${size}k`;

/**
 * The vertices each method places, at least and at most: a size outside them is made with the nearest count. The
 * same as its adapter's (triflow.py MIN_FACES / MAX_FACES at two triangles a vertex, lato2.py MIN_VERTICES /
 * MAX_VERTICES - the latter its models' own bound -, imesh.py MIN_TARGET_VERTICES / MAX_TARGET_VERTICES).
 */
export const RETOPO_VERTICES = { triflow: [250, 99_000], lato2: [200, 10_000], imesh: [100, 100_000] };

/**
 * The vertices `method` is asked for to make a clean mesh of `size`: its own count, or the nearest the method places.
 * The same as the adapters' faces_for / vertices_for (tests/test_web.py compares them).
 */
export function retopoVertices(method, size) {
  const [least, most] = RETOPO_VERTICES[method];
  return Math.min(Math.max(Math.round(1000 * size), least), most);
}

/** The sizes the prop asks for, in the order of its boxes. */
export const retopoSizes = (box) => (Array.isArray(box?.retopo?.settings?.sizes) ? box.retopo.settings.sizes : []);

/** The prop's clean mesh of `size` (one of retopo.variants: {size, obj, stats}), or null. */
export const retopoVariant = (box, size) => (box?.retopo?.variants || []).find((v) => v.size === size) || null;

/** The size of the clean mesh the prop uses (retopo.obj), or null. */
export const retopoUsed = (box) => (box?.retopo?.obj
  ? (box.retopo.variants || []).find((v) => v.obj === box.retopo.obj)?.size ?? null : null);

/**
 * The prop's sizes it has no clean mesh of yet: those without a variant while its clean meshes are up to date
 * (retopoFresh), else all of them. The same as the hub's projects.retopo_missing_sizes.
 */
export function retopoMissing(box) {
  const sizes = retopoSizes(box);
  return retopoFresh(box) ? sizes.filter((size) => !retopoVariant(box, size)) : sizes;
}

/**
 * The prop uses a clean mesh the artist picked (or the one of its only size), made from the 3D model it has now:
 * it is ready for the next stage.
 */
export const retopoReady = (box) => retopoFresh(box) && !!box.retopo.accepted;

/**
 * "Retopo All"'s jobs, one per method in RETOPO_ORDER (each tool loads once for its props): [['retopo', [prop id,
 * ...], tool], ...]. A prop goes when it has a 3D model (or one on its way: then every size is made again), asks
 * for a size it has no clean mesh of yet, its method's tool can take work here and none of its retopo jobs waits or
 * runs. again: every such prop that asks for a size, made or not (its clean meshes are made anew).
 */
export function retopoPlan(boxes, deps, { again = false } = {}) {
  const todo = (b) => (again || deps.pending('mesh', b) ? retopoSizes(b) : retopoMissing(b));
  const need = boxes.filter((b) => !tileable(b) && (!!b.mesh?.glb || deps.pending('mesh', b)) && !deps.pending('retopo', b)
    && deps.open('retopo', retopoMethod(b)) && todo(b).length > 0);
  return RETOPO_ORDER.map((tool) => ['retopo', need.filter((b) => retopoMethod(b) === tool).map((b) => b.id), tool])
    .filter(([, ids]) => ids.length);
}

/**
 * The clean meshes stay with the prop while it is set to another method (retopo.kept[method]), so going back to
 * its method shows them again without making them again: only those whose every variant has fewer vertices than
 * this (their stats say how many). The same number as projects.RETOPO_KEEP_VERTICES.
 */
export const RETOPO_KEEP_VERTICES = 50_000;
// What the clean meshes are in prop.retopo: these keys move to retopo.kept[method] and back.
const RETOPO_RESULT_KEYS = ['obj', 'method', 'source', 'stale', 'accepted', 'stats', 'saved_from', 'variants'];
const NO_RETOPO = { obj: null, method: null, source: null, stale: false, accepted: false, variants: [] };

/** The clean meshes are small enough to stay with the prop while it is set to another method. */
export const retopoKeepable = (result) => !!result?.obj && (result.variants || []).length > 0
  && result.variants.every((v) => typeof v?.stats?.vertices === 'number' && v.stats.vertices < RETOPO_KEEP_VERTICES);

/**
 * prop.retopo once the prop is set to `method` (the same object when its clean meshes stay): the clean meshes
 * `method` made before (retopo.kept) are the prop's again, and the ones the prop had wait there under their own
 * method when they are small enough (retopoKeepable). Ones too big stay the prop's, "to do" again, until another
 * method's replace them. The same as the hub's projects.switch_retopo_method, which does it when the method is
 * saved (the page does it at once, so the meshes are on screen without a wait).
 */
export function retopoSwitched(retopo, method) {
  const kept = { ...(retopo?.kept && typeof retopo.kept === 'object' ? retopo.kept : {}) };
  const back = kept[method]?.obj ? kept[method] : null;
  const madeBy = retopo?.obj ? retopo.method || 'imesh' : null;
  if (madeBy === method || (!back && !retopoKeepable(retopo))) return retopo;
  const rest = Object.fromEntries(Object.entries(retopo || {}).filter(([key]) => !RETOPO_RESULT_KEYS.includes(key)));
  if (retopoKeepable(retopo)) kept[madeBy] = Object.fromEntries(RETOPO_RESULT_KEYS.filter((key) => key in retopo).map((key) => [key, retopo[key]]));
  delete kept[method];
  const out = { ...rest, ...NO_RETOPO, ...(back || {}) };
  delete out.kept;
  if (Object.keys(kept).length) out.kept = kept;
  return out;
}

// ---- Omni-Texture: the 3D model's colour texture painted anew, for the props the artist chooses ---------------

/** The artist chose the prop for Omni-Texture (omnitexture.settings.chosen: the box beside its tile in that stage). */
export const omniChosen = (box) => box?.omnitexture?.settings?.chosen === true;

/**
 * The prop's result is its 3D model's own texture: no Omni-Texture job painted a view onto it, only the artist did -
 * by hand, or by masking a part and having that painted again. The same as the hub's projects.omnitexture_own.
 */
export const omniOwn = (box) => !!box?.omnitexture?.glb && !(box.omnitexture.views || []).length;

/**
 * The prop's Omni-Texture was made from the 3D model it has now (a new 3D model makes it stale). The same as the
 * hub's projects.omnitexture_fresh.
 */
export const omnitextureFresh = (box) => !!box?.omnitexture?.glb && !box.omnitexture.stale
  && (!box.omnitexture.source_mesh || box.omnitexture.source_mesh === box.mesh?.glb);

/** The prop has an Omni-Texture, painted by a job from the 3D model it has now. */
export const omniTextured = (box) => omnitextureFresh(box) && !omniOwn(box);

/**
 * Projection bakes the result's colours and Export all saves it: a fresh Omni-Texture of a chosen prop, or the 3D
 * model's own texture once the artist painted or repainted on it, chosen or not (projects.omnitexture_used).
 */
export const omnitextureUsed = (box) => omnitextureFresh(box) && (omniOwn(box)
  ? box.omnitexture.painted === true || (box.omnitexture.repaints || []).length > 0 : omniChosen(box));

/**
 * The detailed model a prop's bake reads: its Omni-Texture while used, else its 3D model (Retopo always reads the 3D
 * model). The same as the hub's projects.high_poly.
 */
export const highPoly = (box) => (omnitextureUsed(box) ? box.omnitexture.glb : box?.mesh?.glb ?? null);

/**
 * The props "Omni-Texture the chosen" sends: the chosen ones with a 3D model (or one on its way) and no fresh
 * Omni-Texture, none queued for one; again: every such chosen prop, the done ones too (painted anew).
 */
export const omniBatch = (boxes, deps, { again = false } = {}) => boxes.filter((b) => omniChosen(b) && !tileable(b)
  && !deps.pending('omnitexture', b) && (!!b.mesh?.glb || deps.pending('mesh', b)) && (again || !omniTextured(b)));

/**
 * What a prop's tile says in the Omni-Texture stage (besides a job's own state): {state, message}. Not chosen is
 * quiet: it keeps its 3D model's texture. canRun false: Omni-Texture can't run here (no Qwen-Image 2.1).
 */
export function omniStatus(box, { canRun = true, meshComing = false } = {}) {
  if (!canRun) return { state: 'idle', message: 'Omni-Texture needs Qwen-Image 2.1 as the image generator' };
  if (!omniChosen(box)) {
    return { state: 'idle', message: omnitextureUsed(box) ? 'Not chosen: its own texture, with what you painted on it'
      : 'Not chosen: keeps its 3D model’s own texture' };
  }
  if (omniTextured(box)) return { state: 'done', message: 'Has its Omni-Texture' };
  if (box.omnitexture?.glb && !omniOwn(box)) return { state: 'stale', message: '3D model changed: Omni-Texture again' };
  if (!box.mesh?.glb) {
    return meshComing ? { state: 'waiting', message: 'Chosen · waiting for its 3D model' }
      : { state: 'idle', message: 'Chosen, but it has no 3D model yet: make one in 3D' };
  }
  return { state: 'idle', message: 'Chosen: waits for “Omni-Texture the chosen”' };
}

// A repaint's edge fade, in seams' widths: where the page's slider ends and how it steps. The hub refuses more
// (adapters/omnitexture.py MAX_FADE) and the worker works these very values out ahead (FADE_STEPS).
export const MAX_FADE = 8;
export const FADE_STEP = 0.5;

const folderOf = (path) => path.slice(0, path.lastIndexOf('/'));
const pathsOf = (list) => (Array.isArray(list) ? list.filter((v) => typeof v === 'string') : []);

/**
 * What the model an Omni-Texture shows (its `omnitexture` block's glb) is made of: {base: the result its repaints
 * lie on - the job's, or one composed again from fewer of its views (block.bases) - or null when no longer known,
 * repaints: their records, oldest first, each gone onto the model of the one before it ("under")}. The hub's
 * adapters/omnitexture.py parts_of reads the same.
 */
function madeOf(block) {
  const records = new Map((Array.isArray(block.repaints) ? block.repaints : [])
    .filter((r) => r && typeof r.glb === 'string' && typeof r.picture === 'string').map((r) => [r.glb, r]));
  const repaints = [];
  let model = block.glb;
  while (records.has(model) && repaints.length < records.size) {
    repaints.unshift(records.get(model));
    model = records.get(model).under;
  }
  return { base: typeof model === 'string' ? model : null, repaints };
}

/** The record of the result the model shown lies on, when that was made again from the job's (block.bases), or undefined. */
const baseOf = (block) => {
  const { base } = madeOf(block);
  return (Array.isArray(block.bases) ? block.bases : []).find((b) => b && b.glb === base);
};

/**
 * The painted views of the model a prop's Omni-Texture (its `omnitexture` block) shows - not those taken off it -,
 * in the order they were painted: on it, or all of them hidden (omniViewsHidden). Per view its folder holds the
 * painting, what went onto the model (with the lighting taken off) and its layer: {folder, side, picture: what
 * went onto the model, albedo: the lighting-off picture or null, layer: its share of the texture alone or null,
 * sheet: the folder of the sheet it was painted in with others (what Qwen-Image was given lies there) or null,
 * together: that sheet's sides}. Of a result composed again without some (block.bases), the picture and the layer
 * are its own. Paths are relative to the prop's folder.
 */
export function omniViews(block) {
  if (!block?.glb) return [];
  const composed = baseOf(block);
  const now = composed && new Map((Array.isArray(composed.views) ? composed.views : [])
    .filter((v) => v && typeof v.folder === 'string').map((v) => [v.folder, v]));
  const inFolder = (key) => new Map(pathsOf(block[key]).map((v) => [folderOf(v), v]));
  const projected = inFolder('projected');
  const layers = inFolder('layers');
  const sideOf = (folder) => folder.split('/').pop().replace(/^\d+-/, '').replace(/_/g, ' ');
  const sheetOf = new Map();
  for (const sheet of Array.isArray(block.sheets) ? block.sheets : []) {
    if (!sheet || typeof sheet.folder !== 'string' || !Array.isArray(sheet.views)) continue;
    const together = pathsOf(sheet.views);
    for (const v of together) sheetOf.set(v, { folder: sheet.folder, together: together.map(sideOf) });
  }
  return pathsOf(block.views).filter((v) => !now || now.has(folderOf(v))).map((v) => {
    const folder = folderOf(v);
    const own = now?.get(folder);
    return {
      folder,
      side: sideOf(folder),
      picture: (own ? own.picture : projected.get(folder)) || v,
      albedo: projected.has(folder) ? `${folder}/albedo.png` : null,
      layer: (own ? own.layer : layers.get(folder)) || null,
      sheet: sheetOf.get(folder)?.folder || null,
      together: sheetOf.get(folder)?.together || null,
    };
  });
}

/**
 * The artist hid the painted views of the model shown (the eye on one hides them all): it shows the 3D model's own
 * texture in their place, under its repaints. The hub's adapters/omnitexture.py views_hidden reads the same.
 */
export const omniViewsHidden = (block) => !!block?.glb && baseOf(block)?.hidden === true;

/**
 * The eye on a prop's painted views (stages/omnitexture.js) is there: to hide them while the 3D model is the one they
 * were painted on (the hub puts that model's own texture in their place), to show them again always.
 */
export const omniViewsEye = (box) => omniViewsHidden(box?.omnitexture) || omnitextureFresh(box);

/**
 * The repaints the model shown holds - not those of steps taken back with Ctrl+Z, nor those taken off -, oldest
 * first: {glb: the model it made, picture, layer (or null), fade, fadable: the model it went onto is known, so the
 * hub can put it on again, hidden: the artist hid it (the eye on it) - it is in its place, and not on the model}.
 * Paths are relative to the prop's folder.
 */
export function omniRepaints(block) {
  if (!block?.glb) return [];
  return madeOf(block).repaints.map((r) => ({
    glb: r.glb,
    picture: r.picture,
    layer: typeof r.layer === 'string' ? r.layer : null,
    fade: typeof r.fade === 'number' ? r.fade : 1,
    fadable: typeof r.under === 'string',
    hidden: r.hidden === true,
  }));
}

// The step that is no model: a result taken off its prop as a whole. While it is the step shown the prop has no
// result (every rule here reads its empty `glb` so) and shows its 3D model. The hub's projects.TAKEN_OFF.
export const TAKEN_OFF = '';

/** The prop's result was taken off as a whole, and Ctrl+Z still leads back to it. */
export const omniTakenOff = (box) => box?.omnitexture?.glb === TAKEN_OFF
  && (box.omnitexture.steps || []).includes(TAKEN_OFF);

/**
 * The × on a prop's painted pictures (stages/omnitexture.js), which takes one off the model shown, and the one
 * beside their heading, which takes the result off as a whole: {views, repaints: whether theirs have one, whole,
 * waits: whether it waits now}. Offered where Ctrl+Z can bring the picture back (canRun: Omni-Texture can run here,
 * the mask marker's walk is there), never on the only view, and on a view only while the 3D model is the one the
 * views were painted on (the hub composes the result again from it); the whole of a result that has painted views.
 * It waits while an Omni-Texture job of the prop waits or runs (`job`) or its Omni-Texture is being changed
 * (`changing`: a fade, a picture hidden or taken off). No view is taken off while they are hidden.
 */
export function omniTakeOff(box, { canRun = true, job = false, changing = false } = {}) {
  const views = omniViews(box?.omnitexture).length;
  return {
    views: canRun && views > 1 && omnitextureFresh(box) && !omniViewsHidden(box.omnitexture),
    repaints: canRun,
    whole: canRun && views > 0,
    waits: !!job || !!changing,
  };
}

/**
 * The bake was made from the detailed model (highPoly) and the retopo the prop has now: a new Omni-Texture, or one
 * chosen or left out, makes it old (an older bake that names no source_mesh counts as made from it).
 */
export const bakeFresh = (box) => !!box.projection?.glb && !box.projection.stale
  && (!box.projection.source_mesh || box.projection.source_mesh === highPoly(box));

const FRESH = { mesh: meshFresh, retopo: retopoFresh, projection: bakeFresh };
// The work a call to action queues on its way down: Images' "Generate 3D" and Omni-Texture's "Retopologize" make
// the 3D models that are missing, Retopo's "Project textures" bakes. Omni-Texture and Retopo are in no carry: only
// their own buttons start them ("Omni-Texture the chosen", "Retopo All") - going on to a later stage never does, the
// artist who moved on meant to. A prop chosen for Omni-Texture but not painted is baked with the texture it has, and
// a job the artist did start is queued before whatever a later button adds, so the bake that follows reads its result.
const CARRIES = { mesh: 'mesh', retopo: 'mesh', projection: 'projection' };
// What a prop needs before a stage's work can be done for it, besides lacking a fresh result there: what that work
// reads, there or on its way (pending). An imported prop brings its own 3D model; Retopo reads the 3D model, never
// the Omni-Texture; a bake reads the clean mesh the prop uses, with UVs.
const READY = {
  mesh: (b) => !!b.images?.chosen,
  projection: (b, pending) => pending('retopo', b) || (retopoFresh(b) && b.retopo.stats?.has_uv !== false),
};

/**
 * The jobs that take `boxes` (props) down to the stage `upTo` ('mesh' | 'retopo' | 'projection'):
 * [[stage, [prop id, ...], tool]], or [] - the props that lack the work of the stage the carry does (CARRIES), can
 * have it (READY: objects only, never a tileable texture), and have none of it queued. pending(stage, prop): a job of that stage is queued or running for
 * the prop; open(stage, tool): that tool can take work (one that can't run on this computer queues nothing);
 * multiview: Pixal3D's multi-view models are there (meshFresh).
 */
export function carryPlan(upTo, boxes, deps) {
  const stage = CARRIES[upTo];
  const tool = STAGE_TOOLS[stage][0];
  if (!deps.open(stage, tool)) return [];
  const ids = boxes.filter((b) => !tileable(b) && READY[stage](b, deps.pending) && !FRESH[stage](b, deps)
    && !deps.pending(stage, b)).map((b) => b.id);
  return ids.length ? [[stage, ids, tool]] : [];
}

/**
 * The jobs that work on the props in the stage `stage` itself, without going on: carryPlan's work of
 * `stage` alone. [[stage, [prop id, ...], tool]], or [].
 */
export function rowPlan(stage, boxes, deps) {
  return carryPlan(stage, boxes, deps).filter(([s]) => s === stage);
}

/** How many props a plan's jobs of `stage` work on. */
export const planCount = (plan, stage) => plan.reduce((n, [s, ids]) => n + (s === stage ? ids.length : 0), 0);

const props = (n) => `${n} ${n === 1 ? 'prop' : 'props'}`;
const DOES = {
  mesh: (n) => `makes the 3D model${n === 1 ? '' : 's'} of ${props(n)}`,
  retopo: (n) => `retopologizes ${props(n)}`,
  projection: (n) => `bakes ${props(n)}`,
};

/**
 * What a carry plan (carryPlan) does, in one sentence (the tooltip of a stage's own button), so the work
 * it queues before its own is never a surprise: "Makes the 3D models of 2 props first, then
 * retopologizes 3 props." '' for an empty plan. name(tool): a tool's name; with it, a stage whose jobs
 * run on several tools says their groups in order: "Retopologizes 5 props (3 with LATO.2, then 2 with
 * TriFlow)." (each tool loads once for its props).
 */
export function planSummary(plan, name = null) {
  // A stage's jobs said once (Retopo has one per method).
  const stages = [...new Set(plan.map(([stage]) => stage))];
  const groups = (stage) => {
    const jobs = plan.filter(([s, ids, tool]) => s === stage && tool && ids.length);
    if (!name || jobs.length < 2) return '';
    return ` (${jobs.map(([, ids, tool]) => `${ids.length} with ${name(tool)}`).join(', then ')})`;
  };
  const parts = stages.map((stage) => DOES[stage](planCount(plan, stage)) + groups(stage));
  if (!parts.length) return '';
  const last = parts.pop();
  const text = !parts.length ? last : `${parts.join(', ')}${parts.length === 1 ? ' first' : ''}, then ${last}`;
  return `${capital(text)}.`;
}

/** Some prop has a result of `upTo` (or one on its way), or going down there with every prop queues work. */
export function reaches(upTo, boxes, deps) {
  return boxes.some((b) => FRESH[upTo](b, deps) || deps.pending(upTo, b)) || carryPlan(upTo, boxes, deps).length > 0;
}

// ---- names -----------------------------------------------------------------------------------------

// The same as projects.box_name_from_prompt, _NAME_STOP_WORDS and _QUALITY_TAGS (code/tests/test_web.py
// compares them).
const STOP_WORDS = new Set(['with', 'on', 'in', 'for', 'and', 'at', 'from', 'made', 'that', 'wearing', 'holding',
  'against', 'under', 'over', 'by', 'of']);
const QUALITY_TAGS = new Set(['masterpiece', 'best quality', 'high quality', 'highest quality', 'top quality',
  'ultra detailed', 'highly detailed', 'detailed', 'highres', 'absurdres', '4k', '8k', 'hd', 'uhd', 'hdr',
  'photorealistic', 'realistic', '3d', '3d render', 'render', 'game asset', 'concept art', 'white background',
  'simple background']);
const capital = (text) => text.charAt(0).toUpperCase() + text.slice(1);

/**
 * The name the hub gives a new prop without one: the first words of the prompt that name a thing.
 * "a wooden barrel with iron bands, game asset" -> "Wooden barrel";
 * "masterpiece, best quality, (wooden barrel:1.3)" -> "Wooden barrel"; "" when no part does.
 */
export function nameFromPrompt(prompt) {
  const text = String(prompt || '')
    .replace(/<[^>]*>/g, ' ')                           // <lora:name:0.8>
    .replace(/:\s*-?\d+(?:\.\d+)?|[()[\]{}]/g, ' ');    // (words:1.2)
  for (const part of text.split(/[,.;!?|\n]/)) {
    const words = part.split(/\s+/).filter(Boolean);
    while (words.length && /^(a|an|the)$/i.test(words[0])) words.shift();
    if (!words.length || QUALITY_TAGS.has(words.join(' ').toLowerCase())) continue;
    const cut = words.findIndex((w, i) => i && STOP_WORDS.has(w.toLowerCase()));
    return capital(words.slice(0, Math.min(cut < 0 ? words.length : cut, 4)).join(' ').slice(0, 40).trim());
  }
  return '';
}
