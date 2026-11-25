// A prop's small picture, the same wherever the page shows one (a tile of the props list, a stage's heading,
// the views next to a hovered tile): which file it is and which part of it (propPicture), and a box that
// shows a picture, or a rectangular part of it, by CSS alone (croppedPicture). So a sheet sliced into views
// shows its Front slice the moment it is sliced, without waiting for a thumbnail to be drawn and stored:
// the hub's thumbnail only makes the page load faster.

import { fileUrl } from './api.js';
import { h } from './dom.js';
import { bustFor, projectId, thumbOf } from './store.js';
import { thumbCrop } from './stages/rules.js';

/**
 * A prop's small picture, from the prop as the store has it this moment: {url, crop}, or null while it has
 * none. The hub's thumbnail (crop: null) only while it shows the prop's chosen picture and the part of it
 * that stands for the prop (store.thumbOf); else the chosen picture itself, with the part of it to show
 * (crop: [x, y, w, h] in its pixels, the Front slice of a sheet sliced into views: rules.thumbCrop; null: all
 * of it). A prop made from the artist's own 3D model: its model's snapshot. Whatever thumbnail was left on
 * the prop, of whatever picture: asked here, the answer is never a picture other than the chosen one.
 */
export function propPicture(prop, pid = projectId()) {
  const small = (stage) => {
    const t = thumbOf(prop, stage);
    return t ? { url: fileUrl(pid, prop.id, t.path, t.v), crop: null } : null;
  };
  if (prop?.images?.chosen) {
    return small('images')
      || { url: fileUrl(pid, prop.id, prop.images.chosen, bustFor('images', prop.id)), crop: thumbCrop(prop.images) };
  }
  return prop?.mesh?.glb ? small('mesh') : null;
}

/**
 * Where a whole picture of `natural` size ({w, h}) goes in a box that shows its part `rect` ([x, y, w, h])
 * alone: {width, height, left, top} in % of the box. (The same as importpanel.cropBox.)
 */
export function cropBox(rect, natural) {
  const [x, y, w, hh] = rect;
  return { width: (natural.w / w) * 100, height: (natural.h / hh) * 100, left: (-x / w) * 100, top: (-y / hh) * 100 };
}

/**
 * How a part of a picture w x hh px sits in a box `ratio` (width / height) wide: the side of it that fills
 * the box ('width' | 'height'), the other one running over it ('cover') or staying inside it ('contain').
 */
export const fillSide = (w, hh, ratio, fit) => ((fit === 'cover') === (w / hh >= ratio) ? 'height' : 'width');

/**
 * The style of the frame that holds a picture's part w x hh px in such a box: of the part's shape, in the box's
 * middle, and cutting the picture - where the frame does not fill the box ('contain'), nothing around the part shows.
 */
export const frameStyle = (w, hh, ratio, fit) => 'position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);'
  + `overflow:hidden;aspect-ratio:${w} / ${hh};${fillSide(w, hh, ratio, fit)}:100%`;

const IMG = 'position:absolute;display:block;max-width:none;max-height:none;user-select:none;';
const part = (crop) => (Array.isArray(crop) && crop[2] > 0 && crop[3] > 0 ? crop : null);

/**
 * A box that shows the picture at a URL: all of it, or its part `crop` ([x, y, w, h] in the picture's
 * pixels) alone. Returns {el, update(url, crop = null)}; update(null) empties it. `el` is a <span> with
 * `class`, which must give it its place and size and a `position` (the picture is laid out inside it).
 * opts: ratio: the box's width / height (a part is fitted by it); fit: 'cover' (the box is filled, as
 * object-fit does) or 'contain'; lazy: the first picture loads when the box scrolls into view; onError():
 * the picture could not be loaded (the box is empty then).
 * The picture on screen stays until the next one has loaded, and is replaced in one go: never an empty
 * box or a half-placed picture in between (a URL that only got another cache stamp does not flicker).
 * Another part of the file on screen shows at once: nothing loads.
 */
export function croppedPicture({ class: cls = '', ratio = 1, fit = 'cover', lazy = false, onError = null } = {}) {
  const el = h('span', { class: cls, style: 'overflow: hidden' });
  let shown = null;    // {url, crop, frame, img}: the picture on screen
  let coming = null;   // the one loading, hidden, until it takes its place

  /** Lays a loaded picture out: all of it as object-fit has it, or a frame of its part's shape holding it. */
  function place({ frame, img, crop }) {
    const natural = { w: img.naturalWidth, h: img.naturalHeight };
    if (!crop || !natural.w || !natural.h) {
      frame.style.cssText = 'position:absolute;inset:0';
      img.style.cssText = `${IMG}left:0;top:0;width:100%;height:100%;object-fit:${fit}`;
      return;
    }
    const [, , w, hh] = crop;
    frame.style.cssText = frameStyle(w, hh, ratio, fit);
    const c = cropBox(crop, natural);
    img.style.cssText = `${IMG}left:${c.left}%;top:${c.top}%;width:${c.width}%;height:${c.height}%`;
  }

  function arrived(layer) {
    if (coming !== layer) return;   // another picture was asked for meanwhile
    coming = null;
    place(layer);
    shown?.frame.remove();
    shown = layer;
  }

  function failed(layer) {
    if (coming !== layer) return;
    coming = null;
    layer.frame.remove();
    // What the box showed until now is not the picture any more: better none than another one.
    shown?.frame.remove();
    shown = null;
    onError?.();
  }

  function update(url, crop = null) {
    const rect = part(crop);
    if (!url) {
      coming?.frame.remove();
      shown?.frame.remove();
      coming = null;
      shown = null;
      return;
    }
    if (shown?.url === url) {   // the file on screen: nothing to load
      coming?.frame.remove();
      coming = null;
      if (String(shown.crop) !== String(rect)) {
        shown.crop = rect;
        place(shown);
      }
      return;
    }
    if (coming?.url === url) {
      coming.crop = rect;
      return;
    }
    coming?.frame.remove();
    const img = h('img', { alt: '', draggable: 'false', decoding: 'async', loading: lazy && !shown ? 'lazy' : null });
    img.style.cssText = `${IMG}left:0;top:0;width:100%;height:100%`;
    const frame = h('span', { style: 'position:absolute;inset:0;visibility:hidden' }, img);
    const layer = { url, crop: rect, frame, img };
    coming = layer;
    img.addEventListener('load', () => {
      // Decoded before it is shown: no frame without a picture between the old one and this one.
      const ready = img.decode ? img.decode().catch(() => {}) : Promise.resolve();
      ready.then(() => arrived(layer));
    });
    img.addEventListener('error', () => failed(layer));
    el.append(frame);
    img.src = url;
  }

  return { el, update };
}
