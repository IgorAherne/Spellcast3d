// The painter's strokes, without the GPU (paint3d.js paints them on the model): a pen's pressure as the browser
// reports it, the dabs a stroke is made of, the brush size, the undo history's memory, and the patch a save sends.
//
// A stroke is a row of round dabs along the pointer's smoothed path, so close together that the edge stays round.
// Each dab's radius and opacity are interpolated between the pen's readings along the path: a change of pressure
// shows as a taper, never as a step, and a light touch tapers to a point. Windows Ink's first reading of a stroke
// can be off (0, or a spike): it takes the second reading's value. A reading of exactly 0.5 on every sample is a
// pen that reports no pressure (the Pointer Events rule for such hardware); it then paints at full strength.

export const MIN_SIZE = 2;          // brush diameter, CSS px
export const MAX_SIZE = 300;
export const DEFAULT_SIZE = 40;
const SWEEP_PX = 250;               // Shift + this far sideways goes through the whole size range
const SMOOTH_MS = 8;                // the pressure follows the pen within a few samples
const SPIKE = 0.2;                  // a first reading this much above the second is a spike
const NO_PRESSURE_SAMPLES = 6;      // this many readings of exactly 0.5: a pen without pressure

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** The slider's position (0..1000) of a brush size, on a log scale: small sizes get room. */
export const sizeToPos = (size) => Math.round((Math.log(size / MIN_SIZE) / Math.log(MAX_SIZE / MIN_SIZE)) * 1000);
export const posToSize = (pos) => Math.round(MIN_SIZE * (MAX_SIZE / MIN_SIZE) ** (clamp(pos, 0, 1000) / 1000));

/** The size Shift + `dx` screen pixels sideways give from `start` (the whole range in SWEEP_PX, on the same log scale). */
export function dragSize(start, dx) {
  return posToSize(sizeToPos(clamp(start, MIN_SIZE, MAX_SIZE)) + (dx * 1000) / SWEEP_PX);
}

/**
 * The pens' pressure, as far as the page can tell: read(e) -> 0..1, or null for a mouse, a finger, or a pen that
 * reports no pressure (they paint at full strength). One per page: whether the pen reports pressure is learnt from
 * its strokes (start() and end() around each one).
 */
export function createPressureReader() {
  let penHasPressure = true;
  let flat = 0;          // readings of exactly 0.5 in this stroke
  let other = false;     // a reading that is not 0.5
  const settle = () => { if (!other && flat >= NO_PRESSURE_SAMPLES) penHasPressure = false; };
  return {
    start() { flat = 0; other = false; },
    read(e) {
      if (e.pointerType !== 'pen') return null;
      const p = e.pressure;
      if (p === 0.5) flat++;
      else if (p > 0) { other = true; penHasPressure = true; }
      settle();
      return penHasPressure && typeof p === 'number' ? clamp(p, 0, 1) : null;
    },
    end: settle,
    get penHasPressure() { return penHasPressure; },
  };
}

const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, p: (a.p + b.p) / 2 });

/**
 * One stroke: add(x, y, pressure | null, ms) -> the dabs it completes [{x, y, r, a}], end() -> the last ones.
 * radius: the brush radius in the caller's pixels; bySize / byFade: the pressure scales the radius / the opacity.
 * The path is the one strokes.js draws on the Images picture (curves from the middle before a point to the middle
 * after it, bent at the point); a part is placed once the point after it is known, so a dab never moves.
 */
export function createStroke({ radius, bySize = true, byFade = false, spacing = 0.1, minSpacing = 0.5 }) {
  const points = [];       // {x, y, p}: the readings kept (pressure smoothed)
  let first = null;        // the first reading, held until the second says whether it was off
  let smoothed = null;
  let lastMs = 0;
  let nextAt = 0;          // distance along the path where the next dab goes
  let walked = 0;          // distance walked so far
  let placed = 0;          // the parts placed
  let last = null;         // the end of the path walked so far

  const radiusAt = (p) => radius * (bySize ? p : 1);
  const alphaAt = (p) => (byFade ? p : 1);
  const gapAt = (p) => Math.max(minSpacing, spacing * radiusAt(p));

  /** Dabs along the quadratic curve from a to c bent at b (each {x, y, p}), from where the last part ended. */
  function walk(a, b, c, out) {
    const chord = Math.hypot(b.x - a.x, b.y - a.y) + Math.hypot(c.x - b.x, c.y - b.y);
    const steps = Math.max(1, Math.min(512, Math.ceil(chord / 2)));
    let prev = a;
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      const u = 1 - t;
      const q = {
        x: u * u * a.x + 2 * u * t * b.x + t * t * c.x,
        y: u * u * a.y + 2 * u * t * b.y + t * t * c.y,
        p: u * u * a.p + 2 * u * t * b.p + t * t * c.p,
      };
      const len = Math.hypot(q.x - prev.x, q.y - prev.y);
      while (len > 0 && walked + len >= nextAt) {
        const f = (nextAt - walked) / len;
        const p = prev.p + (q.p - prev.p) * f;
        out.push({ x: prev.x + (q.x - prev.x) * f, y: prev.y + (q.y - prev.y) * f, r: radiusAt(p), a: alphaAt(p) });
        nextAt += gapAt(p);
      }
      walked += len;
      prev = q;
    }
    last = c;
  }

  function keep(x, y, p, ms) {
    const value = p == null ? 1 : p;
    const k = smoothed == null ? 1 : 1 - Math.exp(-Math.max(0, ms - lastMs) / SMOOTH_MS);
    smoothed = smoothed == null ? value : smoothed + (value - smoothed) * k;
    lastMs = ms;
    const prev = points[points.length - 1];
    if (prev && Math.hypot(x - prev.x, y - prev.y) < 0.5) {
      prev.p = smoothed;     // a hand at rest adds no point: its pressure still counts
      return;
    }
    points.push({ x, y, p: smoothed });
  }

  /** The parts that the points now known complete (part k needs point k + 1). */
  function place() {
    const out = [];
    if (placed === 0 && points.length >= 1) {
      const p0 = points[0];
      out.push({ x: p0.x, y: p0.y, r: radiusAt(p0.p), a: alphaAt(p0.p) });
      nextAt = gapAt(p0.p);
      last = p0;
      placed = 1;
    }
    for (; placed <= points.length - 2; placed++) {
      const k = placed;
      const from = k === 1 ? points[0] : mid(points[k - 1], points[k]);
      walk(from, points[k], mid(points[k], points[k + 1]), out);
    }
    return out;
  }

  return {
    add(x, y, p, ms = 0) {
      if (first === null) {                            // held until the second reading
        first = { x, y, p, ms };
        return [];
      }
      if (first) {
        let p0 = first.p;
        if (p0 != null && p != null && (p0 === 0 || p0 > p + SPIKE)) p0 = p;
        keep(first.x, first.y, p0, first.ms);
        first = false;
      }
      keep(x, y, p, ms);
      return place();
    },
    end() {
      if (first) {                                     // a tap: one dab (a pen that said 0 paints a half-strength dot)
        keep(first.x, first.y, first.p === 0 ? 0.5 : first.p, first.ms);
        first = false;
      }
      const out = place();
      const n = points.length;
      if (n >= 2 && last) walk(last, points[n - 1], points[n - 1], out);
      return out;
    },
    get length() { return walked; },
  };
}

/**
 * What a painting changed against the texture its files hold (`now`, `was`: RGBA8 of one size, as the card holds
 * them): the texels whose colour differs, with their colour now and alpha 255, every other texel all 0 -> RGBA8, or
 * null when no texel differs. Painting never changes alpha: only the colour is compared.
 */
export function patchOf(now, was) {
  let patch = null;
  for (let i = 0; i < now.length; i += 4) {
    if (now[i] === was[i] && now[i + 1] === was[i + 1] && now[i + 2] === was[i + 2]) continue;
    patch ||= new Uint8Array(now.length);
    patch[i] = now[i];
    patch[i + 1] = now[i + 1];
    patch[i + 2] = now[i + 2];
    patch[i + 3] = 255;
  }
  return patch;
}

/** The colour of RGBA8 texels (the eyedropper's few around a point) as '#rrggbb': their average, alpha left out. */
export function texelHex(data) {
  const n = Math.floor(data.length / 4);
  if (!n) return null;
  const sum = [0, 0, 0];
  for (let i = 0; i < n * 4; i += 4) for (let c = 0; c < 3; c++) sum[c] += data[i + c];
  return `#${sum.map((s) => Math.round(s / n).toString(16).padStart(2, '0')).join('')}`;
}

export const DAB_REACH = 2;         // a dab changes the stroke's mask this many pixels past its radius, at most

/** The screen box ({x0, y0, x1, y1}, px) of `dabs` ([{x, y, r}]) with their reach, joined to `box` when given. */
export function dabsBox(dabs, box = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity }) {
  for (const d of dabs) {
    box.x0 = Math.min(box.x0, d.x - d.r - DAB_REACH);
    box.x1 = Math.max(box.x1, d.x + d.r + DAB_REACH);
    box.y0 = Math.min(box.y0, d.y - d.r - DAB_REACH);
    box.y1 = Math.max(box.y1, d.y + d.r + DAB_REACH);
  }
  return box;
}

/**
 * Where a mesh's corners lie on the screen: `position` (x, y, z per corner, in the mesh's own space) through `matrix`
 * (the camera's view-projection times the mesh's world matrix: 16 numbers, column by column) onto a `width` x
 * `height` px screen, y down -> `into` (Float32Array, 3 per corner): x px, y px, w (w <= 0: behind the camera).
 */
export function screenCorners(position, matrix, width, height, into = new Float32Array(position.length)) {
  const [m0, m1, , m3, m4, m5, , m7, m8, m9, , m11, m12, m13, , m15] = matrix;
  for (let i = 0; i < position.length; i += 3) {
    const x = position[i];
    const y = position[i + 1];
    const z = position[i + 2];
    const w = m3 * x + m7 * y + m11 * z + m15;
    into[i] = ((m0 * x + m4 * y + m8 * z + m12) / w * 0.5 + 0.5) * width;
    into[i + 1] = (0.5 - (m1 * x + m5 * y + m9 * z + m13) / w * 0.5) * height;
    into[i + 2] = w;
  }
  return into;
}

const FOOT_CELL = 8;                // px: the cells a stroke's footprint is kept in

/**
 * Where one stroke's dabs have been on a `width` x `height` px screen, in cells of FOOT_CELL px, to tell the
 * triangles under the stroke from the rest - no dab reaches those, so drawing them would change nothing:
 *   cells(screen, into) -> per triangle of `screen` (screenCorners) the cells its box lies in (Uint16Array, 4 per
 *     triangle: column and row of the first, of the last); a triangle off the screen has none, one with a corner
 *     behind the camera all of them (it may show anywhere);
 *   add(dabs): the cells under these dabs ([{x, y, r}], their reach included) are the stroke's too;
 *   under(cells, into) -> how many corners it wrote into `into` (an index with room for every corner): those of the
 *     triangles with a cell of the stroke's, in the triangles' own order.
 */
export function createFootprint(width, height) {
  const columns = Math.max(1, Math.ceil(width / FOOT_CELL));
  const rows = Math.max(1, Math.ceil(height / FOOT_CELL));
  const marked = new Uint8Array(columns * rows);
  const sums = new Int32Array((columns + 1) * (rows + 1));     // marked cells above and left of each cell corner
  const column = (x) => Math.min(columns - 1, Math.max(0, Math.floor(x / FOOT_CELL)));
  const row = (y) => Math.min(rows - 1, Math.max(0, Math.floor(y / FOOT_CELL)));
  return {
    cells(screen, into = new Uint16Array((screen.length / 9) * 4)) {
      for (let s = 0, t = 0; s + 8 < screen.length; s += 9, t += 4) {
        let x0 = -Infinity;
        let y0 = -Infinity;
        let x1 = Infinity;
        let y1 = Infinity;
        if (screen[s + 2] > 0 && screen[s + 5] > 0 && screen[s + 8] > 0) {
          x0 = Math.min(screen[s], screen[s + 3], screen[s + 6]);
          y0 = Math.min(screen[s + 1], screen[s + 4], screen[s + 7]);
          x1 = Math.max(screen[s], screen[s + 3], screen[s + 6]);
          y1 = Math.max(screen[s + 1], screen[s + 4], screen[s + 7]);
        }
        const off = !(x1 >= 0 && y1 >= 0 && x0 < width && y0 < height);
        into[t] = off ? 1 : column(x0);
        into[t + 1] = off ? 1 : row(y0);
        into[t + 2] = off ? 0 : column(x1);
        into[t + 3] = off ? 0 : row(y1);
      }
      return into;
    },
    add(dabs) {
      for (const d of dabs) {
        const reach = d.r + DAB_REACH;
        if (d.x + reach < 0 || d.y + reach < 0 || d.x - reach >= width || d.y - reach >= height) continue;
        for (let r = row(d.y - reach); r <= row(d.y + reach); r++) marked.fill(1, r * columns + column(d.x - reach), r * columns + column(d.x + reach) + 1);
      }
      for (let r = 0; r < rows; r++) {
        for (let c = 0, inRow = 0; c < columns; c++) {
          inRow += marked[r * columns + c];
          sums[(r + 1) * (columns + 1) + c + 1] = sums[r * (columns + 1) + c + 1] + inRow;
        }
      }
    },
    under(cells, into) {
      const wide = columns + 1;
      let n = 0;
      for (let t = 0, corner = 0; t < cells.length; t += 4, corner += 3) {
        const c0 = cells[t];
        const c1 = cells[t + 2] + 1;
        if (c1 <= c0) continue;
        const above = cells[t + 1] * wide;
        const below = (cells[t + 3] + 1) * wide;
        if (sums[below + c1] - sums[above + c1] - sums[below + c0] + sums[above + c0] === 0) continue;
        into[n] = corner;
        into[n + 1] = corner + 1;
        into[n + 2] = corner + 2;
        n += 3;
      }
      return n;
    },
  };
}

/**
 * The texture tiles a stroke may have changed, for undo: every triangle in front of the camera that faces it (or
 * any, `twoSided`) and whose screen box meets the stroke's (`box`, px, the brush radius included) marks the tiles
 * under its UV box, grown by `margin` texels (the painter paints that far into the gutter). uv: per corner (u, v)
 * in 0..1, 2 per corner; screen: per corner (x px, y px, w), 3 per corner (w <= 0: behind the camera); `tiles` per
 * side of a `size` px texture -> Uint8Array(tiles * tiles), 1 = changed.
 */
export function dirtyTiles({ uv, screen, box, size, tiles, margin = 0, twoSided = false }) {
  const out = new Uint8Array(tiles * tiles);
  const tile = size / tiles;
  const corners = uv.length / 2;
  for (let c = 0; c + 2 < corners; c += 3) {
    const s = c * 3;
    const w0 = screen[s + 2];
    const w1 = screen[s + 5];
    const w2 = screen[s + 8];
    if (w0 <= 0 && w1 <= 0 && w2 <= 0) continue;
    if (w0 > 0 && w1 > 0 && w2 > 0) {
      const x0 = screen[s]; const y0 = screen[s + 1];
      const x1 = screen[s + 3]; const y1 = screen[s + 4];
      const x2 = screen[s + 6]; const y2 = screen[s + 7];
      if (Math.max(x0, x1, x2) < box.x0 || Math.min(x0, x1, x2) > box.x1
        || Math.max(y0, y1, y2) < box.y0 || Math.min(y0, y1, y2) > box.y1) continue;
      // Screen y grows downwards: a front face (counter-clockwise in 3D) turns clockwise on screen.
      if (!twoSided && (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0) > 0) continue;
    }
    // The texels under the UV box (their centres inside it), grown by the margin, then their tiles.
    const u = c * 2;
    const first = (a, b, d) => Math.floor((Math.floor(Math.min(a, b, d) * size - margin)) / tile);
    const last = (a, b, d) => Math.floor((Math.ceil(Math.max(a, b, d) * size + margin) - 1) / tile);
    const tx0 = first(uv[u], uv[u + 2], uv[u + 4]);
    const tx1 = last(uv[u], uv[u + 2], uv[u + 4]);
    const ty0 = first(uv[u + 1], uv[u + 3], uv[u + 5]);
    const ty1 = last(uv[u + 1], uv[u + 3], uv[u + 5]);
    for (let ty = Math.max(0, ty0); ty <= Math.min(tiles - 1, ty1); ty++) {
      for (let tx = Math.max(0, tx0); tx <= Math.min(tiles - 1, tx1); tx++) out[ty * tiles + tx] = 1;
    }
  }
  return out;
}

/** The changed tiles as few rectangles (texels): runs along each row, joined with the same run of the rows below. */
export function tileRects(flags, tiles, size) {
  const tile = size / tiles;
  const open = new Map();     // "x0,x1" -> a rectangle still growing downwards
  const rects = [];
  for (let ty = 0; ty < tiles; ty++) {
    const runs = [];
    for (let tx = 0; tx < tiles;) {
      if (!flags[ty * tiles + tx]) { tx++; continue; }
      const x0 = tx;
      while (tx < tiles && flags[ty * tiles + tx]) tx++;
      runs.push(`${x0},${tx}`);
    }
    for (const [key, rect] of open) {
      if (!runs.includes(key)) {
        rects.push(rect);
        open.delete(key);
      }
    }
    for (const key of runs) {
      const rect = open.get(key);
      if (rect) {
        rect.h += tile;
      } else {
        const [x0, x1] = key.split(',').map(Number);
        open.set(key, { x: x0 * tile, y: ty * tile, w: (x1 - x0) * tile, h: tile });
      }
    }
  }
  rects.push(...open.values());
  return rects;
}

/**
 * Undo and redo of strokes, each a step holding what it changed (any object with `bytes`). The newest `keep` steps
 * always stay; older ones go, the oldest first, while all steps together pass `budget` bytes or number more than
 * `max`. add() returns the steps that went: those pushed out, and the ones undone before (a new stroke after an
 * undo: they can no longer come back).
 */
export function createUndo({ keep = 5, budget = 384 * 1024 * 1024, max = 60 } = {}) {
  let steps = [];
  let undone = [];
  const total = () => [...steps, ...undone].reduce((n, s) => n + (s.bytes || 0), 0);
  return {
    add(step) {
      const dropped = undone;
      undone = [];
      steps.push(step);
      while (steps.length > keep && (total() > budget || steps.length > max)) dropped.push(steps.shift());
      return dropped;
    },
    undo() {
      const s = steps.pop();
      if (s) undone.push(s);
      return s || null;
    },
    redo() {
      const s = undone.pop();
      if (s) steps.push(s);
      return s || null;
    },
    get canUndo() { return steps.length > 0; },
    get canRedo() { return undone.length > 0; },
    get newest() { return steps[steps.length - 1] || null; },
    get nextRedo() { return undone[undone.length - 1] || null; },
    get bytes() { return total(); },
    /** The steps as they stand (the same objects), for the history of the same texture loaded again to take. */
    get all() { return { steps: [...steps], undone: [...undone] }; },
    take(all) {
      steps = [...all.steps];
      undone = [...all.undone];
    },
    clear() {
      const all = [...steps, ...undone];
      steps = [];
      undone = [];
      return all;
    },
  };
}
