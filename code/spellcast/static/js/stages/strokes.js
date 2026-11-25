// Strokes painted by hand on a canvas, the markers' over the Images workbench's picture (marks.js): the points a stroke
// keeps (a hand at rest adds none), the parts it is painted in - the same while it is drawn and when it is painted
// again after an undo - its width under a pen's pressure, the steps to undo and redo, and the keys for them.

const POINT_GAP = 1.5;                     // picture pixels between two points of a stroke: a hand at rest adds none

const round1 = (v) => Math.round(v * 10) / 10;

/** How wide a brush of `size` paints: a pen's pressure thins the line, a mouse or a finger paints it full. */
export function brushWidth(size, pointerType = 'mouse', pressure = 0.5) {
  if (pointerType !== 'pen' || !(pressure > 0)) return size;
  return Math.max(1, round1(size * (0.25 + 0.75 * Math.min(1, pressure))));
}

/** Adds the point `p` ({x, y, w}) to a stroke's points, unless the hand has hardly moved -> whether it did. */
export function addPoint(points, p, gap = POINT_GAP) {
  const last = points[points.length - 1];
  if (last && Math.hypot(p.x - last.x, p.y - last.y) < gap) return false;
  points.push(p);
  return true;
}

const middle = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

/**
 * Part `k` of a stroke through `points`: 0 is the dot it starts with; 1 .. points.length - 2 are curves from
 * the middle before a point to the middle after it, bent at the point, so the line turns instead of breaking.
 * A part never changes once the point after it is there: a stroke is painted part by part while it is drawn.
 */
export function strokePart(points, k) {
  if (k === 0) return { dot: true, to: points[0], width: points[0].w };
  const from = k === 1 ? points[0] : middle(points[k - 1], points[k]);
  return { from, ctrl: points[k], to: middle(points[k], points[k + 1]), width: points[k].w };
}

/** The straight end of a finished stroke, up to its last point (none for a dot). */
export function strokeTail(points) {
  const n = points.length;
  if (n < 2) return null;
  const last = points[n - 1];
  return { from: n === 2 ? points[0] : middle(points[n - 2], last), ctrl: last, to: last, width: last.w };
}

/** Every part of a finished stroke, in the order they are painted. */
export function strokeParts(points) {
  if (!points.length) return [];
  const parts = [strokePart(points, 0)];
  for (let k = 1; k <= points.length - 2; k++) parts.push(strokePart(points, k));
  const tail = strokeTail(points);
  if (tail) parts.push(tail);
  return parts;
}

/** Paints one part of a stroke in `colour` on a canvas's 2D context. */
export function paintPart(ctx, part, colour) {
  ctx.beginPath();
  if (part.dot) {
    ctx.fillStyle = colour;
    ctx.arc(part.to.x, part.to.y, part.width / 2, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  ctx.strokeStyle = colour;
  ctx.lineWidth = part.width;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.moveTo(part.from.x, part.from.y);
  ctx.quadraticCurveTo(part.ctrl.x, part.ctrl.y, part.to.x, part.to.y);
  ctx.stroke();
}

/** Paints a finished stroke ({colour, points}). */
export function paintStroke(ctx, stroke) {
  for (const part of strokeParts(stroke.points)) paintPart(ctx, part, stroke.colour);
}

/**
 * What was painted, step by step, for undo and redo. A step is a stroke ({kind: 'stroke', colour, points}) or
 * "clear" ({kind: 'clear'}: every stroke wiped), which is taken back like a stroke. At most `limit` steps are kept
 * (all by default): add() returns the oldest one when it has to go.
 */
export function createHistory(limit = Infinity) {
  let steps = [];
  let undone = [];
  return {
    add(step) {
      steps.push(step);
      undone = [];                            // a new stroke after an undo: what was taken back is gone
      return steps.length > limit ? steps.shift() : null;
    },
    undo() {
      if (!steps.length) return false;
      undone.push(steps.pop());
      return true;
    },
    redo() {
      if (!undone.length) return false;
      steps.push(undone.pop());
      return true;
    },
    get canUndo() { return steps.length > 0; },
    get canRedo() { return undone.length > 0; },
    /** What shows: the strokes since the last clear, and whether there was one. */
    shown() {
      const at = steps.map((s) => s.kind).lastIndexOf('clear');
      return { cleared: at >= 0, strokes: steps.slice(at + 1) };
    },
    reset() {
      steps = [];
      undone = [];
    },
  };
}

/** 'undo' for Ctrl+Z, 'redo' for Ctrl+Y and Ctrl+Shift+Z (Cmd on a Mac), else null. */
export function undoKey(e) {
  if (!(e.ctrlKey || e.metaKey) || e.altKey) return null;
  const key = String(e.key || '').toLowerCase();
  if (key === 'y') return e.shiftKey ? null : 'redo';
  if (key === 'z') return e.shiftKey ? 'redo' : 'undo';
  return null;
}

