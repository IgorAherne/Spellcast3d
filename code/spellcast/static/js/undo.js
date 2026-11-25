// Ctrl+Z takes back the newest change first, and only changes made in the stage on screen: a stage's undo lists live
// as long as its view, and the props column's deletions belong to the stage they were made in. Each list stamps its
// changes here and registers how to read its newest stamp, so on Ctrl+Z a list acts only when it holds the newest.

let clock = 0;
const lists = new Set();

/** The stamp of a change Ctrl+Z can take back: a later change has a larger one. */
export const undoStamp = () => ++clock;

/** Registers an undo list by `newest` (-> the stamp of its newest change still to undo; 0 or none: empty) -> its removal. */
export function undoList(newest) {
  lists.add(newest);
  return () => lists.delete(newest);
}

/** Whether the change stamped `stamp` is the newest one still to undo on the page. */
export function isNewest(stamp) {
  if (!stamp) return false;
  for (const newest of lists) if ((newest() || 0) > stamp) return false;
  return true;
}

/** Ctrl+Z (Cmd+Z), without Shift (redo) or Alt. */
export const isUndoKey = (e) => !!(e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key?.toLowerCase() === 'z';
