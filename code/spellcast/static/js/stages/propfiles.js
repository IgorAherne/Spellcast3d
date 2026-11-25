// Pictures of several props at once: the "Import folder" card of the Images chooser, and a drop of many pictures
// on it. Which picture belongs to which prop, and which side it shows, is read from the file names:
//
//     knight_front.png  knight_back.png  dragon_front.png  dragon-left.png  Front_Ork.png  ork back.jpg
//
// A side word ("front", "back" or "rear", "left", "right") at the start or the end of the name (after the usual
// separators: "_", "-", a space, a dot) is the side; the rest, with the separators as spaces, is the prop's name
// ("Knight", "Dragon", "Ork"). A file that is nothing but a side word ("front.png" in a folder "knight") takes its
// folder's name. A name without a side word is a prop of its own, with that one picture ("barrel.png" -> "Barrel").
// Two files with the same prop and side ("knight_front.png" and "knight_front (2).png") are not the same prop:
// the second keeps what is left of its name ("Knight (2)"). Pure functions: code/tests/web/propfiles.test.mjs.

const SIDES = { front: 'front', back: 'back', rear: 'back', left: 'left', right: 'right' };
const SEP = '[\\s_\\-.]+';
const AT_START = new RegExp(`^(front|back|rear|left|right)(?:${SEP}|$)`, 'i');
const AT_END = new RegExp(`(?:^|${SEP})(front|back|rear|left|right)$`, 'i');
const MAX_NAME = 40;   // as the hub shortens a prop's name (projects.box_name_from_file)

/** "C:\\art\\Knight_Front.png" or "knight/front.png" -> {stem: "Knight_Front", folder: "knight"}. */
function parts(path) {
  const slash = String(path || '').replace(/\\/g, '/');
  const file = slash.replace(/^.*\//, '');
  const folder = slash.includes('/') ? slash.replace(/\/[^/]*$/, '').replace(/^.*\//, '') : '';
  return { stem: file.replace(/\.[^.]*$/, ''), folder };
}

/** The words of a file or folder name as a prop's name: "old_barrel-01" -> "Old barrel 01". */
export function propName(stem) {
  const name = String(stem || '').replace(/[\s_\-.]+/g, ' ').trim().slice(0, MAX_NAME);
  return name ? name[0].toUpperCase() + name.slice(1) : '';
}

/**
 * One file's place: {prop: the prop's name (the key files are grouped by), side: 'front' | 'back' | 'left' |
 * 'right' | null}. `path` may carry its folders (a drop or a folder picker gives webkitRelativePath).
 */
export function propOf(path) {
  const { stem, folder } = parts(path);
  let side = null;
  let rest = stem;
  let m = stem.match(AT_END);
  if (m) {
    side = SIDES[m[1].toLowerCase()];
    rest = stem.slice(0, m.index);
  } else if ((m = stem.match(AT_START))) {
    side = SIDES[m[1].toLowerCase()];
    rest = stem.slice(m[0].length);
  }
  const name = propName(rest) || (side ? propName(folder) : '') || propName(stem);
  return { prop: name, side };
}

/**
 * Files (File objects, or {name, webkitRelativePath?}) -> the props to make, in the order their first file came:
 * [{name, files: [{file, side}]}]. A file's side is what its name says; a second file for a side the prop already
 * has becomes its own prop (the first keeps the side). Files without a side word are props of their own.
 */
export function groupPropFiles(files) {
  const groups = new Map();   // name (lower) -> {name, files, sides: Set}
  const claim = (name, file, side) => {
    const key = name.toLowerCase();
    let g = groups.get(key);
    if (!g) {
      g = { name, files: [], sides: new Set() };
      groups.set(key, g);
    }
    g.files.push({ file, side });
    if (side) g.sides.add(side);
  };
  for (const file of files) {
    const path = file.webkitRelativePath || file.name || '';
    const { prop, side } = propOf(path);
    if (!side) {
      claim(propName(parts(path).stem) || prop || 'Prop', file, null);
      continue;
    }
    const taken = groups.get(prop.toLowerCase())?.sides.has(side);
    if (!taken) {
      claim(prop, file, side);
      continue;
    }
    // "knight_front (2).png" after "knight_front.png": a prop of its own, named by its whole stem.
    let name = propName(parts(path).stem);
    let n = 2;
    while (groups.get(name.toLowerCase())?.sides.has(side)) name = `${propName(parts(path).stem)} ${n++}`;
    claim(name, file, side);
  }
  return [...groups.values()].map(({ name, files: own }) => ({ name, files: own }));
}

/** Whether `files` dropped together are the sides of several props (more than one group with a side word). */
export const severalProps = (files) => groupPropFiles(files).filter((g) => g.files.some((f) => f.side)).length > 1;
