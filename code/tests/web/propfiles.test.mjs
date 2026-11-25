// Tests of propfiles.js (static/js/stages/propfiles.js): which prop and which side a file name means, and how the
// pictures of a folder (or a drop) are grouped into props.
//   node --test code/tests/web/propfiles.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

const { groupPropFiles, propName, propOf, severalProps } = await import('../../spellcast/static/js/stages/propfiles.js');

const files = (...names) => names.map((name) => ({ name, webkitRelativePath: name }));
const summary = (groups) => groups.map((g) => [g.name, g.files.map((f) => [f.file.name, f.side])]);

test('a side word at the end or the start of the name is the side; the rest is the prop', () => {
  assert.deepEqual(propOf('knight_front.png'), { prop: 'Knight', side: 'front' });
  assert.deepEqual(propOf('Knight-Back.PNG'), { prop: 'Knight', side: 'back' });
  assert.deepEqual(propOf('knight rear.jpg'), { prop: 'Knight', side: 'back' });
  assert.deepEqual(propOf('front_knight.png'), { prop: 'Knight', side: 'front' });
  assert.deepEqual(propOf('Left-old_barrel.webp'), { prop: 'Old barrel', side: 'left' });
  assert.deepEqual(propOf(String.raw`C:rt\dragon_right.png`), { prop: 'Dragon', side: 'right' });
  assert.deepEqual(propOf('art/dragon.right.png'), { prop: 'Dragon', side: 'right' });
});

test('a name without a side word is a prop of its own; a bare side word takes the folder', () => {
  assert.deepEqual(propOf('barrel.png'), { prop: 'Barrel', side: null });
  assert.deepEqual(propOf('frontier_post.png'), { prop: 'Frontier post', side: null }, '"frontier" is not "front"');
  assert.deepEqual(propOf('knight/front.png'), { prop: 'Knight', side: 'front' });
  assert.deepEqual(propOf('props/knight/back.png'), { prop: 'Knight', side: 'back' });
  assert.deepEqual(propOf('front.png'), { prop: 'Front', side: 'front' }, 'no folder: the word is the name');
  assert.equal(propName('old_barrel-01'), 'Old barrel 01');
});

test('the pictures of a folder are grouped into props, in the order their first file came', () => {
  const got = groupPropFiles(files('knight_front.png', 'dragon_front.png', 'knight_back.png', 'dragon_left.png',
    'ork front.png', 'barrel.png', 'knight_right.png'));
  assert.deepEqual(summary(got), [
    ['Knight', [['knight_front.png', 'front'], ['knight_back.png', 'back'], ['knight_right.png', 'right']]],
    ['Dragon', [['dragon_front.png', 'front'], ['dragon_left.png', 'left']]],
    ['Ork', [['ork front.png', 'front']]],
    ['Barrel', [['barrel.png', null]]],
  ]);
});

test('a second picture for a side the prop has becomes its own prop', () => {
  const got = groupPropFiles(files('knight_front.png', 'Knight-Front.png', 'knight front.jpg', 'knight_front (2).png', 'knight_front_v3.png'));
  assert.deepEqual(summary(got), [
    ['Knight', [['knight_front.png', 'front']]],
    ['Knight Front', [['Knight-Front.png', 'front']]],   // the name's own capitals stay
    ['Knight front 2', [['knight front.jpg', 'front']]],
    // Not a side word at the end: a prop of its own, with one picture.
    ['Knight front (2)', [['knight_front (2).png', null]]],
    ['Knight front v3', [['knight_front_v3.png', null]]],
  ]);
});

test('names differing only in case are one prop; a sideless picture joins its namesake', () => {
  const got = groupPropFiles(files('Knight_front.png', 'KNIGHT_back.png', 'knight.png'));
  assert.deepEqual(summary(got), [['Knight', [['Knight_front.png', 'front'], ['KNIGHT_back.png', 'back'], ['knight.png', null]]]]);
});

test('a drop is "several props" only when more than one prop has a side word', () => {
  assert.equal(severalProps(files('knight_front.png', 'knight_back.png')), false, 'one prop: the usual import');
  assert.equal(severalProps(files('knight_front.png', 'dragon_front.png')), true);
  assert.equal(severalProps(files('a.png', 'b.png')), false, 'no sides said: the usual import (one prop, name order)');
  assert.equal(severalProps(files('knight_front.png', 'barrel.png')), false, 'one prop with sides, one picture');
});
