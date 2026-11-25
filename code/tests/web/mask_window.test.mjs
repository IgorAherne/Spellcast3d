// Smart Mask's frame (stages/mask_window.js), the page's copy of the image server's patch_of, window_of and
// to_the_middle (launchers/qwen_server.py): the boxes the two give for the same brush, round a tileable picture's edge
// too, and the parts a window shows in on a tileable picture.
//   node --test code/tests/web/mask_window.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

const {
  maskWindow, patchOf, roundTheEdge, sameShape, windowOf, windowParts,
} = await import(new URL('../../spellcast/static/js/stages/mask_window.js', import.meta.url));

test('the window is the image server\'s for the same brushed box and picture', () => {
  // [brushed, size, window (null: too large)], as window_of gives them
  const cases = [
    [[1454, 1454, 1823, 1823], [2048, 2048], [1120, 1120, 2048, 2048]],
    [[200, 1000, 1700, 1040], [2048, 2048], [0, 764, 1952, 1276]],
    [[500, 500, 1400, 1400], [2048, 2048], null],
    [[624, 624, 1424, 1424], [2048, 2048], [512, 512, 1536, 1536]],
    [[10, 10, 60, 60], [2048, 2048], [0, 0, 768, 768]],
    [[18, 700, 1518, 740], [1536, 1536], [0, 384, 1536, 1056]],
    [[400, 400, 600, 600], [1024, 1024], [116, 116, 884, 884]],
    [[162, 162, 862, 862], [1024, 1024], [0, 0, 1024, 1024]],
    [[100, 500, 400, 800], [832, 1216], [0, 266, 768, 1034]],
    [[0, 0, 2048, 300], [2048, 2048], [0, 0, 2048, 512]],
    [[0, 900, 2048, 1230], [2048, 2048], null],
    [[900, 100, 1000, 1900], [2048, 2048], [694, 0, 1206, 2048]],
    [[600, 600, 937, 937], [1536, 1536], [368, 368, 1168, 1168]],
    [[1990, 1000, 2040, 1050], [2048, 2048], [1280, 641, 2048, 1409]],
    [[300, 300, 1132, 1132], [2048, 2048], [204, 204, 1228, 1228]],
    [[300, 300, 1133, 1133], [2048, 2048], null],
    [[700, 700, 1100, 1100], [1536, 1536], [484, 484, 1316, 1316]],
  ];
  for (const [brushed, size, window] of cases) assert.deepEqual(windowOf(brushed, size), window, `${brushed} in ${size}`);
});

test('round a tileable picture\'s edge, what was brushed lies between the ends of the longest stretch left unbrushed, the first one to_the_middle finds', () => {
  // [runs, side, {start, span}], as to_the_middle lays them (checked against it)
  const cases = [
    [[[400, 500]], 1024, { start: 400, span: 100 }],                 // the rest of the side round the edge
    [[[0, 50], [1000, 1024]], 1024, { start: 1000, span: 74 }],      // across the edge
    [[[0, 1024]], 1024, { start: 0, span: 1024 }],                   // nothing left: all of it
    [[[0, 10], [50, 60]], 100, { start: 50, span: 60 }],             // two as long: the one inside, found first
    [[[30, 100]], 100, { start: 30, span: 70 }],                     // the stretch from 0 on
    [[[20, 30], [60, 70]], 100, { start: 20, span: 50 }],            // the longer one across the edge
    [[[60, 70], [20, 30], [25, 40]], 100, { start: 20, span: 50 }],  // in any order, overlapping
    [[[990, 1020]], 1024, { start: 990, span: 30 }],
  ];
  for (const [runs, side, extent] of cases) assert.deepEqual(roundTheEdge(runs, side), extent, `${JSON.stringify(runs)} of ${side}`);
  assert.equal(roundTheEdge([], 1024), null);
});

// (what was brushed, the picture's size, its patch): tests/test_qwen_launcher.py PATCHES holds the same cases
const PATCHES = [
  [[900, 900, 1100, 1100], [2048, 2048], [488, 488, 1512, 1512]],
  [[1454, 1454, 1823, 1823], [2048, 2048], [955, 955, 1979, 1979]],
  [[1900, 1000, 2040, 1100], [2048, 2048], [1024, 538, 2048, 1562]],
  [[2030, 1000, 2048, 1100], [2048, 2048], [1024, 538, 2048, 1562]],
  [[20, 20, 200, 200], [2048, 2048], [1, 1, 1025, 1025]],
  [[300, 300, 500, 500], [2048, 2048], [81, 81, 1105, 1105]],
  [[180, 180, 620, 260], [2048, 1536], [59, 54, 1083, 1078]],
  [[400, 400, 1600, 1500], [2048, 2048], [154, 154, 1846, 1746]],
  [[200, 300, 1900, 1700], [2048, 2048], [64, 81, 1999, 1963]],
  [[700, 700, 836, 836], [1536, 1536], [256, 256, 1280, 1280]],
  [[1300, 40, 1500, 90], [1536, 1152], [506, 10, 1530, 1034]],
  [[100, 100, 300, 300], [1024, 1024], [0, 0, 1024, 1024]],
  [[100, 100, 300, 300], [1024, 768], [0, 0, 1024, 768]],
  [[300, 500, 500, 700], [832, 1216], [0, 0, 832, 1216]],
  [[100, 100, 300, 300], [1216, 832], [0, 0, 1216, 832]],
  [[300, 500, 500, 700], [1248, 1824], [56, 88, 1080, 1112]],
  [[300, 200, 1500, 900], [2048, 1536], [81, 57, 1773, 1249]],
  [[150, 900, 1650, 1400], [1824, 1248], [49, 179, 1768, 1203]],
  [[901, 900, 1100, 1100], [2048, 2048], [488, 488, 1512, 1512]],
];

test('the patch of a whole redraw is the image server\'s for the same brushed box and picture, nearing the picture\'s side without a jump', () => {
  for (const [brushed, size, patch] of PATCHES) assert.deepEqual(patchOf(brushed, size), patch, `${brushed} in ${size}`);
  // A mask brought nearer the picture's side, px by px: the patch's side follows it in steps of a pixel at the most,
  // and is on the side only with the mask.
  const lefts = [];
  for (let low = 700; low >= 0; low--) lefts.push(patchOf([low, 900, low + 200, 1100], [2048, 2048])[0]);
  assert.deepEqual([lefts[0], lefts.at(-1)], [288, 0]);
  assert.ok(lefts.every((left, i) => i === 0 || (lefts[i - 1] - left >= 0 && lefts[i - 1] - left <= 1)));
  assert.deepEqual([400, 300, 200, 100, 50, 20, 5].map((low) => lefts[700 - low]), [89, 81, 64, 30, 9, 1, 0]);
});

test('the frame of a mask: in the request\'s pixels - a large picture\'s patch, a tileable picture\'s window, wrapped round its edge', () => {
  const line = (from, to, w = 20) => ({ points: [{ ...from, w }, { ...to, w }] });
  const stroke = line({ x: 100, y: 100 }, { x: 300, y: 120 });    // reaches 90 .. 310 across, 90 .. 130 down
  const shown = [1024, 768];
  // All of the picture (1K, not tileable - a tall one too: 832 x 1216 is no more than 1024 x 1024 pixels), or another
  // shape (drawn anew), or nothing brushed on the picture: none.
  assert.equal(maskWindow([stroke], shown, { width: 1024, height: 768, tileable: false }), null);
  assert.equal(maskWindow([stroke], [832, 1216], { width: 832, height: 1216, tileable: false }), null);
  assert.equal(maskWindow([stroke], shown, null), null);
  assert.equal(maskWindow([stroke], shown, { width: 2048, height: 2048, tileable: false }), null);
  assert.equal(maskWindow([line({ x: -100, y: -100 }, { x: -60, y: -100 })], shown, { width: 2048, height: 1536, tileable: false }), null);
  assert.ok(sameShape([1024, 768], [2048, 1536]) && sameShape([832, 1216], [1248, 1824]) && !sameShape([1024, 768], [1024, 1024]));
  // At 2K the brushed box is twice as large (180, 180, 620, 260): the patch of the whole redraw round it, the least
  // one. A larger mask's is larger, and always fits; one that takes all of the picture shows no frame.
  assert.deepEqual(maskWindow([stroke], shown, { width: 2048, height: 1536, tileable: false }), {
    box: [59, 54, 1083, 1078], size: [2048, 1536], fits: true,
  });
  assert.deepEqual(maskWindow([line({ x: 300, y: 300 }, { x: 700, y: 500 })], shown, { width: 2048, height: 1536, tileable: false }), {
    box: [334, 288, 1666, 1312], size: [2048, 1536], fits: true,
  });
  const wide = line({ x: 50, y: 50 }, { x: 1000, y: 700 });
  assert.deepEqual(maskWindow([stroke, wide], shown, { width: 2048, height: 1536, tileable: false }).box, [21, 24, 2045, 1498],
    'most of the picture: off its sides still');
  const all = line({ x: 0, y: 0 }, { x: 1024, y: 768 });
  assert.equal(maskWindow([all], shown, { width: 2048, height: 1536, tileable: false }), null);
  // A tileable picture's is a window; too large for one: the least box round what was brushed, as the picture is moved.
  assert.deepEqual(maskWindow([stroke, wide], shown, { width: 2048, height: 1536, tileable: true }), {
    box: [26, 1520, 2074, 3052], size: [2048, 1536], fits: false,
  });
  // Tileable at 1K: a stroke at the right edge lies in the middle of the moved picture, its window round it there -
  // a full side down, 768 across - and laid back: it leaves the right edge and comes back at the left.
  const edge = line({ x: 1000, y: 300 }, { x: 1010, y: 300 });    // reaches 990 .. 1020 across, 290 .. 310 down
  const got = maskWindow([edge], shown, { width: 1024, height: 768, tileable: true });
  assert.deepEqual(got, { box: [621, 684, 1389, 1452], size: [1024, 768], fits: true });
  assert.deepEqual(windowParts(got.box, got.size), [{ box: [621, 0, 1024, 768], open: ['r'] }, { box: [0, 0, 365, 768], open: ['l'] }]);
  // A finished stroke's reach is kept; one that grows is measured again: 60 px further down, the window 30 px lower.
  edge.points.push({ x: 1010, y: 360, w: 20 });
  assert.deepEqual(maskWindow([edge], shown, { width: 1024, height: 768, tileable: true }).box, [621, 714, 1389, 1482]);
});

test('a window shows in parts round a tileable picture\'s edge, each without an outline where the edge cuts it', () => {
  assert.deepEqual(windowParts([100, 50, 500, 400], [1024, 768]), [{ box: [100, 50, 500, 400], open: [] }]);
  assert.deepEqual(windowParts([900, 700, 1200, 900], [1024, 768]), [
    { box: [900, 700, 1024, 768], open: ['r', 'b'] }, { box: [900, 0, 1024, 132], open: ['r', 't'] },
    { box: [0, 700, 176, 768], open: ['l', 'b'] }, { box: [0, 0, 176, 132], open: ['l', 't'] },
  ]);
  assert.deepEqual(windowParts([300, 0, 1324, 768], [1024, 768]), [{ box: [0, 0, 1024, 768], open: [] }], 'a whole side: all of it');
});
