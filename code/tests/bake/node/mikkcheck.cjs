// Dev-only: compare a baked GLB's TANGENT attribute with reference MikkTSpace
// (mikktspace 1.1.1, the wasm build three.js's computeMikkTSpaceTangents and
// glTF-Transform use), run the way a glTF consumer runs it: on the file's (v-down)
// TEXCOORD_0 with w negated. The baker's Mikk on v-up UVs with w as-is must equal that.
//
//   node mikkcheck.cjs baked.glb [more.glb ...]
//
// Exit code 0 when every corner agrees: tangent angle < 0.01 deg and the same sign of w.
'use strict';

const fs = require('fs');
const path = require('path');
const { generateTangents } = require('mikktspace');

const MAX_ANGLE_DEG = 0.01;
const COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
const TYPES = { 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };

function readGlb(file) {
  const bytes = fs.readFileSync(file);
  if (bytes.readUInt32LE(0) !== 0x46546c67) throw new Error(`${file} is not a GLB`);
  let gltf = null;
  let bin = null;
  for (let at = 12; at < bytes.length;) {
    const length = bytes.readUInt32LE(at);
    const kind = bytes.readUInt32LE(at + 4);
    const chunk = bytes.subarray(at + 8, at + 8 + length);
    if (kind === 0x4e4f534a) gltf = JSON.parse(chunk.toString('utf8'));
    else if (kind === 0x004e4942) bin = chunk;
    at += 8 + length;
  }
  const accessor = (index) => {
    const a = gltf.accessors[index];
    const view = gltf.bufferViews[a.bufferView];
    const Type = TYPES[a.componentType];
    const width = COMPONENTS[a.type];
    const stride = view.byteStride || Type.BYTES_PER_ELEMENT * width;
    const start = (view.byteOffset || 0) + (a.byteOffset || 0);
    const out = new Float64Array(a.count * width);
    const data = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
    const read = {
      5121: (o) => data.getUint8(o), 5123: (o) => data.getUint16(o, true),
      5125: (o) => data.getUint32(o, true), 5126: (o) => data.getFloat32(o, true),
    }[a.componentType];
    for (let i = 0; i < a.count; i++) {
      for (let c = 0; c < width; c++) out[i * width + c] = read(start + i * stride + c * Type.BYTES_PER_ELEMENT);
    }
    return { values: out, width, count: a.count };
  };
  return { gltf, accessor };
}

// Triangle soup (what generateTangents expects): one entry per corner.
function soup(attribute, indices) {
  const out = new Float32Array(indices.length * attribute.width);
  indices.forEach((v, k) => {
    for (let c = 0; c < attribute.width; c++) out[k * attribute.width + c] = attribute.values[v * attribute.width + c];
  });
  return out;
}

function check(file) {
  const { gltf, accessor } = readGlb(file);
  const angles = [];
  let signAgree = 0;
  for (const mesh of gltf.meshes) {
    for (const prim of mesh.primitives) {
      const at = prim.attributes;
      if (at.TANGENT === undefined) {
        console.log(`FAIL ${path.basename(file)}: a primitive has no TANGENT attribute`);
        return false;
      }
      const pos = accessor(at.POSITION);
      const indices = prim.indices !== undefined ? Array.from(accessor(prim.indices).values)
        : Array.from({ length: pos.count }, (_, i) => i);
      const reference = generateTangents(soup(pos, indices), soup(accessor(at.NORMAL), indices),
        soup(accessor(at.TEXCOORD_0), indices));
      const baked = soup(accessor(at.TANGENT), indices);
      for (let k = 0; k < indices.length; k++) {
        const a = [reference[4 * k], reference[4 * k + 1], reference[4 * k + 2]];
        const b = [baked[4 * k], baked[4 * k + 1], baked[4 * k + 2]];
        if (Math.sign(-reference[4 * k + 3]) === Math.sign(baked[4 * k + 3])) signAgree++;
        const cross = Math.hypot(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
        angles.push(Math.atan2(cross, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) * 180 / Math.PI);
      }
    }
  }
  angles.sort((x, y) => x - y);
  const q = (p) => angles[Math.min(angles.length - 1, Math.floor(p * angles.length))];
  const max = angles[angles.length - 1];
  const ok = max < MAX_ANGLE_DEG && signAgree === angles.length;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${path.basename(file)}: ${angles.length} corners, w agrees on `
    + `${(100 * signAgree / angles.length).toFixed(2)}%, tangent angle deg median ${q(0.5).toFixed(4)} `
    + `p99 ${q(0.99).toFixed(4)} max ${max.toFixed(4)} (limit ${MAX_ANGLE_DEG})`);
  return ok;
}

const files = process.argv.slice(2);
// process.exitCode, not process.exit(): exiting while the wasm module winds down trips
// a libuv assertion on Windows (exit code 127).
if (files.length === 0) {
  console.error('usage: node mikkcheck.cjs baked.glb [more.glb ...]');
  process.exitCode = 2;
} else {
  process.exitCode = files.map(check).every(Boolean) ? 0 : 1;
}
