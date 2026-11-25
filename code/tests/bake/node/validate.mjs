// Dev-only: run the Khronos glTF-Validator (2.0.0-dev.3.10) on baked GLBs.
//
//   node validate.mjs baked.glb [more.glb ...]
//
// Exit code 0 when every file has 0 errors and 0 warnings (infos and hints are listed).
import fs from 'fs';
import path from 'path';
import validator from 'gltf-validator';

const SEVERITY = ['E', 'W', 'I', 'H'];

async function validate(file) {
  const report = await validator.validateBytes(new Uint8Array(fs.readFileSync(file)), {
    uri: path.basename(file),
    maxIssues: 100,
    externalResourceFunction: (uri) => Promise.resolve(new Uint8Array(fs.readFileSync(path.join(path.dirname(file), uri)))),
  });
  const { numErrors, numWarnings, numInfos, numHints, messages } = report.issues;
  const ok = numErrors === 0 && numWarnings === 0;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${path.basename(file)}: ${numErrors} errors, ${numWarnings} warnings, `
    + `${numInfos} infos, ${numHints} hints`);
  for (const m of messages) console.log(`  ${SEVERITY[m.severity]} ${m.code} ${m.pointer || ''} ${m.message}`);
  return ok;
}

const files = process.argv.slice(2);
if (files.length === 0) {
  console.error('usage: node validate.mjs baked.glb [more.glb ...]');
  process.exitCode = 2;
} else {
  const results = [];
  for (const file of files) results.push(await validate(file));
  process.exitCode = results.every(Boolean) ? 0 : 1;
}
