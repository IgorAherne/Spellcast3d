// The terms of the tools the artist should know before using a result commercially. TriFlow is free for
// non-commercial use only (its ADPNCL licence, and the MeshLib trial it runs on): wherever it can be
// picked, a small "Non-commercial" chip says so, and its terms are one click away. Qwen-Image 2.1's model
// is under Qwen's research licence, but the pictures made with it are the artist's, also for commercial
// work (Qwen said so on 21 Sep 2026): a quiet "Research licence" chip says that on hover, with the licence,
// Qwen's statement and the config file to switch to Forge Neo one click away. FLUX.3 Image Klein, the Images page's
// optional third model, is under the FLUX Kommunity License: free for non-commercial use, its pictures for commercial
// use only by creators and companies under US$5M of yearly revenue; a "Kommunity licence" chip says so while it is
// the picked model, with the licence one click away. The hub may report a tool's licence in /api/status
// (tools[id].license); these are the defaults when it does not.

import { h } from './dom.js';
import { icon } from './icons.js';
import { toolName, toolStatus } from './store.js';

const TRIFLOW_LICENSE = 'https://github.com/IgorAherne/triflow/blob/873a2cdf68b0a1ec243635bbf9170c87556b2f2b/LICENSE';
// MeshLib's non-commercial trial licence, at the version TriFlow's venv installs (requirements-triflow.txt meshlib==)
const MESHLIB = 'https://github.com/MeshInspector/MeshLib/blob/v3.1.1.211/LICENSE';
// The Qwen RESEARCH LICENSE AGREEMENT at the model revision Spellcast3D downloads (adapters/qwen.py)
const QWEN_LICENSE = 'https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE';
// Qwen's statement of 21 Sep 2026 that the pictures are the user's (Qwen Developers on X; adapters/qwen.py)
const QWEN_STATEMENT = 'https://x.com/QwenDevs/status/2101917379785838660';
// The FLUX Kommunity License v1.0 of FLUX.3 Image Klein (the Images page's optional third model), at the revision
// Spellcast3D downloads it from (install/models.py FLUX3_GGUF_REVISION)
const FLUX3_LICENSE = 'https://huggingface.co/unsloth/flux-3-image-klein-GGUF/blob/12ef13ce920d53f45fe805093910c899668b9e76/LICENSE.md';

/**
 * Per tool (or model: flux3): {noncommercial | research | kommunity, url (its licence), links: [{label, url}] (other
 * terms it comes with, or what its note rests on), owner (whose research licence), name (the licence's own name), title
 * (what the line calls it when it is no tool: toolName otherwise), note (what it means for the results), alternative
 * (how to use another tool: [the question, the words of the link that shows the file to change])}.
 */
const LICENSES = {
  // FLUX.3 Image Klein (Black Forest Labs; the name only says which model it is, no endorsement): free for
  // non-commercial use, and commercial use of its pictures (its "Outputs") only for "Qualifying Users", whose yearly
  // revenue (with their affiliates) is under US$5M.
  flux3: {
    kommunity: true, url: FLUX3_LICENSE, name: 'FLUX Kommunity License', title: 'FLUX.3 Image Klein',
    note: 'Free for non-commercial use; commercial use of the pictures only for creators and companies under US$5M '
      + 'yearly revenue.',
  },
  triflow: { noncommercial: true, url: TRIFLOW_LICENSE, links: [{ label: 'MeshLib terms', url: MESHLIB }] },
  qwen: {
    research: true, url: QWEN_LICENSE, links: [{ label: 'Qwen’s statement', url: QWEN_STATEMENT }], owner: 'Qwen',
    note: 'The pictures you make are yours, also for commercial work (Qwen, 21 Sep 2026).',
    alternative: ['Prefer Forge Neo?', 'Show spellcast3d-config.ini'],
  },
};

const isUrl = (u) => typeof u === 'string' && /^https?:\/\//.test(u);

/**
 * The tool's terms: the hub's (tools[id].license: {noncommercial?, commercial?, research?, url?, name?, links?,
 * note?} or a URL) over the defaults above: {noncommercial, research, url, links, note, owner, alternative}.
 * null for a tool with nothing to say (open source, fine for commercial work).
 */
export function licenseOf(tool) {
  const base = LICENSES[tool] || null;
  const hub = toolStatus(tool)?.license;
  if (typeof hub === 'string' && isUrl(hub)) return base ? terms(base, { url: hub }) : null;
  if (!hub || typeof hub !== 'object') return base && terms(base, {});
  const noncommercial = hub.noncommercial ?? (hub.commercial === false ? true : base?.noncommercial ?? false);
  const research = !noncommercial && !!(hub.research ?? base?.research);
  if (!noncommercial && !research) return null;
  return terms(base || {}, {
    noncommercial, research,
    url: isUrl(hub.url) ? hub.url : base?.url || null,
    links: Array.isArray(hub.links) ? hub.links.filter((l) => isUrl(l?.url)) : base?.links || [],
    note: typeof hub.note === 'string' && hub.note.trim() ? hub.note.trim() : base?.note || '',
  });
}

/** The defaults `base` with what the hub says `over` them, every field there. */
const terms = (base, over) => ({
  noncommercial: !!base.noncommercial, research: !base.noncommercial && !!base.research,
  kommunity: !base.noncommercial && !base.research && !!base.kommunity, url: base.url || null,
  links: base.links || [], note: base.note || '', owner: base.owner || '', name: base.name || '',
  title: base.title || '', alternative: base.alternative || null, ...over,
});

/** Free for non-commercial use only (TriFlow). */
export const nonCommercial = (tool) => !!licenseOf(tool)?.noncommercial;

/** The model is under a research licence; what it makes may still be the artist's (Qwen-Image 2.1). */
export const researchLicence = (tool) => !!licenseOf(tool)?.research;

/** The FLUX Kommunity License: non-commercial, and the pictures commercially only under US$5M (FLUX.3 Image Klein). */
export const kommunityLicence = (tool) => !!licenseOf(tool)?.kommunity;

/** "FLUX.3 Image Klein is under the FLUX Kommunity License: free for non-commercial use; commercial use of the pictures
 * only for creators and companies under US$5M yearly revenue." */
export function kommunityText(tool) {
  const lic = licenseOf(tool);
  if (!lic?.kommunity) return '';
  const note = lic.note ? `${lic.note.charAt(0).toLowerCase()}${lic.note.slice(1)}` : '';
  return `${lic.title || toolName(tool)} is under the ${lic.name || 'FLUX Kommunity License'}${note ? `: ${note}` : '.'}`;
}

/** The words the page uses for it, in one sentence. */
export const nonCommercialText = (tool) => `${toolName(tool)} is free for non-commercial use.`;

/** "Qwen-Image 2.1 is under Qwen’s research licence. The pictures you make are yours, also for commercial work (…)." */
export function researchText(tool) {
  const lic = licenseOf(tool);
  if (!lic?.research) return '';
  return [`${toolName(tool)} is under ${lic.owner ? `${lic.owner}’s` : 'a'} research licence.`, lic.note].filter(Boolean).join(' ');
}

/** A link that opens outside the page (the artist's browser), with the ↗ that says so. */
const outLink = (label, url) => h('a', { href: url, target: '_blank', rel: 'noopener noreferrer' }, `${label} ↗`);

/**
 * The small chip next to the tool's name: "Non-commercial" (TriFlow) or "Research licence" (Qwen-Image 2.1),
 * its terms on hover; nothing for a tool without one. onclick: the chip is a button (Images: it shows the
 * line with the links, licenseLine).
 */
export function licenseChip(tool, onclick = null) {
  if (nonCommercial(tool)) {
    return h('span', { class: 'chip nc', title: `${nonCommercialText(tool)} Not for work you sell or are paid for.` }, 'Non-commercial');
  }
  if (kommunityLicence(tool)) {
    // "Kommunity licence": FLUX.3 Image Klein's (Images, while it is the picked model), its terms on hover
    const title = `${kommunityText(tool)}${onclick ? '\nClick for the licence.' : ''}`;
    return onclick
      ? h('button', { type: 'button', class: 'chip kl', title, 'aria-expanded': 'false', onclick }, 'Kommunity licence')
      : h('span', { class: 'chip kl', title }, 'Kommunity licence');
  }
  if (!researchLicence(tool)) return null;
  const title = `${researchText(tool)}${onclick ? '\nClick for the licence, and how to switch to another image generator.' : ''}`;
  return onclick
    ? h('button', { type: 'button', class: 'chip rl', title, 'aria-expanded': 'false', onclick }, 'Research licence')
    : h('span', { class: 'chip rl', title }, 'Research licence');
}

/**
 * The visible line about the tool's terms: "TriFlow is free for non-commercial use." over "License ↗ ·
 * MeshLib terms ↗" (under the picker while it is picked), or "Qwen-Image 2.1 is under Qwen’s research
 * licence. The pictures you make are yours, also for commercial work (…). Licence ↗ · Qwen’s statement ↗ ·
 * Prefer Forge Neo? Show spellcast3d-config.ini" (Images, from its chip; showFile() shows that file in File
 * Explorer, whose first lines say how to switch). null for a tool without one.
 */
export function licenseLine(tool, { showFile = null } = {}) {
  const lic = licenseOf(tool);
  if (lic?.kommunity) {
    // "FLUX.3 Image Klein is under the FLUX Kommunity License: free for non-commercial use; ... Licence ↗"
    const links = [lic.url ? outLink('Licence', lic.url) : null, ...lic.links.map((l) => outLink(l.label || 'Terms', l.url))].filter(Boolean);
    return h('div', { class: 'license-line kommunity' }, icon('info'),
      h('span', null, `${kommunityText(tool)} `, links.flatMap((a, i) => (i ? [' · ', a] : [a]))));
  }
  if (lic?.research) {
    const links = [lic.url ? outLink('Licence', lic.url) : null, ...lic.links.map((l) => outLink(l.label || 'Terms', l.url))].filter(Boolean);
    const [question, fileWords] = lic.alternative || [];
    const file = !fileWords ? null : showFile
      ? h('a', { href: '#', onclick: (e) => { e.preventDefault(); showFile(); } }, fileWords)
      : 'Switch the line in spellcast3d-config.ini (Spellcast3D’s folder).';
    return h('div', { class: 'license-line research' }, icon('info'),
      h('span', null, `${researchText(tool)} `, links.flatMap((a, i) => (i ? [' · ', a] : [a])),
        question ? `${links.length ? ' · ' : ''}${question} ` : null, question ? file : null));
  }
  if (!lic?.noncommercial) return null;
  const links = [lic.url ? outLink('License', lic.url) : null, ...lic.links.map((l) => outLink(l.label || 'Terms', l.url))].filter(Boolean);
  return h('div', { class: 'license-line' }, icon('info'),
    h('span', null, nonCommercialText(tool), h('br'), links.flatMap((a, i) => (i ? [' · ', a] : [a]))));
}
