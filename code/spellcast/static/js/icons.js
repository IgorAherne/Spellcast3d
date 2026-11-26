// Inline stroke icons (24x24, currentColor), so the UI needs no icon font or network.

const PATHS = {
  left: '<path d="M15 18l-6-6 6-6"/>',
  right: '<path d="M9 18l6-6-6-6"/>',
  down: '<path d="M12 5v14M19 12l-7 7-7-7"/>',
  up: '<path d="M12 19V5M5 12l7-7 7 7"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  x: '<path d="M18 6L6 18M6 6l12 12"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  trash: '<path d="M4 7h16M9 7V4.5h6V7M18 7l-.8 12.2a1.5 1.5 0 0 1-1.5 1.3H8.3a1.5 1.5 0 0 1-1.5-1.3L6 7M10 11v6M14 11v6"/>',
  dice: '<rect x="3.5" y="3.5" width="17" height="17" rx="4"/><circle cx="8.5" cy="8.5" r="1.1" fill="currentColor"/><circle cx="15.5" cy="15.5" r="1.1" fill="currentColor"/><circle cx="12" cy="12" r="1.1" fill="currentColor"/><circle cx="15.5" cy="8.5" r="1.1" fill="currentColor"/><circle cx="8.5" cy="15.5" r="1.1" fill="currentColor"/>',
  download: '<path d="M12 4v11M7 10.5l5 5 5-5M5 20h14"/>',
  upload: '<path d="M12 20V9M7 13.5l5-5 5 5M5 4h14"/>',
  image: '<rect x="3.5" y="3.5" width="17" height="17" rx="3"/><circle cx="9" cy="9" r="1.8"/><path d="M20.5 15l-4.5-4.5L5 20.5"/>',
  cube: '<path d="M12 2.8l8.2 4.6v9.2L12 21.2l-8.2-4.6V7.4z"/><path d="M12 21.2V12M20.2 7.4L12 12 3.8 7.4"/>',
  grid: '<path d="M4 4.5c5 1.5 11 1.5 16 0M4 19.5c5-1.5 11-1.5 16 0M4.5 4c1.5 5 1.5 11 0 16M19.5 4c-1.5 5-1.5 11 0 16M4 12h16M12 4v16"/>',
  sun: '<circle cx="12" cy="12" r="3.5"/><path d="M12 3v2.5M12 18.5V21M3 12h2.5M18.5 12H21M5.6 5.6l1.8 1.8M16.6 16.6l1.8 1.8M5.6 18.4l1.8-1.8M16.6 7.4l1.8-1.8"/>',
  figure: '<circle cx="12" cy="5" r="2"/><path d="M12 7.5v6.5M12 14l-3.5 6.5M12 14l3.5 6.5M6 10.5h12"/>',
  refresh: '<path d="M20 12a8 8 0 1 1-2.4-5.7L20 8.5M20 4v4.5h-4.5"/>',
  reset: '<path d="M4 12a8 8 0 1 0 2.4-5.7L4 8.5M4 4v4.5h4.5"/>',
  folder: '<path d="M3.5 7.5A2 2 0 0 1 5.5 5.5h3.8l2 2h7.2a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>',
  sparkles: '<path d="M11 3.5l1.7 4.6 4.6 1.7-4.6 1.7L11 16.1l-1.7-4.6-4.6-1.7 4.6-1.7z"/><path d="M18 14.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/>',
  power: '<path d="M12 3.5v8M7 6.2a7.5 7.5 0 1 0 10 0"/>',
  pencil: '<path d="M4.5 19.5h4l10-10a2.8 2.8 0 0 0-4-4l-10 10z"/>',
  maskbrush: '<path d="M20.5 3.5l-8 8"/><path d="M12.6 11.3c-2-.9-4.6.3-5.2 2.6-.4 1.6-1.3 2.7-3 3.3 3.3 1.9 8.3.6 9.6-3 .4-1.1-.2-2.3-1.4-2.9z"/>'
    + '<path d="M5.5 3.5v4M3.5 5.5h4"/>',
  terminal: '<path d="M5 16.5l5-4.5-5-4.5M12.5 18h6.5"/>',
  info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8h.01"/>',
  alert: '<path d="M12 3.8l9 15.7H3z"/><path d="M12 10v4M12 17h.01"/>',
  stop: '<rect x="6.5" y="6.5" width="11" height="11" rx="2"/>',
  save: '<path d="M5 4.5h11l3.5 3.5v11a1.5 1.5 0 0 1-1.5 1.5H6A1.5 1.5 0 0 1 4.5 19V6A1.5 1.5 0 0 1 5 4.5z"/><path d="M8 4.5v4h7v-4M8 20.5v-6h8v6"/>',
  chevron: '<path d="M6.5 9.5l5.5 5.5 5.5-5.5"/>',
  chat: '<path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 9 9 0 0 1-3.9-.9L3 20.5l1.5-4.6A8.4 8.4 0 0 1 12 3a8.5 8.5 0 0 1 9 8.5z"/>',
  heart: '<path d="M12 19.5s-7.5-4.4-7.5-10a4.2 4.2 0 0 1 7.5-2.6 4.2 4.2 0 0 1 7.5 2.6c0 5.6-7.5 10-7.5 10z"/>',
  video: '<rect x="3" y="5" width="18" height="14" rx="3.5"/><path d="M10.2 9.2v5.6l4.9-2.8z"/>',
  more: '<circle cx="5.5" cy="12" r="1.3" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.3" fill="currentColor" stroke="none"/><circle cx="18.5" cy="12" r="1.3" fill="currentColor" stroke="none"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 13.5a7.6 7.6 0 0 0 0-3l2-1.6-2-3.4-2.4.9a7.6 7.6 0 0 0-2.6-1.5L14 2.5h-4l-.4 2.4A7.6 7.6 0 0 0 7 6.4l-2.4-.9-2 3.4 2 1.6a7.6 7.6 0 0 0 0 3l-2 1.6 2 3.4 2.4-.9a7.6 7.6 0 0 0 2.6 1.5l.4 2.4h4l.4-2.4a7.6 7.6 0 0 0 2.6-1.5l2.4.9 2-3.4z"/>',
  clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
  gauge: '<path d="M4.5 16.5a8 8 0 1 1 15 0"/><path d="M12 14.5l3.5-4.5"/><circle cx="12" cy="15" r="1.2" fill="currentColor" stroke="none"/>',
  paste: '<rect x="6" y="4.5" width="12" height="16" rx="2"/><path d="M9.5 4.5V3.8a1 1 0 0 1 1-1h3a1 1 0 0 1 1 1v.7M9.5 10.5h5M9.5 14h5"/>',
  square: '<rect x="5.5" y="5.5" width="13" height="13" rx="2"/>',
  portrait: '<rect x="7" y="3.5" width="10" height="17" rx="2"/>',
  landscape: '<rect x="3.5" y="7" width="17" height="10" rx="2"/>',
  orbit: '<ellipse cx="12" cy="12" rx="8.5" ry="3.6"/><path d="M12 3.5v17"/><path d="M17.8 6.6l2.4.6-.7 2.3"/>',
  mask: '<rect x="3.5" y="3.5" width="17" height="17" rx="4.5" stroke-dasharray="3.1 2.9"/><path d="M8.4 14.6c1-3.1 3.2-4.9 7.3-5.1" stroke-width="3.8"/>',
  eye: '<path d="M2.5 12s3.5-6.5 9.5-6.5 9.5 6.5 9.5 6.5-3.5 6.5-9.5 6.5S2.5 12 2.5 12z"/><circle cx="12" cy="12" r="2.8"/>',
  eyeoff: '<path d="M2.5 12s3.5-6.5 9.5-6.5 9.5 6.5 9.5 6.5-3.5 6.5-9.5 6.5S2.5 12 2.5 12z"/><circle cx="12" cy="12" r="2.8"/><path d="M4.5 19.5l15-15"/>',
  lens: '<path d="M9.2 19.5h5.6L20 4.5H4z"/><path d="M6.6 12h10.8"/>',
  stamp: '<path d="M9.5 3.8h5a1.3 1.3 0 0 1 1.3 1.3v1.5a3 3 0 0 1-1.4 2.6l-.4.2v2.6h4.2a2 2 0 0 1 2 2v1.5H3.8V14a2 2 0 0 1 2-2H10V9.4l-.4-.2a3 3 0 0 1-1.4-2.6V5.1a1.3 1.3 0 0 1 1.3-1.3z"/><path d="M5 19.5h14"/>',
};

export function icon(name, cls = '') {
  const tpl = document.createElement('template');
  tpl.innerHTML = `<svg class="icon ${cls}" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" `
    + `stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${PATHS[name] || ''}</svg>`;
  return tpl.content.firstChild;
}
