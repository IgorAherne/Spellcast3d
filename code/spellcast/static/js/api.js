// The hub HTTP API (see ARCHITECTURE.md "Hub HTTP API"). Every error becomes an ApiError whose
// message is the hub's user-facing "detail".

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

const OFFLINE = "Can't reach Spellcast3D. Make sure its black console window is still open, then try again.";

export async function api(method, path, body) {
  const init = { method, headers: {} };
  if (body instanceof FormData) {
    init.body = body;
  } else if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
    init.keepalive = init.body.length < 60000;   // lets the last autosave finish while the tab closes
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, OFFLINE);
  }
  return answer(res.status, await res.text());
}

/**
 * POSTs files as multipart form data: one `field` part per {file, path}, named by its path so folders
 * survive ("textures/wood.png"), after the plain text `fields` ({name: value}). onProgress(fraction | null)
 * follows the upload itself (fetch cannot); the promise settles like api() once the hub has answered.
 */
export function uploadFiles(path, files, { field = 'files', onProgress, fields = {} } = {}) {
  const form = new FormData();
  for (const [key, value] of Object.entries(fields)) form.append(key, value);
  for (const { file, path: name } of files) form.append(field, file, name);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', path);
    xhr.upload.onprogress = (e) => onProgress?.(e.lengthComputable ? e.loaded / e.total : null);
    xhr.onload = () => {
      try {
        resolve(answer(xhr.status, xhr.responseText));
      } catch (e) {
        reject(e);
      }
    };
    xhr.onerror = () => reject(new ApiError(0, OFFLINE));
    xhr.send(form);
  });
}

/** PUTs raw bytes (application/octet-stream) and settles like api(). */
export async function putBytes(path, bytes) {
  let res;
  try {
    res = await fetch(path, { method: 'PUT', headers: { 'Content-Type': 'application/octet-stream' }, body: bytes });
  } catch {
    throw new ApiError(0, OFFLINE);
  }
  return answer(res.status, await res.text());
}

function answer(status, text) {
  let data = null;
  if (text) {
    try { data = JSON.parse(text); } catch { data = null; }
  }
  if (status < 200 || status >= 300) throw new ApiError(status, detailOf(data, status));
  return data;
}

function detailOf(data, status) {
  const d = data?.detail;
  if (typeof d === 'string' && d) return d;
  if (Array.isArray(d) && d.length) return d.map((x) => x?.msg || String(x)).join('; ');
  return `Spellcast3D could not do that (error ${status}). The console window may show more details. Please try again.`;
}

const enc = encodeURIComponent;

export const projectApi = (project) => `/api/projects/${enc(project)}`;
export const propApi = (project, propId) => `${projectApi(project)}/props/${enc(propId)}`;

/** URL of a file inside projects/<project>/props/<propId>/ (rel as stored in project.json). */
export function fileUrl(project, propId, rel, bust) {
  if (!rel) return null;
  const path = ['props', propId, ...rel.split('/')].map(enc).join('/');
  return `/files/${enc(project)}/${path}${bust ? `?v=${bust}` : ''}`;
}

/** Tool URLs are reported as http://127.0.0.1:P/...; point them at whatever host this page uses. */
export function localUrl(url) {
  if (!url) return null;
  try {
    const u = new URL(url, location.href);
    u.hostname = location.hostname;
    return u.href;
  } catch {
    return url;
  }
}
