// AI Embroidery Digitizer — the four screens. Plain JavaScript, no build step.
'use strict';

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

// Little line drawings for the picture buttons (24 x 24, drawn with the text color).
const SHIRT = 'M8 3 4 5 2 9l3 2 1-1v10h12V10l1 1 3-2-2-4-4-2a4 4 0 0 1-8 0Z';
const ICONS = {
  garment: SHIRT,
  project: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Z M12 7l1.5 3.2 3.5.4-2.6 2.4.7 3.5-3.1-1.8-3.1 1.8.7-3.5L7 10.6l3.5-.4Z',
  tshirt: SHIRT,
  hoodie: 'M8 4 4 6 2 11l3 1 1-2v10h12V10l1 2 3-1-2-5-4-2c-1 2-2 3-4 3S9 6 8 4Z M10 13h4v3h-4Z',
  polo: 'M8 3 4 5 2 9l3 2 1-1v10h12V10l1 1 3-2-2-4-4-2-4 4-4-4Z M12 7v5',
  hat: 'M4 15a8 8 0 0 1 16 0v1H4Z M20 16l2 2.5H10 M12 7V5.5',
  softhat: 'M4 15a8 8 0 0 1 16 0v1H4Z M20 16l2 2.5H10 M12 7V5.5',
  towel: 'M6 3h12v15H6Z M6 18v3h12v-3 M9 6.5h6 M9 9h6',
  denim: 'M7 3h10l1 18h-4l-2-10-2 10H6Z M7 6h10',
  fleece: 'M8 4 4 6 2 11l3 1 1-2v10h12V10l1 2 3-1-2-5-4-2c-1 2-2 3-4 3S9 6 8 4Z',
  leather: 'M5 4h14l-1 5 1 11H5l1-11Z M9 9h6',
  neoprene: 'M8 4h8l1 16H7Z M8 9h8',
  delicate: 'M4 6c5-3 11 3 16 0v12c-5 3-11-3-16 0Z',
  beanie: 'M5 16a7 7 0 0 1 14 0Z M4 16h16v4H4Z M12 9V7',
  left_chest: `${SHIRT} M14 9.5h2.5V12H14Z`,
  full_front: `${SHIRT} M9 9.5h6v6H9Z`,
  hat_front: 'M4 15a8 8 0 0 1 16 0v1H4Z M20 16l2 2.5H10 M10 11h4v3h-4Z',
  custom: 'M3 8h18v8H3Z M7 8v3 M11 8v4 M15 8v3 M19 8v4',
  feltie: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Z M12 7l1.5 3.2 3.5.4-2.6 2.4.7 3.5-3.1-1.8-3.1 1.8.7-3.5L7 10.6l3.5-.4Z',
  keyfob: 'M12 9a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11Z M10 9V4a2 2 0 0 1 4 0v5',
  coaster: 'M7 4h10a3 3 0 0 1 3 3v10a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V7a3 3 0 0 1 3-3Z M9 9h6v6H9Z',
  ornament: 'M12 7a7 7 0 1 0 0 14 7 7 0 0 0 0-14Z M10 7V4h4v3 M12 4V2',
  plush: 'M12 6a7 7 0 1 0 0 14 7 7 0 0 0 0-14Z M6.5 7.5a2.5 2.5 0 1 1 3-3 M17.5 7.5a2.5 2.5 0 1 0-3-3 M10 12.5h.01 M14 12.5h.01 M11 15.5c.6.5 1.4.5 2 0',
  zipbag: 'M3 7h18v12H3Z M3 10.5h18 M8 8.7h8',
  pillow: 'M5 5c4 1 10 1 14 0-1 4-1 10 0 14-4-1-10-1-14 0 1-4 1-10 0-14Z',
  pillowpanel: 'M4 4h16v16H4Z M8 8h8v8H8Z',
  lace: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Z M12 7.5a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9Z M12 3v4.5 M12 16.5V21 M3 12h4.5 M16.5 12H21',
  circle: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Z',
  square: 'M7 4h10a3 3 0 0 1 3 3v10a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V7a3 3 0 0 1 3-3Z',
  follow: 'M12 3c4 0 8 2 8 6 0 3-2 4-2 6s1 5-6 6c-5 0-8-3-8-7 0-3 2-4 2-6 0-3 2-5 6-5Z',
  rect: 'M3 7h18v10H3Z',
  auto: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Z M8.5 12.5l2.5 2.5 4.5-5',
  api: 'M14.5 9.5a4.5 4.5 0 1 0-4 4L8 16H6v2H4v2h3l6.5-6.5a4.5 4.5 0 0 0 1-4Z',
};

function icon(id) {
  const d = ICONS[id];
  if (!d) return null;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', 'ico');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.5');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true');
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', d);
  svg.append(path);
  return svg;
}

const GARMENT_COLORS = ['#eeece6', '#9aa0a8', '#23262d', '#1f2a44', '#8f1f24'];
const NO_SERVER = "We couldn't reach the digitizer. Make sure its window is still open, then try again.";

const remembered = (key, fallback) => {
  try { return localStorage.getItem(key) || fallback; } catch { return fallback; }
};
const remember = (key, value) => {
  try { localStorage.setItem(key, value); } catch { /* private window: fine, just not remembered */ }
};

const state = {
  options: null,
  ai: null,                 // { provider, chatgpt, has_key, spent_today_usd, estimate_usd, ... }
  skipAi: false,            // the user pressed "Skip AI" for this picture
  file: null,
  size: 'left_chest',
  sizeChosenByHand: false,
  fabric: 'tshirt',
  mode: 'garment',          // 'garment' (decorate something) or 'project' (in-the-hoop item)
  project: { type: 'feltie', shape: 'circle', size_in: 3 },
  cleanupId: null,          // the picture as cleaned up on screen 2
  cleanupFor: null,         // which picture/size/garment that clean-up was made for
  cleanup: null,
  version: null,            // the chosen version
  stitchedFor: null,        // which version/colors the current stitches were made for
  colors: 6,
  maxColors: 6,
  sizeIn: null,             // widest side chosen on the Stitches screen (null = screen 1's)
  density: 0,               // softer (-2) .. denser (+2)
  inside: 'auto',           // blank areas inside the design: auto | sew | open
  dropTextMm: 0,            // text shorter than this is left out (a one-click fix)
  letters: 'auto',          // words: auto | retype (embroidery lettering) | trace
  aiAssist: 'auto',         // ChatGPT plans the stitches: auto | on | off
  placing: null,            // words waiting for a click to say where they go
  moreFabrics: false,
  jobId: null,
  result: null,
  machine: remembered('machine', 'brother'),
  brand: remembered('brand', ''),   // thread brand shown on the Download screen
  edits: [],                // the user's own changes, made by clicking the preview
  actual: false,            // preview at real size instead of fitted
  garment: GARMENT_COLORS[0],
  runToken: 0,
};

const money = (usd) => (usd < 1 ? `${Math.max(1, Math.round(usd * 100))}¢` : `$${usd.toFixed(2)}`);

// ───────────────────────── talking to the server ─────────────────────────
async function api(path, options) {
  let res;
  try { res = await fetch(path, options); } catch { throw new Error(NO_SERVER); }
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new Error((body && body.error) || NO_SERVER);
  return body;
}

// Wait for a background job, showing its plain-words progress. Resolves with the job.
async function waitFor(path, token, labelEl) {
  while (token === state.runToken) {
    const job = await api(path);
    if (token !== state.runToken) break;
    if (job.status === 'done') return job;
    if (job.status === 'error') throw new Error(job.error || NO_SERVER);
    labelEl.textContent = job.label || 'Working…';
    await new Promise((r) => setTimeout(r, 350));
  }
  return null;   // the user moved on
}

// ───────────────────────── moving between screens ─────────────────────────
function go(step) {
  for (const n of [1, 2, 3, 4, 5]) $('#screen' + n).hidden = n !== step;
  if (step !== 3) stopSim();
  document.querySelectorAll('.steps li').forEach((li) => {
    const n = Number(li.dataset.step);
    li.classList.toggle('current', n === step);
    li.classList.toggle('done', n < step);
  });
  window.scrollTo(0, 0);
  if (step === 1) renderAiLine();
  if (step === 3 && state.result) draw();
}

// ───────────────────────── screen 1: picture ─────────────────────────
function choiceButtons(box, items, current, onPick) {
  box.innerHTML = '';
  for (const item of items) {
    const b = el('button');
    b.type = 'button';
    b.dataset.id = item.id;
    const picture = icon(item.icon || item.id);
    if (picture) b.append(picture);
    const label = el('span', 'label', item.name);
    if (item.detail) label.appendChild(el('small', '', item.detail));
    b.append(label);
    b.classList.toggle('on', item.id === current);
    b.addEventListener('click', () => onPick(item.id));
    box.appendChild(b);
  }
}

function renderChoices() {
  choiceButtons($('#modes'), [
    { id: 'garment', name: 'Decorate a garment', detail: 'Shirt, hat, towel…' },
    { id: 'project', name: 'Make a project', detail: 'Patch, key fob, coaster…' },
  ], state.mode, (id) => { state.mode = id; renderChoices(); });
  $('#garmentQs').hidden = state.mode !== 'garment';
  $('#projectQs').hidden = state.mode !== 'project';
  const types = state.options.projects.types;
  const type = types.find((t) => t.id === state.project.type) || types[0];
  choiceButtons($('#projTypes'), types, type.id, (id) => {
    const t = types.find((x) => x.id === id);
    const shape = t.shapes.includes(state.project.shape) ? state.project.shape : t.shapes[0];
    state.project = { type: id, shape, size_in: t.default_in };
    renderChoices();
  });
  if (!type.shapes.includes(state.project.shape)) state.project.shape = type.shapes[0];
  $('#projShapeQ').hidden = type.shapes.length < 2;       // a zipper bag is always a rectangle
  const shapes = state.options.projects.shapes.filter((sh) => type.shapes.includes(sh.id));
  choiceButtons($('#projShapes'), shapes, state.project.shape, (id) => {
    state.project = { ...state.project, shape: id };
    renderChoices();
  });
  choiceButtons($('#projSizes'), type.sizes_in.map((v) => ({ id: String(v), name: `${v} in` })),
    String(state.project.size_in), (id) => {
      state.project = { ...state.project, size_in: Number(id) };
      renderChoices();
    });
  choiceButtons($('#sizes'), state.options.sizes, state.size, (id) => {
    state.size = id;
    state.sizeChosenByHand = true;
    renderChoices();
  });
  $('#customRow').hidden = state.size !== 'custom';
  const shown = state.options.fabrics.filter((f) => state.moreFabrics || !f.more || f.id === state.fabric);
  $('#moreFabrics').hidden = state.moreFabrics || !state.options.fabrics.some((f) => f.more);
  choiceButtons($('#fabrics'), shown, state.fabric, (id) => {
    state.fabric = id;
    // hats have their own size; pick it unless the user already chose a size themselves
    if (!state.sizeChosenByHand) state.size = ['hat', 'softhat'].includes(id) ? 'hat_front' : 'left_chest';
    renderChoices();
  });
}

function renderAiLine() {
  const p = $('#aiLine');
  p.innerHTML = '';
  if (!state.ai) return;
  if (state.ai.provider === 'chatgpt') {
    p.textContent = 'AI clean-up is on, using your ChatGPT plan. It takes a minute or two a picture.';
  } else if (state.ai.provider === 'api') {
    p.textContent = `AI clean-up costs about ${money(state.ai.estimate_usd)} a picture. ` +
      `Today so far: $${state.ai.spent_today_usd.toFixed(2)} of $${state.ai.daily_spend_cap_usd.toFixed(2)}.`;
  } else {
    p.append('AI clean-up is off. ');
    const b = el('button', 'link', 'Turn it on');
    b.type = 'button';
    b.addEventListener('click', openSettings);
    p.append(b, ' in Settings. The app works without it too.');
  }
}

function showUploadError(text) {
  const p = $('#uploadError');
  p.textContent = text || '';
  p.hidden = !text;
}

function setFile(file) {
  if (!file) return;
  if (!/^image\/(png|jpeg|webp)$/.test(file.type)) {
    showUploadError("That file isn't a picture we can use. Please choose a PNG, JPG or WEBP.");
    return;
  }
  showUploadError('');
  state.file = file;
  state.fileStamp = Date.now();
  state.colors = state.maxColors;      // a new picture starts from the default again
  state.skipAi = false;
  const img = $('#thumb');
  if (img.src) URL.revokeObjectURL(img.src);
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  $('#drop').classList.add('has');
  $('#dropText').querySelector('strong').textContent = 'Click or drop to use a different picture';
  $('#next1').disabled = false;
}

function setupScreen1() {
  const drop = $('#drop');
  const input = $('#fileInput');
  drop.addEventListener('click', () => input.click());
  drop.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  input.addEventListener('change', () => { setFile(input.files[0]); input.value = ''; });
  for (const ev of ['dragenter', 'dragover']) {
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); });
  }
  for (const ev of ['dragleave', 'drop']) {
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); });
  }
  drop.addEventListener('drop', (e) => setFile(e.dataTransfer.files[0]));
  // dropping a file anywhere else on the page should not navigate away
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => e.preventDefault());
  window.addEventListener('paste', (e) => {
    if ($('#screen1').hidden) return;
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
    if (item) setFile(item.getAsFile());
  });
  $('#sampleBtn').addEventListener('click', async () => {
    try {
      const blob = await (await fetch('samples/sample_logo.png')).blob();
      setFile(new File([blob], 'sample logo.png', { type: 'image/png' }));
    } catch {
      showUploadError("We couldn't load the sample. Please choose your own picture.");
    }
  });
  $('#next1').addEventListener('click', () => { if (state.file) startCleanup(); });
  $('#moreFabrics').addEventListener('click', () => { state.moreFabrics = true; renderChoices(); });
}

// ───────────────────────── screen 2: clean up ─────────────────────────
function showCleanupStage(which) {
  $('#cWorking').hidden = which !== 'working';
  $('#cFailed').hidden = which !== 'failed';
  $('#cReady').hidden = which !== 'ready';
  $('#next2').disabled = which !== 'ready';
}

function cleanupKey() {
  return [state.fileStamp, state.size, state.size === 'custom' ? $('#customIn').value : '',
    state.fabric, state.ai && state.ai.provider, state.skipAi,
    state.mode === 'project' ? JSON.stringify(state.project) : ''].join('|');
}

async function startCleanup() {
  go(2);
  // Coming back with nothing changed? Keep the versions (and the choice) already made.
  if (state.cleanup && state.cleanupFor === cleanupKey()) { showCleanup(); return; }

  const token = ++state.runToken;
  state.cleanup = null;
  state.result = null;
  state.stitchedFor = null;
  state.sizeIn = null;                 // a new clean-up starts from screen 1's size again
  state.density = 0;
  state.inside = 'auto';
  state.dropTextMm = 0;
  state.letters = 'auto';
  state.edits = [];
  showCleanupStage('working');
  $('#cLabel').textContent = 'Getting started…';
  $('#skipAi').hidden = state.skipAi || !(state.ai && state.ai.provider);
  const key = cleanupKey();

  const form = new FormData();
  form.append('file', state.file, state.file.name || 'design.png');
  form.append('size', state.size);
  form.append('custom_in', $('#customIn').value || '4');
  form.append('fabric', state.fabric);
  form.append('use_ai', state.skipAi ? 'false' : 'true');
  if (state.mode === 'project') form.append('project', JSON.stringify(state.project));
  try {
    const { id } = await api('api/cleanup', { method: 'POST', body: form });
    const job = await waitFor('api/cleanup/' + id, token, $('#cLabel'));
    if (!job) return;
    state.cleanupId = id;
    state.cleanupFor = key;
    state.cleanup = job.result;
    state.version = job.result.default;
    state.imageStamp = Date.now();
    showCleanup();
  } catch (err) {
    if (token !== state.runToken) return;
    $('#cFailText').textContent = err.message;
    showCleanupStage('failed');
  }
}

function notice(sel, text, withSettingsLink) {
  const p = $(sel);
  p.innerHTML = '';
  p.hidden = !text;
  if (!text) return;
  p.append(text + ' ');
  if (withSettingsLink) {
    const b = el('button', 'link', 'Open Settings');
    b.type = 'button';
    b.addEventListener('click', openSettings);
    p.append(b);
  }
}

function showCleanup() {
  const c = state.cleanup;
  showCleanupStage('ready');
  if (c.spend) state.ai = c.spend;
  notice('#cNote', c.ai.note, ['off', 'no_key', 'bad_key', 'cap', 'chatgpt_missing',
    'chatgpt_signed_out'].includes(c.ai.problem));
  if (c.ai.problem === 'skipped') {
    const p = $('#cNote');
    p.hidden = false;
    p.textContent = 'You skipped AI for this picture. ';
    const b = el('button', 'link', 'Use AI after all');
    b.type = 'button';
    b.addEventListener('click', () => { state.skipAi = false; startCleanup(); });
    p.append(b);
  }
  notice('#cTip', c.tip);
  notice('#cFlash', c.flash);
  if (!c.versions.some((v) => v.id === state.version)) state.version = c.default;

  const box = $('#versions');
  box.innerHTML = '';
  const img = (name) => `api/cleanup/${state.cleanupId}/image/${name}?t=${state.imageStamp}`;
  for (const v of c.versions) {
    const card = el('div', 'version' + (v.id === state.version ? ' on' : ''));
    card.tabIndex = 0;
    card.setAttribute('role', 'radio');
    card.setAttribute('aria-checked', String(v.id === state.version));
    const pick = () => {
      if (state.version !== v.id) state.edits = [];
      state.version = v.id;
      showCleanup();
    };
    card.addEventListener('click', pick);
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); }
    });

    const title = el('h3');
    title.append(el('span', 'tick'), v.name);
    card.append(title, el('p', '', v.blurb));

    // before / after slider: your picture on the left, this version on the right
    const ba = el('div', 'ba');
    const after = el('img');
    after.src = img(v.id);
    after.alt = v.name + ' version';
    const before = el('img');
    before.src = img('source');
    before.alt = 'Your picture';
    const slider = el('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = '100';
    slider.value = '50';
    slider.setAttribute('aria-label', 'Slide to compare your picture with this version');
    const clip = () => { before.style.clipPath = `inset(0 ${100 - slider.value}% 0 0)`; };
    slider.addEventListener('input', clip);
    slider.addEventListener('click', (e) => e.stopPropagation());
    clip();
    ba.append(after, before, el('span', 'tag l', 'Before'), el('span', 'tag r', 'After'), slider);
    card.append(ba);

    const chips = el('div', 'chips');
    chips.setAttribute('aria-label', 'Thread colors');
    for (const hex of v.colors) {
      const s = el('span');
      s.style.background = hex;
      chips.append(s);
    }
    card.append(chips);

    if (v.text_warning) {
      const w = el('div', 'warn');
      w.append(`AI changed your text. It should say "${v.text_warning.expected.join(' / ')}" ` +
        `but this version says "${v.text_warning.found.join(' / ') || 'nothing'}".`);
      if (['faithful', 'simplified', 'bold'].includes(v.id)) {
        const b = el('button', '', 'Fix it: draw again');
        b.type = 'button';
        b.addEventListener('click', (e) => { e.stopPropagation(); changeVersions('retry', v.id); });
        w.append(document.createElement('br'), b);
      }
      card.append(w);
    }
    box.append(card);
  }

  $('#tweakBox').hidden = !c.ai.used;
  const s = state.ai;
  $('#cCost').textContent = c.ai.used && s && s.provider === 'api'
    ? `AI spending today: $${s.spent_today_usd.toFixed(2)} of $${s.daily_spend_cap_usd.toFixed(2)}.`
    : '';
}

// "Make this change" and "draw again": both ask the AI for one more version.
async function changeVersions(kind, version) {
  const token = ++state.runToken;
  const form = new FormData();
  form.append('version', version);
  if (kind === 'tweak') form.append('change', $('#tweakText').value);
  showCleanupStage('working');
  $('#cLabel').textContent = kind === 'tweak' ? 'Making your change…' : 'Drawing it again…';
  try {
    await api(`api/cleanup/${state.cleanupId}/${kind}`, { method: 'POST', body: form });
    const job = await waitFor('api/cleanup/' + state.cleanupId, token, $('#cLabel'));
    if (!job) return;
    state.cleanup = job.result;
    state.imageStamp = Date.now();
    if (job.result.picked && !job.result.flash) state.version = job.result.picked;
    if (kind === 'tweak' && !job.result.flash) $('#tweakText').value = '';
    state.stitchedFor = null;
    showCleanup();
  } catch (err) {
    if (token !== state.runToken) return;
    state.cleanup.flash = err.message;
    showCleanup();
  }
}

function setupScreen2() {
  $('#back2').addEventListener('click', () => { state.runToken++; go(1); });
  $('#cFailBack').addEventListener('click', () => { state.runToken++; go(1); });
  $('#next2').addEventListener('click', () => startJob());
  $('#skipAi').addEventListener('click', () => { state.skipAi = true; startCleanup(); });
  $('#tweakBtn').addEventListener('click', () => {
    if ($('#tweakText').value.trim()) changeVersions('tweak', state.version);
  });
  $('#tweakText').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); $('#tweakBtn').click(); }
  });
}

// ───────────────────────── screen 3: stitches ─────────────────────────
function showStage(which) {
  $('#working').hidden = which !== 'working';
  $('#failed').hidden = which !== 'failed';
  $('#ready').hidden = which !== 'ready';
  $('#next3').disabled = which !== 'ready';
}

async function startJob(force) {
  go(3);
  const key = [state.cleanupId, state.version, state.imageStamp].join('|');
  if (!force && state.result && state.stitchedFor === key) { showResult(); return; }

  const token = ++state.runToken;
  state.result = null;
  showStage('working');
  $('#workLabel').textContent = 'Getting started…';
  const form = new FormData();
  form.append('cleanup_id', state.cleanupId);
  form.append('version', state.version);
  form.append('colors', String(state.colors));
  form.append('density', String(state.density));
  form.append('inside', state.inside);
  form.append('drop_text_mm', String(state.dropTextMm));
  form.append('letters', state.letters);
  form.append('ai_assist', state.skipAi ? 'off' : state.aiAssist);
  form.append('edits', JSON.stringify(state.edits));
  if (state.sizeIn) form.append('size_in', String(state.sizeIn));
  try {
    const { id } = await api('api/jobs', { method: 'POST', body: form });
    const job = await waitFor('api/jobs/' + id, token, $('#workLabel'));
    if (!job) return;
    state.jobId = id;
    state.result = job.result;
    state.stitchedFor = key;
    showResult();
  } catch (err) {
    if (token !== state.runToken) return;
    $('#failText').textContent = err.message;
    showStage('failed');
  }
}

const LIGHT_TEXT = {
  green: 'Ready to sew',
  yellow: 'Will sew, but check these',
  red: "This won't sew well yet",
};

function showResult() {
  const r = state.result;
  showStage('ready');

  $('#light').className = 'light ' + r.check.status;
  $('#lightText').textContent = LIGHT_TEXT[r.check.status];
  const ul = $('#messages');
  ul.innerHTML = '';
  for (const m of r.check.messages) {
    const li = el('li', '', m.text);
    if (m.fixes && m.fixes.length) {
      const row = el('div', 'fixes');
      for (const fix of m.fixes) {
        const b = el('button', '', fix.label);
        b.type = 'button';
        b.addEventListener('click', () => applyFix(fix.set));
        row.append(b);
      }
      li.append(row);
    }
    ul.append(li);
  }

  const s = r.stats;
  const rows = [
    ['Size', `${s.width_in} × ${s.height_in} in  (${s.width_mm} × ${s.height_mm} mm)`],
    ['Stitches', s.stitches.toLocaleString()],
    ['Thread colors', String(s.colors)],
    ['Sewing time', `about ${s.sew_minutes} min`],
    ['Smallest hoop', r.check.hoop || 'none fits'],
  ];
  const dl = $('#stats');
  dl.innerHTML = '';
  for (const [k, v] of rows) dl.append(el('dt', '', k), el('dd', '', v));

  // Show how many colors the design really uses. If it came back with fewer than we allowed,
  // the picture has no more colors to give, so "+" is switched off.
  const noMore = s.colors < state.colors;
  state.colors = s.colors;
  $('#colorCount').textContent = String(s.colors);
  $('#fewer').disabled = s.colors <= 1;
  $('#more').disabled = noMore || s.colors >= state.maxColors;
  // for an in-the-hoop project the size is the finished item's, not the stitching's
  const sizeNow = r.project ? r.project.size_in : s.size_in;
  $('#sizeOut').textContent = `${sizeNow} in`;
  $('#smaller').disabled = sizeNow <= (r.project ? 1.5 : 0.75);
  $('#bigger').disabled = sizeNow >= (!r.project ? 12 : r.project.type === 'pillowpanel' ? 18 : 7);
  $('#density').value = String(r.density);
  $('#insideRow').hidden = !r.inside.present;
  $('#insideBox').checked = r.inside.sewn;
  $('#undoBtn').hidden = state.edits.length === 0;
  renderLettering(r);
  renderAiHelp(r);
  renderThreadChips();
  draw();
  renderDownload();
  saveProject();
}

// A one-click fix from the checks, or the size / softer-denser controls: change the setting
// and stitch again.
function applyFix(set) {
  if (set.size_in !== undefined) state.sizeIn = Math.min(12, Math.max(0.5, Number(set.size_in)));
  if (set.density !== undefined) state.density = Math.min(2, Math.max(-2, Number(set.density)));
  if (set.drop_text_mm !== undefined) state.dropTextMm = Number(set.drop_text_mm) || 0;
  startJob(true);
}

function shade(hex, amount) {
  const n = parseInt(hex.slice(1), 16);
  const ch = [n >> 16 & 255, n >> 8 & 255, n & 255].map((v) =>
    Math.round(amount < 0 ? v * (1 + amount) : v + (255 - v) * amount));
  return `rgb(${ch[0]},${ch[1]},${ch[2]})`;
}

// ───────────────────────── the thread picture and the sew-out simulator ─────────────────────────
// Every stitch is drawn as a thread-thick line, in sewing order so later thread covers
// earlier. The picture is drawn several times larger than it is shown (at least 12 pixels
// per millimetre) and then shrunk smoothly, so each thread stays a clean line.
// `sim` remembers how far the drawing has got, so the simulator can add stitches a few at a
// time instead of starting again for every frame.
const sim = { big: null, g: null, s: 1, ox: 0, oy: 0, total: 0, drawn: 0, cursor: null,
  playing: false, raf: 0, last: null };

function totalStitches(r) {
  let n = 0;
  for (const b of r.blocks) for (const seg of b.segments) n += seg.length / 2;
  return n;
}

function setupCanvas() {
  const r = state.result;
  const cv = $('#preview');
  const [minx, miny, maxx, maxy] = r.bounds;
  const dpr = window.devicePixelRatio || 1;
  let W;
  let H;
  if (state.actual) {                       // real size: CSS millimetres, plus a margin
    const pxPerMm = 96 / 25.4;
    W = Math.round((maxx - minx + 12) * pxPerMm);
    H = Math.round((maxy - miny + 12) * pxPerMm);
  } else {
    W = cv.parentElement.clientWidth;
    H = Math.round(Math.max(260, Math.min(W, window.innerHeight * 0.62)));
  }
  cv.width = Math.round(W * dpr);
  cv.height = Math.round(H * dpr);
  cv.style.width = W + 'px';
  cv.style.height = H + 'px';

  const pad = (state.actual ? 6 * (96 / 25.4) : 18) * dpr;
  const shown = Math.min((cv.width - 2 * pad) / (maxx - minx), (cv.height - 2 * pad) / (maxy - miny));
  const k = Math.max(1, Math.min(12 / shown, 4096 / Math.max(cv.width, cv.height)));
  sim.big = document.createElement('canvas');
  sim.big.width = Math.round(cv.width * k);
  sim.big.height = Math.round(cv.height * k);
  sim.g = sim.big.getContext('2d');
  sim.k = k;
  sim.s = shown * k;                                     // pixels per millimetre while drawing
  sim.ox = (sim.big.width - (maxx - minx) * sim.s) / 2 - minx * sim.s;
  sim.oy = (sim.big.height - (maxy - miny) * sim.s) / 2 - miny * sim.s;
  sim.total = totalStitches(r);
  resetDrawing();
}

function resetDrawing() {
  sim.g.fillStyle = state.garment;
  sim.g.fillRect(0, 0, sim.big.width, sim.big.height);
  sim.g.lineCap = 'round';
  sim.g.lineJoin = 'round';
  sim.drawn = 0;
  sim.cursor = { b: 0, sg: 0, p: 0 };
  sim.last = null;
}

// Add stitches to the big picture until `n` of them are drawn.
function drawTo(n) {
  const r = state.result;
  n = Math.max(0, Math.min(sim.total, Math.round(n)));
  if (n < sim.drawn) resetDrawing();
  const g = sim.g;
  const s = sim.s;
  // Thread is about 0.4 mm but flattens and spreads when sewn, closing the gaps between
  // rows, so it is drawn a touch wider: otherwise the underlay shows through as a grid.
  // Every stitch is drawn as its own piece of thread, the way it lies on the fabric: a soft
  // shadow under it, the thread itself shaded by which way it runs (thread catches the light
  // differently by direction, which is what makes satin shine and fills look woven), and a
  // thin highlight down its middle.
  const lw = 0.42 * s;
  const c = sim.cursor;
  g.lineCap = 'round';
  while (sim.drawn < n && c.b < r.blocks.length) {
    const block = r.blocks[c.b];
    const seg = block.segments[c.sg];
    if (!seg) { c.b += 1; c.sg = 0; c.p = 0; continue; }
    const points = seg.length / 2;
    if (c.p >= points) { c.sg += 1; c.p = 0; continue; }
    const end = Math.min(points, c.p + Math.min(60, n - sim.drawn));
    const rgb = hexRgb(block.hex);
    // shadows of this run of stitches first, so a stitch's shadow falls on the fabric, not
    // on the thread beside it (which would streak a satin column)
    const sh = 0.12 * s;
    g.strokeStyle = rgbStr(rgb, 0.45); g.globalAlpha = 0.3; g.lineWidth = lw * 1.1;
    g.beginPath();
    for (let q = Math.max(1, c.p); q < end; q += 1) {
      g.moveTo(seg[(q - 1) * 2] * s + sim.ox + sh, seg[(q - 1) * 2 + 1] * s + sim.oy + sh);
      g.lineTo(seg[q * 2] * s + sim.ox + sh, seg[q * 2 + 1] * s + sim.oy + sh);
    }
    g.stroke();
    g.globalAlpha = 1;
    for (let q = Math.max(1, c.p); q < end; q += 1) {
      const x0 = seg[(q - 1) * 2] * s + sim.ox; const y0 = seg[(q - 1) * 2 + 1] * s + sim.oy;
      const x1 = seg[q * 2] * s + sim.ox; const y1 = seg[q * 2 + 1] * s + sim.oy;
      const len = Math.hypot(x1 - x0, y1 - y0);
      if (len < 0.05 * s) continue;
      const lit = threadLight(Math.atan2(y1 - y0, x1 - x0));
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1);
      g.strokeStyle = rgbStr(rgb, 0.84 * lit); g.lineWidth = lw; g.stroke();          // thread edge
      g.strokeStyle = rgbStr(rgb, lit); g.lineWidth = lw * 0.86; g.stroke();          // thread
      g.globalAlpha = 0.4;
      g.strokeStyle = rgbStr(rgb, 1.25 * lit, 0.25); g.lineWidth = lw * 0.22; g.stroke(); // sheen
      g.globalAlpha = 1;
    }
    sim.drawn += end - c.p;
    sim.last = [seg[(end - 1) * 2], seg[(end - 1) * 2 + 1]];
    c.p = end;
  }
}

function hexRgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [v >> 16 & 255, v >> 8 & 255, v & 255];
}

// How bright a thread looks by the way it runs (smooth, so the two slants of a satin
// zig-zag look alike and the column reads as one smooth band).
function threadLight(angle) {
  return 0.9 + 0.1 * Math.cos(2 * angle);      // the same for a stitch and its zig-zag partner
}

// rgb scaled by `k`, then blended `toWhite` of the way to white (for highlights).
function rgbStr(rgb, k, toWhite = 0) {
  const ch = rgb.map((v) => {
    const x = Math.min(255, v * k + (k > 1 ? 18 * (k - 1) * 4 : 0));
    return Math.round(x + (255 - x) * toWhite);
  });
  return `rgb(${ch[0]},${ch[1]},${ch[2]})`;
}

// Put the big picture on the screen. `fine` shrinks it in halves for the smoothest result;
// while the simulator is playing a single quick shrink is enough.
function show(fine, needle) {
  const cv = $('#preview');
  let src = sim.big;
  if (fine) {
    while (src.width > cv.width * 2) {
      const half = document.createElement('canvas');
      half.width = Math.round(src.width / 2);
      half.height = Math.round(src.height / 2);
      const hg = half.getContext('2d');
      hg.imageSmoothingQuality = 'high';
      hg.drawImage(src, 0, 0, half.width, half.height);
      src = half;
    }
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, cv.width, cv.height);
  if (needle && sim.last) {                              // where the needle is right now
    const x = (sim.last[0] * sim.s + sim.ox) / sim.k;
    const y = (sim.last[1] * sim.s + sim.oy) / sim.k;
    ctx.beginPath();
    ctx.arc(x, y, 5 * (window.devicePixelRatio || 1), 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,.9)';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#111';
    ctx.stroke();
  }
}

function draw() {
  const r = state.result;
  const cv = $('#preview');
  if (!r || cv.offsetParent === null) return;
  stopSim();
  setupCanvas();
  drawTo(sim.total);
  show(true, false);
  const scrub = $('#scrub');
  scrub.max = String(sim.total);
  scrub.value = String(sim.total);
  drawStrip();
}

// The bar under the slider: one band per thread color, with a dark tick wherever the
// thread is cut.
function drawStrip() {
  const r = state.result;
  const cv = $('#simStrip');
  const g = cv.getContext('2d');
  g.clearRect(0, 0, cv.width, cv.height);
  let at = 0;
  for (const b of r.blocks) {
    let n = 0;
    for (const seg of b.segments) n += seg.length / 2;
    g.fillStyle = b.hex;
    g.fillRect((at / sim.total) * cv.width, 0, Math.ceil((n / sim.total) * cv.width), cv.height);
    let inside = at;
    b.segments.forEach((seg, i) => {
      if (i > 0) { g.fillStyle = '#111'; g.fillRect((inside / sim.total) * cv.width, 0, 2, cv.height); }
      inside += seg.length / 2;
    });
    at += n;
    g.fillStyle = '#fff';                                // a white gap marks each color change
    g.fillRect((at / sim.total) * cv.width - 1, 0, 2, cv.height);
  }
}

function stopSim() {
  if (sim.raf) cancelAnimationFrame(sim.raf);
  sim.raf = 0;
  sim.playing = false;
  const b = $('#playBtn');
  if (b) b.textContent = '▶ Watch it sew';
}

function playSim() {
  if (!state.result) return;
  if (sim.playing) { stopSim(); show(true, true); return; }
  if (sim.drawn >= sim.total) drawTo(0);
  sim.playing = true;
  $('#playBtn').textContent = '❚❚ Pause';
  const perFrame = Math.max(2, sim.total / (14 * 60));    // the whole design in about 14 s
  const step = () => {
    drawTo(sim.drawn + perFrame);
    $('#scrub').value = String(sim.drawn);
    if (sim.drawn >= sim.total) { stopSim(); show(true, false); return; }
    show(false, true);
    sim.raf = requestAnimationFrame(step);
  };
  sim.raf = requestAnimationFrame(step);
}

// ───────────────────────── clicking the design to change it ─────────────────────────
function pointAt(clientX, clientY) {
  const r = state.result;
  const cv = $('#preview');
  const box = cv.getBoundingClientRect();
  const dpr = cv.width / box.width;
  const x = (((clientX - box.left) * dpr) * sim.k - sim.ox) / sim.s;     // millimetres
  const y = (((clientY - box.top) * dpr) * sim.k - sim.oy) / sim.s;
  return { x, y, at: [x / r.ref, y / r.ref] };
}

function areaAt(clientX, clientY) {
  const r = state.result;
  const { x, y } = pointAt(clientX, clientY);
  let best = null;
  for (const a of r.areas) {
    const ring = a.ring;
    let inside = false;
    for (let i = 0, j = ring.length - 2; i < ring.length; j = i, i += 2) {
      const xi = ring[i]; const yi = ring[i + 1]; const xj = ring[j]; const yj = ring[j + 1];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    if (inside && (!best || a.area < best.area)) best = a;
  }
  return best ? { area: best, at: [x / r.ref, y / r.ref] } : null;
}

let picked = null;

function closeAreaMenu() {
  $('#areaMenu').hidden = true;
  picked = null;
}

function addEdit(edit) {
  state.edits.push(edit);
  closeAreaMenu();
  startJob(true);
}

// "Let ChatGPT plan the stitches", and what ChatGPT said about the design.
function renderAiHelp(r) {
  const info = r.ai_digitize || { used: false, notes: [] };
  const can = !!(state.ai && state.ai.provider) && !state.skipAi;
  $('#aiAssistRow').hidden = !can;
  $('#aiAssistBox').checked = state.aiAssist !== 'off' && (info.used || state.aiAssist === 'on' || state.aiAssist === 'auto');
  const ul = $('#aiNotes');
  ul.innerHTML = '';
  const notes = info.used ? [...info.notes] : [];
  if (info.problem && state.aiAssist !== 'off' && can) {
    notes.push("ChatGPT couldn't help this time, so the app planned the stitches on its own.");
  }
  for (const n of notes) ul.append(el('li', '', n));
  ul.hidden = notes.length === 0;
}

// The "Words" choice, shown when the design has lettering in it.
function renderLettering(r) {
  const info = r.lettering || { rows: 0, retyped: 0 };
  $('#lettersRow').hidden = info.rows === 0;
  $('#lettersMode').value = state.letters;
  const hint = $('#lettersHint');
  hint.hidden = info.rows === 0;
  if (state.letters === 'trace') {
    hint.textContent = 'The words are sewn from the picture\'s own letters.';
  } else if (info.retyped > 0) {
    hint.textContent = 'Words in embroidery lettering sew sharper than letters traced from a picture. ' +
      'Click a word to change it.';
  } else {
    hint.textContent = 'Click a word on the design to retype it in embroidery lettering.';
  }
}

function stopPlacing() {
  state.placing = null;
  $('#stage').classList.remove('placing');
  $('#clickHint').textContent = 'Click any part of the design to change its color, make it outline-only, or delete it.';
}

function renderThreadChips() {
  const box = $('#threadChips');
  box.innerHTML = '';
  const seen = new Set();                 // hats sew a color more than once: list it once
  for (const c of state.result.colors) {
    if (c.label || seen.has(c.hex)) continue;            // (and skip the towel's flattening layer)
    seen.add(c.hex);
    const b = el('button');
    b.type = 'button';
    const chip = el('span', 'chip');
    chip.style.background = c.hex;
    b.append(chip, el('span', '', threadName(c) || c.hex));
    b.setAttribute('aria-label', 'Change this thread color');
    b.addEventListener('click', () => {
      const picker = $('#chipPicker');
      picker.value = c.hex;
      picker.onchange = () => addEdit({ type: 'recolor', from: c.hex, hex: picker.value });
      picker.click();
    });
    box.append(b);
  }
}

function threadName(c) {
  const t = c.threads && c.threads[state.brand];
  return t ? `${t.brand} ${t.label}`.trim() : '';
}

function setupScreen3() {
  const box = $('#garmentColors');
  box.append('Garment color:');
  for (const c of GARMENT_COLORS) {
    const b = el('button');
    b.type = 'button';
    b.style.background = c;
    b.setAttribute('aria-label', 'Garment color ' + c);
    b.classList.toggle('on', c === state.garment);
    b.addEventListener('click', () => {
      state.garment = c;
      renderDownload();
      box.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
      draw();
    });
    box.appendChild(b);
  }
  const changeColors = (delta) => {
    const n = Math.min(state.maxColors, Math.max(1, state.colors + delta));
    if (n === state.colors) return;
    state.colors = n;
    startJob(true);
  };
  const quarter = (v) => Math.round(v * 4) / 4;
  const sizeOf = () => (state.result.project ? state.result.project.size_in : state.result.stats.size_in);
  $('#smaller').addEventListener('click', () => applyFix({ size_in: quarter(sizeOf() - 0.25) }));
  $('#bigger').addEventListener('click', () => applyFix({ size_in: quarter(sizeOf() + 0.25) }));
  $('#density').addEventListener('change', (e) => applyFix({ density: Number(e.target.value) }));
  $('#insideBox').addEventListener('change', (e) => {
    state.inside = e.target.checked ? 'sew' : 'open';
    startJob(true);
  });
  $('#fewer').addEventListener('click', () => changeColors(-1));
  $('#more').addEventListener('click', () => changeColors(1));
  $('#back3').addEventListener('click', () => { state.runToken++; go(2); showCleanup(); });
  $('#failBack').addEventListener('click', () => { state.runToken++; go(2); showCleanup(); });
  $('#next3').addEventListener('click', () => go(4));
  window.addEventListener('resize', draw);
  $('#playBtn').addEventListener('click', playSim);
  $('#scrub').addEventListener('input', (e) => {
    stopSim();
    drawTo(Number(e.target.value));
    show(false, Number(e.target.value) < sim.total);
  });
  $('#scrub').addEventListener('change', (e) => show(true, Number(e.target.value) < sim.total));
  $('#actualBtn').addEventListener('click', () => {
    state.actual = !state.actual;
    $('#actualBtn').textContent = state.actual ? 'Fit to screen' : 'Actual size';
    $('#stage').classList.toggle('actual', state.actual);
    draw();
  });
  $('#undoBtn').addEventListener('click', () => { state.edits.pop(); startJob(true); });
  $('#preview').addEventListener('click', (e) => {
    if (!state.result || sim.playing) return;
    if (state.placing) {                       // putting new words where the click is
      const { at } = pointAt(e.clientX, e.clientY);
      const words = state.placing;
      stopPlacing();
      addEdit({ type: 'text', text: words.text, at, size: words.size, style: words.style, hex: words.hex });
      return;
    }
    const hit = areaAt(e.clientX, e.clientY);
    if (!hit) { closeAreaMenu(); return; }
    picked = hit;
    const menu = $('#areaMenu');
    const stage = $('#stage').getBoundingClientRect();
    menu.hidden = false;
    $('#areaOutline').hidden = hit.area.kind === 'run';
    $('#areaRetype').hidden = !hit.area.text;
    $('#retypeForm').hidden = true;
    menu.style.left = Math.max(4, Math.min(e.clientX - stage.left + $('#stage').scrollLeft, stage.width - 190)) + 'px';
    menu.style.top = Math.max(4, Math.min(e.clientY - stage.top - 20, stage.height - menu.offsetHeight - 4)) + $('#stage').scrollTop + 'px';
  });
  $('#areaCancel').addEventListener('click', closeAreaMenu);
  $('#areaRetype').addEventListener('click', () => {
    if (!picked) return;
    $('#retypeText').value = picked.area.words || '';
    $('#retypeForm').hidden = false;
    $('#areaRetype').hidden = true;
    $('#retypeText').focus();
  });
  const retype = () => picked && addEdit({ type: 'retype', at: picked.at,
    text: $('#retypeText').value.trim(), style: $('#retypeStyle').value });
  $('#retypeGo').addEventListener('click', retype);
  $('#retypeText').addEventListener('keydown', (e) => { if (e.key === 'Enter') retype(); });
  $('#lettersMode').addEventListener('change', (e) => { state.letters = e.target.value; startJob(true); });
  $('#aiAssistBox').addEventListener('change', (e) => { state.aiAssist = e.target.checked ? 'on' : 'off'; startJob(true); });
  $('#addWordsBtn').addEventListener('click', () => {
    const panel = $('#wordsPanel');
    panel.hidden = !panel.hidden;
    if (!panel.hidden) {
      const colors = (state.result ? state.result.colors : []).filter((c) => !c.label);
      if (colors.length) $('#wordsColor').value = colors[colors.length - 1].hex;
      $('#wordsText').focus();
    }
  });
  $('#wordsCancel').addEventListener('click', () => { $('#wordsPanel').hidden = true; stopPlacing(); });
  $('#wordsPlace').addEventListener('click', () => {
    const text = $('#wordsText').value.trim();
    if (!text || !state.result) { $('#wordsText').focus(); return; }
    const sizeMm = Number($('#wordsSize').value) * 25.4;
    state.placing = { text, size: sizeMm / state.result.ref, style: $('#wordsStyle').value,
      hex: $('#wordsColor').value };
    $('#wordsPanel').hidden = true;
    closeAreaMenu();
    $('#stage').classList.add('placing');
    $('#clickHint').textContent = 'Now click on the design where the words should go.';
  });
  $('#areaDelete').addEventListener('click', () => picked && addEdit({ type: 'delete', at: picked.at }));
  $('#areaOutline').addEventListener('click', () => picked && addEdit({ type: 'outline', at: picked.at }));
  $('#areaColor').addEventListener('click', () => {
    if (!picked) return;
    const picker = $('#areaPicker');
    const at = picked.at;
    picker.value = picked.area.hex;
    picker.onchange = () => addEdit({ type: 'color', at, hex: picker.value });
    picker.click();
  });
}

// ───────────────────────── screen 4: download ─────────────────────────
function renderDownload() {
  const r = state.result;
  if (!r) return;
  choiceButtons($('#machines'), state.options.machines.map((m) => ({
    id: m.id, name: m.name, detail: m.format.toUpperCase() + ' file',
  })), state.machine, (id) => { state.machine = id; remember('machine', id); renderDownload(); });

  const machine = state.options.machines.find((m) => m.id === state.machine) || state.options.machines[0];
  const fmt = machine.format;
  const file = r.files[fmt];
  const sel = $('#brandSelect');
  if (!sel.options.length) {
    for (const b of state.options.thread_brands) {
      const o = el('option', '', b.line);
      o.value = b.id;
      sel.append(o);
    }
  }
  sel.value = state.brand;
  const q = `brand=${encodeURIComponent(state.brand)}&machine=${encodeURIComponent(state.machine)}` +
    `&color=${encodeURIComponent(state.garment.slice(1))}`;
  $('#worksheetBtn').href = `api/jobs/${state.jobId}/worksheet.pdf?${q}`;
  $('#mockupBtn').href = `api/jobs/${state.jobId}/mockup.png?${q}&download=true`;
  $('#zipBtn').href = `api/jobs/${state.jobId}/all.zip?${q}`;
  $('#instructionsBtn').hidden = !r.project;
  $('#templatesBtn').hidden = !r.project || r.project.pieces.length === 0;
  const extra = $('#extraLinks');
  extra.innerHTML = '';
  for (const [key, e] of Object.entries((r.project && r.project.extras) || {})) {
    const fmt = (state.options.machines.find((m) => m.id === state.machine) || state.options.machines[0]).format;
    if (!e.files[fmt] || !e.files[fmt].ok) continue;
    const a = el('a', 'btn', `${e.title}: ${fmt.toUpperCase()} file`);
    a.href = `api/jobs/${state.jobId}/extra/${key}/${fmt}`;
    a.setAttribute('download', '');
    extra.append(a);
  }
  $('#instructionsBtn').href = `api/jobs/${state.jobId}/instructions.pdf`;
  $('#templatesBtn').href = `api/jobs/${state.jobId}/templates.pdf`;
  const btn = $('#downloadBtn');
  if (file && file.ok) {
    btn.href = `api/jobs/${state.jobId}/download/${fmt}`;
    btn.textContent = `Download ${fmt.toUpperCase()} file`;
    btn.style.pointerEvents = '';
    btn.style.opacity = '';
    $('#downloadNote').textContent = `${r.name}.${fmt} · checked: ${file.stitches.toLocaleString()} stitches read back correctly`;
  } else {
    btn.removeAttribute('href');
    btn.textContent = 'File not ready';
    btn.style.pointerEvents = 'none';
    btn.style.opacity = '.45';
    $('#downloadNote').textContent = "This file didn't pass its safety check. Go back and try again.";
  }

  const notes = [
    `Stabilizer for ${r.fabric.name.toLowerCase()}: ${r.fabric.stabilizer.toLowerCase()}.`,
    r.check.hoop ? `Hoop: ${r.check.hoop} or larger.` : 'This design does not fit any hoop yet. Go back and choose a smaller size.',
    `Finished size: ${r.stats.width_in} × ${r.stats.height_in} inches.`,
  ];
  for (const n of r.fabric.notes || []) notes.push(n);
  if (r.project) {
    notes.unshift(`${r.project.name}: the machine stops after each step. Follow the instructions sheet.`);
    for (const n of r.project.notes || []) notes.push(n);
  }
  if (fmt === 'dst') notes.push('DST files do not store colors. Follow the thread order below.');
  const ul = $('#sewNotes');
  ul.innerHTML = '';
  for (const n of notes) ul.append(el('li', '', n));

  const ol = $('#threadOrder');
  ol.innerHTML = '';
  let designColor = 0;                 // construction steps and the towel layer aren't numbered
  r.colors.forEach((c, i) => {
    const li = el('li');
    const chip = el('span', 'chip');
    chip.style.background = c.hex;
    const name = el('span');
    if (!c.label) designColor += 1;
    name.append(c.label || `Color ${designColor}`);
    if (!c.label && threadName(c)) name.append(el('em', '', threadName(c)));
    li.append(chip, name, el('small', '', `${c.stitches.toLocaleString()} stitches`));
    ol.appendChild(li);
  });
}

function setupScreen4() {
  $('#back4').addEventListener('click', () => go(3));
  $('#brandSelect').addEventListener('change', (e) => {
    state.brand = e.target.value;
    remember('brand', state.brand);
    renderDownload();
    renderThreadChips();
  });
  $('#again').addEventListener('click', () => {
    state.runToken++;
    state.result = null;
    state.cleanup = null;
    state.file = null;
    $('#thumb').hidden = true;
    $('#drop').classList.remove('has');
    $('#dropText').querySelector('strong').textContent = 'Drop a picture here';
    $('#next1').disabled = true;
    go(1);
  });
}

// ───────────────────────── my projects ─────────────────────────
// Every stitched design is saved automatically, with the choices made, so it can be reopened.
async function saveProject() {
  if (!state.cleanupId || !state.jobId) return;
  const form = new FormData();
  form.append('cleanup_id', state.cleanupId);
  form.append('job_id', state.jobId);
  form.append('settings_json', JSON.stringify({
    version: state.version, colors: state.colors, sizeIn: state.sizeIn, density: state.density,
    inside: state.inside, edits: state.edits, garment: state.garment, letters: state.letters,
    aiAssist: state.aiAssist,
  }));
  try { await api('api/projects', { method: 'POST', body: form }); } catch { /* saving is a bonus */ }
}

async function showProjects() {
  state.runToken++;
  go(5);
  const grid = $('#projectGrid');
  grid.innerHTML = '';
  let list = [];
  try { list = (await api('api/projects')).projects; } catch { /* shown as empty */ }
  $('#noProjects').hidden = list.length > 0;
  for (const pr of list) {
    const card = el('div', 'project');
    const img = el('img');
    img.src = `api/projects/${pr.id}/thumb.png?t=${Date.now()}`;
    img.alt = '';
    const sum = pr.summary || {};
    const when = new Date(pr.saved).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    card.append(img, el('strong', '', pr.name),
      el('p', '', `${sum.size_in || '?'} in · ${sum.colors || '?'} colors · ${(sum.stitches || 0).toLocaleString()} stitches`),
      el('p', '', when));
    const row = el('div', 'row');
    const open = el('button', '', 'Open');
    open.type = 'button';
    open.addEventListener('click', () => openProject(pr.id));
    const del = el('button', '', 'Delete');
    del.type = 'button';
    del.addEventListener('click', async () => {
      if (!window.confirm(`Delete "${pr.name}"? This can't be undone.`)) return;
      try { await api(`api/projects/${pr.id}`, { method: 'DELETE' }); } catch { /* list refresh shows the truth */ }
      showProjects();
    });
    row.append(open, del);
    card.append(row);
    grid.append(card);
  }
}

async function openProject(id) {
  try {
    const pr = await api(`api/projects/${id}/open`, { method: 'POST' });
    const st = pr.settings || {};
    state.runToken++;
    state.file = null;
    state.cleanupId = pr.cleanup_id;
    state.cleanup = pr.result;
    state.cleanupFor = 'project';
    state.imageStamp = Date.now();
    state.fabric = pr.fabric || state.fabric;
    state.version = st.version || pr.result.default;
    state.colors = st.colors || state.maxColors;
    state.sizeIn = st.sizeIn || null;
    state.density = st.density || 0;
    state.inside = st.inside || 'auto';
    state.edits = Array.isArray(st.edits) ? st.edits : [];
    state.letters = st.letters || 'auto';
    state.aiAssist = st.aiAssist || 'auto';
    state.garment = st.garment || state.garment;
    state.result = null;
    state.stitchedFor = null;
    if (pr.result.spend) state.ai = pr.result.spend;
    startJob(true);
  } catch (err) {
    window.alert(err.message);
  }
}

// ───────────────────────── settings ─────────────────────────
function renderSettings() {
  const s = state.ai;
  const useKey = s.provider_setting === 'api';
  choiceButtons($('#providers'), [
    { id: 'auto', name: 'My ChatGPT plan', detail: 'No extra cost' },
    { id: 'api', name: 'An OpenAI key', detail: 'Pay for each picture' },
  ], useKey ? 'api' : 'auto', (id) => saveSettings({ ai_provider: id }));
  $('#planBox').hidden = useKey;
  $('#keyBox').hidden = !useKey;
  $('#removeKey').hidden = !useKey || !s.has_key;
  $('#saveSettings').hidden = !useKey;

  const plan = $('#planStatus');
  plan.className = 'keyStatus' + (s.chatgpt === 'ready' ? ' ok' : '');
  plan.textContent = {
    ready: 'Signed in to ChatGPT. AI clean-up is on.',
    signed_out: "ChatGPT isn't signed in yet. Press Sign in, finish in the window that opens, then press Check again.",
    missing: "The ChatGPT helper program (Codex) isn't installed on this PC. Get it at chatgpt.com/codex, then press Check again.",
  }[s.chatgpt] || '';
  $('#signInBtn').hidden = s.chatgpt !== 'signed_out';
  $('#recheckBtn').hidden = s.chatgpt === 'ready';

  const status = $('#keyStatus');
  status.className = 'keyStatus' + (s.has_key ? ' ok' : '');
  status.textContent = s.has_key
    ? `A key is saved${s.key_hint ? ` (ends in ${s.key_hint.replace('…', '')})` : ''}. AI clean-up is on.`
    : 'No key saved yet.';
  $('#keyInput').value = '';
  $('#keyInput').placeholder = s.has_key ? 'Paste a new key to replace it' : 'sk-…';
  $('#capInput').value = s.daily_spend_cap_usd;
  $('#spendLine').textContent = `Spent today: $${s.spent_today_usd.toFixed(2)}. ` +
    `A picture costs about ${money(s.estimate_usd)}. When the limit is reached the app keeps working without AI.`;
  $('#settingsError').hidden = true;
}

function settingsError(text) {
  const p = $('#settingsError');
  p.textContent = text;
  p.hidden = false;
}

async function refreshAi() {
  try {
    state.ai = await api('api/settings?refresh=true');
    renderSettings();
    renderAiLine();
  } catch (err) { settingsError(err.message); }
}

function openSettings() {
  renderSettings();
  $('#settings').showModal();
  refreshAi();
}

async function saveSettings(body) {
  try {
    state.ai = await api('api/settings', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    renderSettings();
    renderAiLine();
    return true;
  } catch (err) {
    settingsError(err.message);
    return false;
  }
}

function setupSettings() {
  $('#settingsBtn').addEventListener('click', openSettings);
  $('#closeSettings').addEventListener('click', () => $('#settings').close());
  $('#removeKey').addEventListener('click', () => saveSettings({ remove_key: true }));
  $('#recheckBtn').addEventListener('click', refreshAi);
  $('#signInBtn').addEventListener('click', async () => {
    try { await api('api/chatgpt/signin', { method: 'POST' }); } catch (err) { settingsError(err.message); }
  });
  $('#saveSettings').addEventListener('click', async () => {
    const body = { daily_spend_cap_usd: $('#capInput').value };
    const key = $('#keyInput').value.trim();
    if (key) body.api_key = key;
    if (await saveSettings(body)) $('#settings').close();
  });
}

// ───────────────────────── start ─────────────────────────
async function init() {
  setupScreen1();
  setupScreen2();
  setupScreen3();
  setupScreen4();
  setupSettings();
  $('#projectsBtn').addEventListener('click', showProjects);
  $('#back5').addEventListener('click', () => go(state.result ? 3 : 1));
  try {
    state.options = await api('api/options');
    state.ai = await api('api/settings');
  } catch {
    showUploadError("We couldn't reach the digitizer. Close this page and start the app again.");
    return;
  }
  state.maxColors = state.options.max_colors;
  state.colors = state.maxColors;
  if (!state.options.machines.some((m) => m.id === state.machine)) state.machine = state.options.machines[0].id;
  if (!state.options.thread_brands.some((b) => b.id === state.brand)) state.brand = state.options.thread_brands[0].id;
  renderChoices();
  renderAiLine();
}

init();
