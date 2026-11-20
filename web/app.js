// AI Embroidery Digitizer — the four screens. Plain JavaScript, no build step.
'use strict';

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

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
    b.textContent = item.name;
    if (item.detail) b.appendChild(el('small', '', item.detail));
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
  const lw = 0.47 * s;
  const c = sim.cursor;
  while (sim.drawn < n && c.b < r.blocks.length) {
    const block = r.blocks[c.b];
    const seg = block.segments[c.sg];
    if (!seg) { c.b += 1; c.sg = 0; c.p = 0; continue; }
    const points = seg.length / 2;
    if (c.p >= points) { c.sg += 1; c.p = 0; continue; }
    const end = Math.min(points, c.p + Math.min(40, n - sim.drawn));
    const from = Math.max(0, c.p - 1);                   // join on to the stitch before
    g.beginPath();
    g.moveTo(seg[from * 2] * s + sim.ox, seg[from * 2 + 1] * s + sim.oy);
    for (let q = from + 1; q < end; q += 1) g.lineTo(seg[q * 2] * s + sim.ox, seg[q * 2 + 1] * s + sim.oy);
    g.strokeStyle = shade(block.hex, -0.16); g.lineWidth = lw * 1.1; g.stroke();   // soft edge
    g.strokeStyle = block.hex; g.lineWidth = lw * 0.9; g.stroke();                 // the thread
    g.globalAlpha = 0.5;
    g.strokeStyle = shade(block.hex, 0.22); g.lineWidth = lw * 0.28; g.stroke();   // a little sheen
    g.globalAlpha = 1;
    sim.drawn += end - c.p;
    sim.last = [seg[(end - 1) * 2], seg[(end - 1) * 2 + 1]];
    c.p = end;
  }
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
function areaAt(clientX, clientY) {
  const r = state.result;
  const cv = $('#preview');
  const box = cv.getBoundingClientRect();
  const dpr = cv.width / box.width;
  const x = (((clientX - box.left) * dpr) * sim.k - sim.ox) / sim.s;     // millimetres
  const y = (((clientY - box.top) * dpr) * sim.k - sim.oy) / sim.s;
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

function renderThreadChips() {
  const box = $('#threadChips');
  box.innerHTML = '';
  for (const c of state.result.colors) {
    if (c.label) continue;                               // the towel's flattening layer
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
    const hit = areaAt(e.clientX, e.clientY);
    if (!hit) { closeAreaMenu(); return; }
    picked = hit;
    const menu = $('#areaMenu');
    const stage = $('#stage').getBoundingClientRect();
    menu.hidden = false;
    $('#areaOutline').hidden = hit.area.kind === 'run';
    menu.style.left = Math.max(4, Math.min(e.clientX - stage.left + $('#stage').scrollLeft, stage.width - 190)) + 'px';
    menu.style.top = Math.max(4, Math.min(e.clientY - stage.top - 20, stage.height - menu.offsetHeight - 4)) + $('#stage').scrollTop + 'px';
  });
  $('#areaCancel').addEventListener('click', closeAreaMenu);
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
    inside: state.inside, edits: state.edits, garment: state.garment,
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
