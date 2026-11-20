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
  ai: null,                 // { has_key, key_hint, daily_spend_cap_usd, spent_today_usd, estimate_usd }
  file: null,
  size: 'left_chest',
  sizeChosenByHand: false,
  fabric: 'tshirt',
  cleanupId: null,          // the picture as cleaned up on screen 2
  cleanupFor: null,         // which picture/size/garment that clean-up was made for
  cleanup: null,
  version: null,            // the chosen version
  stitchedFor: null,        // which version/colors the current stitches were made for
  colors: 6,
  maxColors: 6,
  jobId: null,
  result: null,
  machine: remembered('machine', 'brother'),
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
  for (const n of [1, 2, 3, 4]) $('#screen' + n).hidden = n !== step;
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
  choiceButtons($('#sizes'), state.options.sizes, state.size, (id) => {
    state.size = id;
    state.sizeChosenByHand = true;
    renderChoices();
  });
  $('#customRow').hidden = state.size !== 'custom';
  choiceButtons($('#fabrics'), state.options.fabrics, state.fabric, (id) => {
    state.fabric = id;
    // hats have their own size; pick it unless the user already chose a size themselves
    if (!state.sizeChosenByHand) state.size = id === 'hat' ? 'hat_front' : 'left_chest';
    renderChoices();
  });
}

function renderAiLine() {
  const p = $('#aiLine');
  p.innerHTML = '';
  if (!state.ai) return;
  if (state.ai.has_key) {
    p.textContent = `AI clean-up costs about ${money(state.ai.estimate_usd)} a picture. ` +
      `Today so far: $${state.ai.spent_today_usd.toFixed(2)} of $${state.ai.daily_spend_cap_usd.toFixed(2)}.`;
  } else {
    p.append('AI clean-up is off. ');
    const b = el('button', 'link', 'Add your OpenAI key');
    b.type = 'button';
    b.addEventListener('click', openSettings);
    p.append(b, ' to turn it on. It works without it too.');
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
    state.fabric, state.ai && state.ai.has_key].join('|');
}

async function startCleanup() {
  go(2);
  // Coming back with nothing changed? Keep the versions (and the choice) already made.
  if (state.cleanup && state.cleanupFor === cleanupKey()) { showCleanup(); return; }

  const token = ++state.runToken;
  state.cleanup = null;
  state.result = null;
  state.stitchedFor = null;
  showCleanupStage('working');
  $('#cLabel').textContent = 'Getting started…';
  const key = cleanupKey();

  const form = new FormData();
  form.append('file', state.file, state.file.name || 'design.png');
  form.append('size', state.size);
  form.append('custom_in', $('#customIn').value || '4');
  form.append('fabric', state.fabric);
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
  notice('#cNote', c.ai.note, ['no_key', 'bad_key', 'cap'].includes(c.ai.problem));
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
    const pick = () => { state.version = v.id; showCleanup(); };
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
  $('#cCost').textContent = c.ai.used && s
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
  for (const m of r.check.messages) ul.append(el('li', '', m.text));

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
  draw();
  renderDownload();
}

function shade(hex, amount) {
  const n = parseInt(hex.slice(1), 16);
  const ch = [n >> 16 & 255, n >> 8 & 255, n & 255].map((v) =>
    Math.round(amount < 0 ? v * (1 + amount) : v + (255 - v) * amount));
  return `rgb(${ch[0]},${ch[1]},${ch[2]})`;
}

// Draw every stitch as a thread-thick line, in sewing order so later thread covers earlier.
function draw() {
  const r = state.result;
  const cv = $('#preview');
  if (!r || cv.offsetParent === null) return;
  const W = cv.parentElement.clientWidth - 20;
  const H = Math.round(Math.max(260, Math.min(W, window.innerHeight * 0.62)));
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(W * dpr);
  cv.height = Math.round(H * dpr);
  cv.style.height = H + 'px';
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = state.garment;
  ctx.fillRect(0, 0, W, H);

  const [minx, miny, maxx, maxy] = r.bounds;
  const pad = 18;
  const s = Math.min((W - 2 * pad) / (maxx - minx), (H - 2 * pad) / (maxy - miny));
  const ox = (W - (maxx - minx) * s) / 2 - minx * s;
  const oy = (H - (maxy - miny) * s) / 2 - miny * s;
  const lw = Math.max(0.8, 0.38 * s);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (const block of r.blocks) {
    const dark = shade(block.hex, -0.28);
    const light = shade(block.hex, 0.35);
    for (const seg of block.segments) {
      for (let i = 0; i < seg.length - 2; i += 80) {          // 40 stitches at a time
        const end = Math.min(seg.length, i + 82);
        ctx.beginPath();
        ctx.moveTo(seg[i] * s + ox, seg[i + 1] * s + oy);
        for (let j = i + 2; j < end; j += 2) ctx.lineTo(seg[j] * s + ox, seg[j + 1] * s + oy);
        ctx.strokeStyle = dark; ctx.lineWidth = lw + 1; ctx.stroke();
        ctx.strokeStyle = block.hex; ctx.lineWidth = lw; ctx.stroke();
        if (lw >= 2.5) { ctx.strokeStyle = light; ctx.lineWidth = Math.max(0.6, lw * 0.25); ctx.stroke(); }
      }
    }
  }
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
  $('#fewer').addEventListener('click', () => changeColors(-1));
  $('#more').addEventListener('click', () => changeColors(1));
  $('#back3').addEventListener('click', () => { state.runToken++; go(2); showCleanup(); });
  $('#failBack').addEventListener('click', () => { state.runToken++; go(2); showCleanup(); });
  $('#next3').addEventListener('click', () => go(4));
  window.addEventListener('resize', draw);
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
  if (fmt === 'dst') notes.push('DST files do not store colors. Follow the thread order below.');
  const ul = $('#sewNotes');
  ul.innerHTML = '';
  for (const n of notes) ul.append(el('li', '', n));

  const ol = $('#threadOrder');
  ol.innerHTML = '';
  r.colors.forEach((c, i) => {
    const li = el('li');
    const chip = el('span', 'chip');
    chip.style.background = c.hex;
    li.append(chip, el('span', '', `Color ${i + 1}`), el('small', '', `${c.stitches.toLocaleString()} stitches`));
    ol.appendChild(li);
  });
}

function setupScreen4() {
  $('#back4').addEventListener('click', () => go(3));
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

// ───────────────────────── settings ─────────────────────────
function renderSettings() {
  const s = state.ai;
  const status = $('#keyStatus');
  status.className = 'keyStatus' + (s.has_key ? ' ok' : '');
  status.textContent = s.has_key
    ? `A key is saved${s.key_hint ? ` (ends in ${s.key_hint.replace('…', '')})` : ''}. AI clean-up is on.`
    : 'No key saved. AI clean-up is off.';
  $('#keyInput').value = '';
  $('#keyInput').placeholder = s.has_key ? 'Paste a new key to replace it' : 'sk-…';
  $('#capInput').value = s.daily_spend_cap_usd;
  $('#removeKey').hidden = !s.has_key;
  $('#spendLine').textContent = `Spent today: $${s.spent_today_usd.toFixed(2)}. ` +
    `A picture costs about ${money(s.estimate_usd)}. When the limit is reached the app keeps working without AI.`;
  $('#settingsError').hidden = true;
}

function openSettings() {
  renderSettings();
  $('#settings').showModal();
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
    const p = $('#settingsError');
    p.textContent = err.message;
    p.hidden = false;
    return false;
  }
}

function setupSettings() {
  $('#settingsBtn').addEventListener('click', openSettings);
  $('#closeSettings').addEventListener('click', () => $('#settings').close());
  $('#removeKey').addEventListener('click', () => saveSettings({ remove_key: true }));
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
  renderChoices();
  renderAiLine();
}

init();
