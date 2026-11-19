// AI Embroidery Digitizer — the three screens. Plain JavaScript, no build step.
'use strict';

const $ = (sel) => document.querySelector(sel);

const GARMENT_COLORS = ['#eeece6', '#9aa0a8', '#23262d', '#1f2a44', '#8f1f24'];

const remembered = (key, fallback) => {
  try { return localStorage.getItem(key) || fallback; } catch { return fallback; }
};
const remember = (key, value) => {
  try { localStorage.setItem(key, value); } catch { /* private window: fine, just not remembered */ }
};

const state = {
  options: null,
  file: null,
  size: 'left_chest',
  sizeChosenByHand: false,
  fabric: 'tshirt',
  colors: 6,
  maxColors: 6,
  jobId: null,
  result: null,
  machine: remembered('machine', 'brother'),
  garment: GARMENT_COLORS[0],
  runToken: 0,
};

// ───────────────────────── moving between screens ─────────────────────────
function go(step) {
  for (const n of [1, 2, 3]) $('#screen' + n).hidden = n !== step;
  document.querySelectorAll('.steps li').forEach((li) => {
    const n = Number(li.dataset.step);
    li.classList.toggle('current', n === step);
    li.classList.toggle('done', n < step);
  });
  window.scrollTo(0, 0);
  if (step === 2 && state.result) draw();
}

// ───────────────────────── screen 1 ─────────────────────────
function choiceButtons(box, items, current, onPick) {
  box.innerHTML = '';
  for (const item of items) {
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.id = item.id;
    b.textContent = item.name;
    if (item.detail) {
      const s = document.createElement('small');
      s.textContent = item.detail;
      b.appendChild(s);
    }
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

function showUploadError(text) {
  const el = $('#uploadError');
  el.textContent = text || '';
  el.hidden = !text;
}

function setFile(file) {
  if (!file) return;
  if (!/^image\/(png|jpeg|webp)$/.test(file.type)) {
    showUploadError("That file isn't a picture we can use. Please choose a PNG, JPG or WEBP.");
    return;
  }
  showUploadError('');
  state.file = file;
  state.colors = state.maxColors;      // a new picture starts from the default again
  const img = $('#thumb');
  if (img.src) URL.revokeObjectURL(img.src);
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  $('#drop').classList.add('has');
  $('#dropText').querySelector('strong').textContent = 'Click or drop to use a different picture';
  $('#next1').disabled = false;
}

function setupUpload() {
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
}

// ───────────────────────── screen 2 ─────────────────────────
function showStage(which) {
  $('#working').hidden = which !== 'working';
  $('#failed').hidden = which !== 'failed';
  $('#ready').hidden = which !== 'ready';
  $('#next2').disabled = which !== 'ready';
}

async function startJob() {
  const token = ++state.runToken;
  state.result = null;
  go(2);
  showStage('working');
  $('#workLabel').textContent = 'Getting started…';

  const form = new FormData();
  form.append('file', state.file, state.file.name || 'design.png');
  form.append('size', state.size);
  form.append('custom_in', $('#customIn').value || '4');
  form.append('fabric', state.fabric);
  form.append('colors', String(state.colors));

  try {
    const res = await fetch('api/jobs', { method: 'POST', body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error);
    state.jobId = body.id;
    while (token === state.runToken) {
      const job = await (await fetch('api/jobs/' + state.jobId)).json();
      if (token !== state.runToken) return;
      if (job.status === 'done') { state.result = job.result; showResult(); return; }
      if (job.status === 'error' || job.error) throw new Error(job.error);
      $('#workLabel').textContent = job.label || 'Working…';
      await new Promise((r) => setTimeout(r, 350));
    }
  } catch (err) {
    if (token !== state.runToken) return;
    const known = err && err.message && !/fetch|network|JSON|undefined/i.test(err.message);
    $('#failText').textContent = known ? err.message
      : "We couldn't reach the digitizer. Make sure its window is still open, then try again.";
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

  const light = $('#light');
  light.className = 'light ' + r.check.status;
  $('#lightText').textContent = LIGHT_TEXT[r.check.status];
  const ul = $('#messages');
  ul.innerHTML = '';
  for (const m of r.check.messages) {
    const li = document.createElement('li');
    li.textContent = m.text;
    ul.appendChild(li);
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
  for (const [k, v] of rows) {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    dl.append(dt, dd);
  }
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

function setupScreen2() {
  const box = $('#garmentColors');
  box.append('Garment color:');
  for (const c of GARMENT_COLORS) {
    const b = document.createElement('button');
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
    startJob();
  };
  $('#fewer').addEventListener('click', () => changeColors(-1));
  $('#more').addEventListener('click', () => changeColors(1));
  $('#back2').addEventListener('click', () => { state.runToken++; go(1); });
  $('#failBack').addEventListener('click', () => { state.runToken++; go(1); });
  $('#next2').addEventListener('click', () => go(3));
  window.addEventListener('resize', draw);
}

// ───────────────────────── screen 3 ─────────────────────────
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
  for (const n of notes) { const li = document.createElement('li'); li.textContent = n; ul.appendChild(li); }

  const ol = $('#threadOrder');
  ol.innerHTML = '';
  r.colors.forEach((c, i) => {
    const li = document.createElement('li');
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.style.background = c.hex;
    const label = document.createElement('span');
    label.textContent = `Color ${i + 1}`;
    const small = document.createElement('small');
    small.textContent = `${c.stitches.toLocaleString()} stitches`;
    li.append(chip, label, small);
    ol.appendChild(li);
  });
}

function setupScreen3() {
  $('#back3').addEventListener('click', () => go(2));
  $('#again').addEventListener('click', () => {
    state.runToken++;
    state.result = null;
    state.file = null;
    $('#thumb').hidden = true;
    $('#drop').classList.remove('has');
    $('#dropText').querySelector('strong').textContent = 'Drop a picture here';
    $('#next1').disabled = true;
    go(1);
  });
}

// ───────────────────────── start ─────────────────────────
async function init() {
  setupUpload();
  setupScreen2();
  setupScreen3();
  $('#next1').addEventListener('click', () => { if (state.file) startJob(); });
  try {
    state.options = await (await fetch('api/options')).json();
  } catch {
    showUploadError("We couldn't reach the digitizer. Close this page and start the app again.");
    return;
  }
  state.maxColors = state.options.max_colors;
  state.colors = state.maxColors;
  if (!state.options.machines.some((m) => m.id === state.machine)) state.machine = state.options.machines[0].id;
  renderChoices();
}

init();
