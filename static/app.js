const fileInput   = document.getElementById('fileInput');
const uploadZone  = document.getElementById('uploadZone');
const fileInfo    = document.getElementById('fileInfo');
const fileName    = document.getElementById('fileName');
const fileMeta    = document.getElementById('fileMeta');
const runBtn      = document.getElementById('runBtn');
const pulseFill   = document.getElementById('pulseFill');
const statusEl    = document.getElementById('status');
const snrBefore   = document.getElementById('snrBefore');
const snrAfter    = document.getElementById('snrAfter');
const improveEl   = document.getElementById('improve');
const playBefore  = document.getElementById('playBefore');
const playAfter   = document.getElementById('playAfter');
const stopBtn     = document.getElementById('stopBtn');
const downloadBtn = document.getElementById('downloadBtn');
const audioPlayer = document.getElementById('audioPlayer');

let currentFileId = null;

// ── Upload zone click ────────────────────────────────────────
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── Handle file upload ───────────────────────────────────────
async function handleFile(file) {
  setStatus('Uploading file...', 'processing');
  const form = new FormData();
  form.append('file', file);

  try {
    const res  = await fetch('/upload', { method: 'POST', body: form });
    const data = await res.json();

    currentFileId = data.file_id;
    fileName.textContent = data.filename;
    fileMeta.textContent = `${data.duration}s · 16kHz · mono`;
    fileInfo.style.display = 'block';
    runBtn.disabled = false;

    drawWave('waveBefore', '#F5A623', true);
    drawWave('waveAfter',  '#3DD68C', false);

    playBefore.disabled = false;
    setStatus('File loaded — click Apply Noise Reduction.', 'success');
  } catch (e) {
    setStatus('Upload failed. Please try again.', 'error');
  }
}

// ── Run denoising ────────────────────────────────────────────
runBtn.addEventListener('click', async () => {
  if (!currentFileId) return;

  runBtn.disabled = true;
  runBtn.textContent = 'Processing...';
  pulseFill.classList.add('animating');
  setStatus('Pass 1 — non-stationary sweep...', 'processing');

  try {
    const res  = await fetch('/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_id: currentFileId })
    });
    const data = await res.json();

    snrBefore.textContent = `${data.snr_before} dB`;
    snrAfter.textContent  = `${data.snr_after} dB`;
    const sign = data.improvement >= 0 ? '+' : '';
    improveEl.textContent = `▲ ${sign}${data.improvement} dB improvement across 3 passes`;

    playAfter.disabled  = false;
    downloadBtn.disabled = false;

    drawWave('waveAfter', '#3DD68C', false);
    setStatus('✔  Noise removed — use Before/After to compare.', 'success');
  } catch (e) {
    setStatus('Processing failed. Please try again.', 'error');
  } finally {
    runBtn.disabled  = false;
    runBtn.textContent = '⚡ Apply Noise Reduction';
    pulseFill.classList.remove('animating');
  }
});

// ── Playback ─────────────────────────────────────────────────
playBefore.addEventListener('click', () => {
  audioPlayer.src = `/audio/original/${currentFileId}`;
  audioPlayer.play();
  playBefore.textContent = '▶ Playing...';
  playAfter.textContent  = '▶ Play';
  setStatus('▶ Playing original noisy audio...', 'processing');
  setTimeout(() => playBefore.textContent = '▶ Play', 3000);
});

playAfter.addEventListener('click', () => {
  audioPlayer.src = `/audio/cleaned/${currentFileId}`;
  audioPlayer.play();
  playAfter.textContent  = '▶ Playing...';
  playBefore.textContent = '▶ Play';
  setStatus('▶ Playing cleaned audio...', 'processing');
  setTimeout(() => playAfter.textContent = '▶ Play', 3000);
});

stopBtn.addEventListener('click', () => {
  audioPlayer.pause();
  audioPlayer.currentTime = 0;
  playBefore.textContent = '▶ Play';
  playAfter.textContent  = '▶ Play';
  setStatus('⏹ Playback stopped.', '');
});

downloadBtn.addEventListener('click', () => {
  window.location.href = `/download/${currentFileId}`;
});

// ── Waveform drawing ─────────────────────────────────────────
function drawWave(id, color, noisy) {
  const canvas = document.getElementById(id);
  const ctx    = canvas.getContext('2d');
  canvas.width  = canvas.offsetWidth;
  canvas.height = 64;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  const pts = 200;
  for (let i = 0; i < pts; i++) {
    const x = (i / pts) * canvas.width;
    let amp;
    if (noisy) {
      amp = (Math.random() - 0.5) * 48;
    } else {
      const t = i / pts;
      const speech = (t > 0.15 && t < 0.9)
        ? Math.sin(t * 45) * 18 + Math.sin(t * 20) * 8
        : 0;
      amp = speech + (Math.random() - 0.5) * 3;
    }
    const y = 32 + amp;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// ── Status helper ────────────────────────────────────────────
function setStatus(msg, type) {
  statusEl.textContent  = msg;
  statusEl.className    = `status ${type}`;
}
// ── Theme toggle ──────────────────────────────────────────────
function toggleTheme() {
  const body   = document.body;
  const btn    = document.getElementById('themeToggle');
  const isLight = body.classList.toggle('light');

  btn.textContent = isLight ? '🌙 Dark' : '☀️ Light';

  // Save preference
  localStorage.setItem('theme', isLight ? 'light' : 'dark');
}

// Load saved theme on page load
(function() {
  const saved = localStorage.getItem('theme');
  if (saved === 'light') {
    document.body.classList.add('light');
    const btn = document.getElementById('themeToggle');
    if (btn) btn.textContent = '🌙 Dark';
  }
})();
// ── Particle background ───────────────────────────────────────
(function () {
  const canvas = document.getElementById('particles');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, pts = [];

  function resize() {
    W = canvas.offsetWidth;
    H = canvas.offsetHeight;
    canvas.width  = W;
    canvas.height = H;
  }

  function rand(a, b) { return a + Math.random() * (b - a); }

  function init() {
    pts = [];
    for (let i = 0; i < 60; i++) {
      pts.push({
        x:  rand(0, W),
        y:  rand(0, H),
        vx: rand(-0.25, 0.25),
        vy: rand(-0.25, 0.25),
        r:  rand(1.5, 3),
        a:  rand(0.3, 0.8)
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // dots
    pts.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(124,110,245,${p.a})`;
      ctx.fill();
    });

    // connecting lines
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const dx   = pts[i].x - pts[j].x;
        const dy   = pts[i].y - pts[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 110) {
          ctx.beginPath();
          ctx.moveTo(pts[i].x, pts[i].y);
          ctx.lineTo(pts[j].x, pts[j].y);
          ctx.strokeStyle = `rgba(124,110,245,${0.15 * (1 - dist / 110)})`;
          ctx.lineWidth   = 0.6;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', () => { resize(); init(); });
  setTimeout(() => {
  resize();
  init();
  draw();
}, 100);
})();