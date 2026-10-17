/* Vearl, site behaviour: navigation, reveals, ticker, ASCII pearl, Freivalds console, closing ripples. */
(() => {
  'use strict';
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $ = (s) => document.querySelector(s);

  /* ---------------------------------------------------------------- contact address, assembled here so that it is not in the page source for address harvesters */
  document.querySelectorAll('.mail').forEach((s) => {
    const addr = s.dataset.u + '@' + s.dataset.d, a = document.createElement('a');
    a.href = 'mailto:' + addr; a.textContent = s.dataset.label || addr; a.className = 'mail-link';
    s.replaceWith(a);
  });

  /* ---------------------------------------------------------------- navigation */
  const nav = $('#nav'), burger = $('#burger'), menu = $('#menu');
  const onScroll = () => nav.classList.toggle('solid', window.scrollY > 24);
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
  const setMenu = (open) => {
    nav.classList.toggle('open', open);
    burger.setAttribute('aria-expanded', String(open));
    burger.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
  };
  burger.addEventListener('click', () => setMenu(!nav.classList.contains('open')));
  menu.querySelectorAll('a').forEach((a) => a.addEventListener('click', () => setMenu(false)));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setMenu(false); });
  window.addEventListener('resize', () => { if (window.innerWidth > 960) setMenu(false); });

  // current section highlighted in the navigation
  const links = [...menu.querySelectorAll('a')];
  const byId = new Map(links.map((a) => [a.getAttribute('href').slice(1), a]));
  if ('IntersectionObserver' in window) {
    const spy = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a) => a.classList.remove('active'));
        const a = byId.get(e.target.id);
        if (a) a.classList.add('active');
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    document.querySelectorAll('main section[id]').forEach((s) => spy.observe(s));
  }

  // deep links (/#faq, /#workloads...): land on the section at once instead of a long smooth scroll from the top
  if (location.hash.length > 1) {
    const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (target) {
      const html = document.documentElement, prev = html.style.scrollBehavior;
      html.style.scrollBehavior = 'auto';
      target.scrollIntoView();
      html.style.scrollBehavior = prev;
      target.querySelectorAll('.reveal').forEach((el) => el.classList.add('in'));
    }
  }

  /* ---------------------------------------------------------------- reveal on scroll */
  const reveals = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window && !reduce) {
    const io = new IntersectionObserver((entries) => entries.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    }), { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
    reveals.forEach((el) => io.observe(el));
  } else {
    reveals.forEach((el) => el.classList.add('in'));
  }

  /* ---------------------------------------------------------------- ticker: figures measured on the testnet */
  const facts = [
    ['Network', 'Closed testnet', true], ['TinyStories-33M', 'writes stories · every word verified'], ['DeiT-Tiny', '224 × 224 · ≈ 1.7 s / photo'], ['BGE-base', '110M params · ≈ 3.3 s / text'], ['MiniLM-L6', '≈ 1.0 s / text'],
    ['Verified GEMM', '8192³'], ['Signatures', 'SLH-DSA · FIPS 205'], ['Difficulty', 'ASERT'],
    ['Tests', '171 passing', true], ['Outputs', 'bit-identical'], ['Reference GPU', 'NVIDIA A10'], ['Tokens', 'no value']
  ];
  const row = facts.map(([k, v, hi]) => `<div class="tk">${k}<b class="${hi ? 'hi' : ''}">${v}</b></div>`).join('');
  $('#ticker').innerHTML = row + row;

  /* ---------------------------------------------------------------- hero: characters that form the code of the chain */
  const HUES = 12;
  const STOPS = [[0, [217, 255, 247]], [0.45, [159, 240, 226]], [0.75, [185, 166, 255]], [1, [244, 188, 214]]];
  const nacre = (t) => {
    t = Math.min(1, Math.max(0, t));
    for (let k = 0; k < STOPS.length - 1; k++) {
      const [a, ca] = STOPS[k], [b, cb] = STOPS[k + 1];
      if (t <= b) { const u = (t - a) / (b - a); return ca.map((v, i) => Math.round(v + (cb[i] - v) * u)); }
    }
    return STOPS[STOPS.length - 1][1];
  };
  const palette = Array.from({ length: HUES }, (_, i) => nacre(i / (HUES - 1)));

  const cv = $('#pearl'), cx = cv.getContext('2d'), slot = $('#scene'), hero0 = $('.hero');
  const MONO = '"IBM Plex Mono", ui-monospace, monospace';
  const SCR = '@#%*+=-:.';                                  // characters a cell passes through while it changes
  const RAMP = ' .·:-=+*%#@';
  // colour slots of a scene: 1 lines, 2 mint, 3 lavender, 4 rose, 5 bright, 6 soft text
  const SCOL = [null, [96, 106, 124], [159, 240, 226], [185, 166, 255], [244, 188, 214], [245, 244, 238], [176, 182, 196]];
  const SW = 50, SH = 16;                                   // the scene is a grid of characters
  let W = 0, H = 0, cw = 0, ch = 0, cols = 0, rows = 0, running = false, visible = true;
  let scw = 0, sch = 0, sfs = 12, sx = 0, sy = 0, px = -1e4, py = -1e4;
  const fsOf = () => (W < 600 ? 9 : 12);

  /* ---- scenes: real names from the node's code, drawn with characters ---- */
  const blank = () => ({ c: new Array(SW * SH).fill(' '), k: new Array(SW * SH).fill(0) });
  const put = (b, x, y, s, k) => {
    if (y < 0 || y >= SH) return;
    for (let i = 0; i < s.length; i++) { const xx = x + i; if (xx >= 0 && xx < SW && s[i] !== ' ') { b.c[y * SW + xx] = s[i]; b.k[y * SW + xx] = k; } }
  };
  const rh = (a, b, c) => { let h = (Math.imul(a, 374761393) + Math.imul(b, 668265263) + Math.imul(c, 2147483647)) | 0; h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967296; };
  const hexs = (seed, n) => { let s = ''; for (let i = 0; i < n; i++) s += '0123456789abcdef'[Math.floor(rh(seed, i, 7) * 16)]; return s; };
  const KW = /^(for|in|let|if|fn|mut|pub|as)$/;
  function code(b, x, y, line, hot) {                        // a small syntax highlighter
    const re = /(\/\/.*$)|([A-Za-z_][A-Za-z0-9_]*)(\()?|(\d+)|(.)/g; let m, p = x;
    while ((m = re.exec(line))) {
      if (m[1]) { put(b, p, y, m[1], hot ? 6 : 1); p += m[1].length; }
      else if (m[2]) { put(b, p, y, m[2], hot ? 5 : KW.test(m[2]) ? 3 : m[3] ? 2 : 6); p += m[2].length; if (m[3]) { put(b, p, y, '(', hot ? 5 : 6); p++; } }
      else if (m[4]) { put(b, p, y, m[4], hot ? 5 : 4); p += m[4].length; }
      else { put(b, p, y, m[5], hot ? 5 : 6); p++; }
    }
  }

  function scenePearl(t) {                                   // the brand pearl, then it turns into something useful
    const b = blank(), cxs = SW / 2, cys = 6.6, R = 6.3;
    let lx = Math.cos(t * 0.45) * 0.7, ly = -0.35 + Math.sin(t * 0.45) * 0.3, lz = 0.85; const ln = Math.hypot(lx, ly, lz); lx /= ln; ly /= ln; lz /= ln;
    for (let r = 0; r < 14; r++) for (let c = 0; c < SW; c++) {
      const dx = (c - cxs) * 0.56 / R, dy = (r - cys) / R, d2 = dx * dx + dy * dy, i = r * SW + c;
      if (d2 < 1) {
        const nz = Math.sqrt(1 - d2), diff = Math.max(0, dx * lx + dy * ly + nz * lz);
        const spec = Math.pow(Math.max(0, (dx * lx + dy * ly + nz * (lz + 1)) / Math.hypot(lx, ly, lz + 1)), 30);
        const swirl = Math.sin(3 * Math.atan2(dy, dx) + t * 0.5 + nz * 4) * 0.5 + 0.5;
        const v = 0.22 + diff * 0.72 + spec * 0.9 + swirl * 0.08, hue = (nz * 0.6 + swirl * 0.5 + t * 0.04) % 1;
        const k = Math.min(RAMP.length - 1, Math.floor(v * (RAMP.length - 1)));
        if (k > 0) { b.c[i] = RAMP[k]; b.k[i] = hue < 0.34 ? 2 : hue < 0.68 ? 3 : 4; }
      } else if (d2 < 2.4) {
        const w = Math.sin(Math.sqrt(d2) * 9 - t * 2.2);
        if (w > 0.55) { b.c[i] = '.'; b.k[i] = 1; }
      }
    }
    put(b, 10, 14, 'v e a r l', 5); put(b, 21, 14, '·  proof of useful work', 1);
    return b;
  }

  function sceneTile(t) {                                     // one mining attempt per tile of the product
    const b = blank(), q = 7, cyc = Math.floor(t * 8 / 49), step = Math.floor(t * 8) % 49, ti = Math.floor(step / q), tj = step % q;
    const hit = cyc % 2 === 0 && step === 33;
    code(b, 0, 0, "// C' = A' . B'   int8 x int8 -> int32");
    put(b, 0, 2, 'A + noise', 1); put(b, 17, 2, 'B + noise', 1); put(b, 34, 2, "C'", 1);
    const cell = (r, c, x0, seed, on) => {
      const v = rh(seed, r * q + c, Math.floor(t * 1.4 + rh(seed, r, c) * 3)), s = '.:+*#@'[Math.floor(v * 5.99)];
      put(b, x0 + c * 2, 3 + r, s + s, on ? 5 : rh(seed, r * 9, c) > 0.5 ? 2 : 3);
    };
    for (let r = 0; r < q; r++) for (let c = 0; c < q; c++) {
      cell(r, c, 0, 1, r === ti);
      cell(r, c, 17, 2, c === tj);
      const k = r * q + c, x = 34 + c * 2;
      if (k === step) put(b, x, 3 + r, '@@', 5);
      else if (k < step) put(b, x, 3 + r, rh(9, k, cyc) > 0.66 ? '##' : rh(9, k, cyc) > 0.33 ? '%%' : '**', k === 33 && cyc % 2 === 0 ? 2 : rh(8, k, cyc) > 0.7 ? 3 : 1);
      else put(b, x, 3 + r, '··', 1);
    }
    put(b, 15, 6, 'x', 6); put(b, 32, 6, '=', 6);
    code(b, 0, 11, 'jackpot = blake3(sigma, ti, tj, fold);');
    put(b, 0, 12, `tile (${ti}, ${tj})`, 6);
    const hx = hit ? '00' + hexs(step + cyc, 14) : hexs(step * 7 + cyc, 16);
    put(b, 14, 12, '0x' + hx, hit ? 2 : 6);
    if (hit) put(b, 0, 14, 'jackpot < target  ->  this tile is a block', 2); else put(b, 0, 14, 'jackpot > target  ->  next tile', 1);
    return b;
  }

  function sceneCode(t) {                                     // the real functions of the proof of work
    const b = blank();
    const L = ['// work.rs : one attempt = one 16x16 tile', 'let noise = derive_noise(&sigma, &p);', 'for s in 0..n / fold_k {', '    acc += a_rows[..] * b_cols[..];', '    fold_step(&mut fold, &acc, s);', '}',
      'let jackpot = jackpot_from_fold(', '    &sigma, ti, tj, &fold);', 'if meets_target(&jackpot, target) {', '    // this tile is a block', '}', '', '// decode(): C = A.B, what the client paid'];
    const hot = Math.floor(t * 2.2) % 11;
    L.forEach((s, i) => { code(b, 3, 1 + i, s, i === hot && s.trim() && !s.startsWith('//')); put(b, 0, 1 + i, String(i + 1).padStart(2, ' '), 1); });
    put(b, 0, 15, 'fold_step() keeps the hash cost negligible', 1);
    return b;
  }

  function sceneMerkle(t) {                                   // what a validator checks to accept a block
    const b = blank(), k = Math.floor(t / 1.3) % 4, id = Math.floor(t / 4), side = k < 2 ? 0 : 1;
    const hl = (x1, x2, y, col) => { for (let x = x1; x <= x2; x++) put(b, x, y, '-', col); };
    code(b, 0, 0, 'verify_row(&root, i, &row, &path);');
    const cR = 24, top = [12, 36], leaf = [6, 18, 30, 42];
    put(b, cR - 3, 2, 'a_root', 2);
    // root -> the two halves
    hl(top[0], top[1], 3, 1); hl(top[side] < cR ? top[side] : cR, top[side] < cR ? cR : top[side], 3, 2);
    put(b, top[0], 3, '.', side === 0 ? 2 : 1); put(b, top[1], 3, '.', side === 1 ? 2 : 1); put(b, cR, 3, "'", 2);
    top.forEach((x, n) => { put(b, x, 4, '|', n === side ? 2 : 4); put(b, x - 2, 5, hexs(n * 31 + id, 4), n === side ? 2 : 4); });
    // each half -> its two rows
    top.forEach((c, n) => {
      const l0 = leaf[n * 2], l1 = leaf[n * 2 + 1], mine = n === side, kk = k - n * 2;
      hl(l0, l1, 6, 1);
      if (mine) hl(kk === 0 ? l0 : c, kk === 0 ? c : l1, 6, 2);
      put(b, l0, 6, '.', mine && kk === 0 ? 2 : 1); put(b, l1, 6, '.', mine && kk === 1 ? 2 : 1); put(b, c, 6, "'", mine ? 2 : 1);
      [l0, l1].forEach((x, m) => {
        const me = mine && kk === m, sb = mine && kk !== m;
        put(b, x, 7, '|', me ? 2 : sb ? 4 : 1); put(b, x - 2, 8, hexs((n * 2 + m) * 17 + id + 3, 4), me ? 2 : sb ? 4 : 1); put(b, x - 2, 9, 'row ' + (n * 2 + m), me ? 5 : 1);
      });
    });
    put(b, 0, 11, 'green: the row being proven', 2); put(b, 0, 12, 'pink : siblings sent in the proof', 4);
    put(b, 0, 14, 'block proof = 16 rows of A + 16 columns of B', 6);
    put(b, 0, 15, 'recompute ONE tile, not the whole product', 1);
    return b;
  }

  function sceneFraud(t) {                                    // how a wrong result gets caught
    const b = blank(), tt = t % 6, r = Math.min(24, Math.floor(tt * 6));
    code(b, 0, 0, 'freivalds(a, b, c, n, rounds, seed);');
    put(b, 0, 2, 'A . (B . x)   =?=   C . x        cost O(n^2)', 6);
    put(b, 0, 4, 'x = random vector, repeated 24 times', 1);
    put(b, 0, 6, '[' + '#'.repeat(r) + '.'.repeat(24 - r) + ']  ' + r + '/24', r === 24 ? 2 : 5);
    put(b, 0, 7, r === 24 ? 'all rounds agree: error <= 2^-24' : 'checking the worker result...', r === 24 ? 2 : 1);
    put(b, 0, 9, r === 24 ? 'ACCEPT  ->  worker is paid' : '', 2);
    code(b, 0, 11, 'find_bad_entry(a, b, c, n, seed) // if they differ');
    put(b, 0, 12, 'Fraud { i, j, a_row, b_col, c_row, paths }', 6);
    put(b, 0, 13, 'one wrong value is enough, checked in O(n)', 1);
    put(b, 0, 15, 'bond slashed: half to the challenger, half burned', 4);
    return b;
  }

  function sceneChain(t) {                                    // blocks, and what decides between two chains
    const b = blank(), n0 = 1038 + Math.floor(t / 2.4);
    for (let k = 0; k < 4; k++) {
      const n = n0 + k, job = rh(n, 1, 1) > 0.4, x = 1 + k * 12, col = job ? 2 : 3, last = k === 3 && (t % 2.4) < 0.7;
      put(b, x, 0, '+------+', last ? 5 : col); put(b, x, 1, '|#' + n + '|', last ? 5 : col); put(b, x, 2, '|' + (job ? ' job  ' : 'native') + '|', last ? 5 : col); put(b, x, 3, '+------+', last ? 5 : col);
      if (k < 3) put(b, x + 8, 1, '---', 1);
    }
    code(b, 0, 6, 'weight = 2^64 / (target + 1);');
    put(b, 0, 7, 'the chain with the most cumulative work wins', 6);
    code(b, 0, 9, 'asert(anchor, dt, dh, halflife = 900);');
    put(b, 0, 10, 'target 10 s per block, no window to game', 6);
    code(b, 0, 12, 'reward = base + min(base / 2, rebate_left);');
    put(b, 0, 13, 'rebate <= what the client burned', 6);
    put(b, 0, 15, 'native: security  |  job: security + result', 1);
    return b;
  }

  function sceneProgram(t) {                                  // an AI model as verified operations
    const b = blank(), ops = [['Gemm', 'W . x >> shift   (the mined work)'], ['Bias', '+ b'], ['Lut', 'GELU(x), 128-entry table'], ['GemmAT', 'K^T . Q  attention scores'], ['CausalSoftmax', 'per column, integer'],
      ['GemmAT', 'V . A'], ['LayerNorm', 'integer mean and std'], ['Add', 'residual connection']], hot = Math.floor(t * 2.4) % ops.length;
    put(b, 0, 0, '// tensor program: at most 240 ops, each provable', 1);
    ops.forEach(([n, d], i) => { put(b, 0, 2 + i, i === hot ? '>' : ' ', 5); put(b, 2, 2 + i, n, i === hot ? 5 : 2); put(b, 16, 2 + i, d, i === hot ? 5 : 6); });
    put(b, 0, 11, 'DeiT-Tiny       photos        224 x 224', 3);
    put(b, 0, 12, 'BGE-base        110M params   text', 3);
    put(b, 0, 13, 'TinyStories     writes stories', 3);
    put(b, 0, 15, 'every op: its own fraud proof, exact integers', 1);
    return b;
  }

  const scenes = [[scenePearl, 4.4], [sceneTile, 6.4], [sceneCode, 6.4], [sceneMerkle, 6.0], [sceneFraud, 6.0], [sceneChain, 6.4], [sceneProgram, 6.4]];
  const TR = 1.2;
  let idx = 0, phase0 = 0, morphing = false, delays = new Float32Array(SW * SH);

  function size() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const fs = fsOf();
    cx.font = `500 ${fs}px ${MONO}`;
    cw = cx.measureText('M').width; ch = fs * 1.18;
    cols = Math.ceil(W / cw); rows = Math.ceil(H / ch);
    const r = slot.getBoundingClientRect(), hr = cv.getBoundingClientRect();
    sfs = Math.max(9, Math.min(17, r.width / (SW * 0.6)));
    cx.font = `500 ${sfs}px ${MONO}`;
    scw = cx.measureText('M').width; sch = sfs * 1.22;
    sx = r.left - hr.left + (r.width - SW * scw) / 2; sy = r.top - hr.top + (r.height - SH * sch) / 2;
  }

  function compose(t) {                                       // the characters shown now (mid-change cells scramble)
    const el = t - phase0, cur = scenes[idx][0](t);
    if (!morphing) return cur;
    const nxt = scenes[(idx + 1) % scenes.length][0](t), out = blank();
    for (let i = 0; i < SW * SH; i++) {
      const p = (el - delays[i]) / 0.45;
      if (p <= 0) { out.c[i] = cur.c[i]; out.k[i] = cur.k[i]; }
      else if (p >= 1) { out.c[i] = nxt.c[i]; out.k[i] = nxt.k[i]; }
      else if (cur.c[i] !== ' ' || nxt.c[i] !== ' ') { out.c[i] = SCR[Math.floor(el * 26 + i * 7) % SCR.length]; out.k[i] = 2 + (i % 3); }
    }
    return out;
  }

  function frame(now) {
    const t = now / 1000, aspect = ch / cw;
    if (!phase0) phase0 = t;
    if (!reduce) {
      const el = t - phase0;
      if (!morphing && el >= scenes[idx][1]) { morphing = true; phase0 = t; for (let i = 0; i < delays.length; i++) delays[i] = Math.random() * 0.5 + ((i % SW) / SW) * 0.45; }
      else if (morphing && el >= TR + 0.5) { morphing = false; idx = (idx + 1) % scenes.length; phase0 = t; }
    }
    const scene = compose(t);
    cx.clearRect(0, 0, W, H);
    // quiet background: rings and grain spreading from the scene
    cx.font = `500 ${fsOf()}px ${MONO}`;
    const ccx = (sx + SW * scw / 2) / cw, ccy = (sy + SH * sch / 2) / ch, R = Math.max(10, SW * scw * 0.5 / cw);
    const x0 = sx / cw - 1, x1 = (sx + SW * scw) / cw + 1, y0 = sy / ch - 1, y1 = (sy + SH * sch) / ch + 1;
    const lines = Array.from({ length: HUES }, () => []);
    for (let r = 0; r < rows; r++) {
      const buf = new Array(HUES);
      for (let c = 0; c < cols; c++) {
        if (c > x0 && c < x1 && r > y0 && r < y1) continue;
        const dx = (c - ccx) / R, dy = (r - ccy) * aspect / R, dist = Math.max(0.6, Math.sqrt(dx * dx + dy * dy));
        const wave = Math.sin(dist * 9 - t * 2.2) * Math.exp(-(dist - 1) * 1.1), grain = Math.sin(c * 0.31 + t * 0.7) * Math.cos(r * 0.43 - t * 0.5);
        const v = Math.max(0, wave) * 0.42 + Math.max(0, grain) * 0.07, k = Math.min(RAMP.length - 1, Math.floor(v * (RAMP.length - 1)));
        if (k <= 0) continue;
        const p = Math.min(HUES - 1, Math.floor(((dist * 0.25 + t * 0.04) % 1) * HUES));
        (buf[p] || (buf[p] = new Array(cols).fill(' ')))[c] = RAMP[k];
      }
      for (let p = 0; p < HUES; p++) if (buf[p]) lines[p].push([r, buf[p].join('')]);
    }
    for (let p = 0; p < HUES; p++) {
      const [R_, G_, B_] = palette[p];
      cx.fillStyle = `rgba(${R_},${G_},${B_},0.34)`;
      for (const [r, s] of lines[p]) cx.fillText(s, 0, (r + 1) * ch - ch * 0.2);
    }
    // the scene: code, matrices and trees made of characters
    cx.font = `500 ${sfs}px ${MONO}`;
    const buckets = Array.from({ length: SCOL.length }, () => []);
    for (let i = 0; i < SW * SH; i++) {
      const c = scene.c[i]; if (c === ' ') continue;
      const cxp = sx + (i % SW) * scw, cyp = sy + (Math.floor(i / SW) + 1) * sch - sch * 0.22;
      const near = Math.abs(cxp + scw / 2 - px) < 70 && Math.abs(cyp - sch / 2 - py) < 50;
      buckets[near ? 5 : scene.k[i]].push(c, cxp, cyp);
    }
    for (let k = 1; k < SCOL.length; k++) {
      const a = buckets[k]; if (!a.length) continue;
      cx.fillStyle = `rgba(${SCOL[k][0]},${SCOL[k][1]},${SCOL[k][2]},${k === 1 ? 0.75 : 0.95})`;
      for (let j = 0; j < a.length; j += 3) cx.fillText(a[j], a[j + 1], a[j + 2]);
    }
    if (!reduce && visible && !document.hidden) requestAnimationFrame(frame); else running = false;
  }
  const kick = () => { if (!running) { running = true; requestAnimationFrame(frame); } };
  const follow = (e) => { const r = cv.getBoundingClientRect(); px = e.clientX - r.left; py = e.clientY - r.top; };
  window.addEventListener('pointermove', follow, { passive: true });
  window.addEventListener('pointerdown', follow, { passive: true });
  window.addEventListener('resize', () => { size(); if (reduce) kick(); });
  if ('ResizeObserver' in window) new ResizeObserver(() => { size(); if (reduce) kick(); }).observe(hero0);
  if ('IntersectionObserver' in window) {
    new IntersectionObserver((e) => { visible = e[0].isIntersecting; if (visible && !reduce) kick(); }).observe(cv);
  }
  document.addEventListener('visibilitychange', () => { if (!document.hidden && !reduce) kick(); });
  if (reduce) idx = 1;
  (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(() => { size(); kick(); });

  /* ---------------------------------------------------------------- silent demo film: starts when it is on screen, pauses when it leaves, a click or tap pauses it */
  const demo = $('#demo');
  if (demo) {
    const frame = demo.parentElement, calm = reduce || (navigator.connection && navigator.connection.saveData);
    let byUser = false;
    const sync = () => frame.classList.toggle('paused', demo.paused);
    const play = () => { if (!demo.src) demo.src = demo.dataset.src; const p = demo.play(); if (p && p.catch) p.catch(() => {}); };
    demo.addEventListener('play', sync); demo.addEventListener('pause', sync);
    const toggle = () => { byUser = !demo.paused; if (demo.paused) { byUser = false; play(); } else demo.pause(); };
    demo.addEventListener('click', toggle);
    demo.addEventListener('keydown', (e) => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); toggle(); } });
    sync();
    if ('IntersectionObserver' in window) {
      new IntersectionObserver((e) => {
        if (e[0].isIntersecting) { if (!calm && !byUser) play(); } else if (!demo.paused) demo.pause();
      }, { threshold: 0.35 }).observe(demo);
    }
  }

  /* ---------------------------------------------------------------- the launch film: opens over the page when asked, with its sound; never by itself.
     Without dialog support (or without this script) the button is a plain link to the film. */
  const filmBtn = $('#film-open'), film = $('#film'), fv = $('#film-video');
  if (filmBtn && film && fv && typeof film.showModal === 'function') {
    // the film stops with its window, whatever closes it (the button, a click outside, Escape); not left to the 'close' event alone,
    // which some embedded browsers do not send
    const stop = () => { fv.pause(); filmBtn.focus(); };
    const close = () => { stop(); if (film.open) film.close(); };
    film.addEventListener('cancel', stop);
    new MutationObserver(() => { if (!film.open && !fv.paused) stop(); }).observe(film, { attributes: true, attributeFilter: ['open'] });
    filmBtn.addEventListener('click', (e) => {
      e.preventDefault();
      if (!fv.getAttribute('src')) {
        // 1080p on large screens, 720p on phones and when the visitor asked to save data
        const px = Math.max(screen.width, screen.height) * (window.devicePixelRatio || 1);
        const save = navigator.connection && navigator.connection.saveData;
        fv.src = px > 1500 && !save ? fv.dataset.hd : fv.dataset.sd;
      }
      film.showModal();
      const p = fv.play(); if (p && p.catch) p.catch(() => {});
    });
    $('#film-close').addEventListener('click', close);
    film.addEventListener('click', (e) => { if (e.target === film) close(); });   // a click outside the film
    film.addEventListener('close', () => { if (!fv.paused) stop(); });
  }

  /* ---------------------------------------------------------------- hero feed: replay of a recorded testnet run */
  // Every item below comes from runs verified on the testnet: the 32 CIFAR-10 test images classified by ResNet-20
  // (30 correct; the thumbnails are rebuilt from the model's own input tensors), the 8 sentences embedded by MiniLM-L6
  // with their actual nearest neighbour, and the 8192^3 tiled product.
  // CC0 photos classified by DeiT-Tiny on the chain (assets/photos.js, credits: /credits)
  const PHOTOS = (window.VEARL_PHOTOS || []).slice();
  const SENTENCES = [
    ['the striker scored twice as the team won the championship final on saturday', 'the coach said the players trained hard before the season opener'],
    ['shares fell sharply as oil prices rose and investors worried about inflation', 'the company reported quarterly profits above analysts expectations'],
    ['the chip maker unveiled a faster processor for computers and mobile phones', 'researchers released a new software update with improved security'],
    ['the president met foreign leaders to discuss the peace treaty', 'the government announced new sanctions after the election'],
  ];
  // stories written word by word on the chain by TinyStories-33M (greedy decoding, every word verified before payment;
  // prompt, then the first words the chain produced : site/tools/gpt-stories.json)
  const STORIES = [
    ['Once upon a time, there was a little cat named Tom.', 'Tom was a very curious cat. He liked to explore and find new things.'],
    ['Sam wanted to build a big tower with his blocks.', 'He put one block on top of another, and then another, and another.'],
    ['There was a friendly dragon who lived in a cave.', 'He was very kind and he liked to help people.'],
    ['Anna saw a sad puppy in the rain.', 'She asked her mom, “Why is the puppy so sad?”'],
    ['One day, a small bird found a shiny key.', 'The bird was very happy and wanted to show it to his friends.'],
  ];
  const replay = [];
  for (let i = 0; i < PHOTOS.length; i++) {
    if (i % 4 === 1) replay.push({ kind: 'gen', model: 'TinyStories-33M', io: STORIES[(i / 4 | 0) % STORIES.length] });
    replay.push({ kind: 'img', model: 'DeiT-Tiny · 224 × 224', photo: PHOTOS[i], io: ['photo → ', PHOTOS[i].label] });
    if (i % 4 === 3) replay.push({ kind: 'txt', model: 'BGE-base · 110M', io: SENTENCES[(i / 4 | 0) % SENTENCES.length] });
    if (i === 10) replay.push({ kind: 'mat', model: 'Tiled GEMM', io: ['8192 × 8192 × 8192 → ', '1.1 trillion ops'] });
  }
  const feed = $('#feed');
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text) e.textContent = text; return e; };
  function jobNode(j, settled) {
    const li = el('li', 'job');
    const ico = el('div', 'job-ico');
    if (j.kind === 'img') {
      ico.classList.add('photo');
      const im = document.createElement('img');
      im.src = j.photo.src;
      im.alt = `Photo classified as ${j.photo.label}`;
      im.width = 56; im.height = 56;
      im.decoding = 'async';
      ico.append(im);
    } else {
      li.classList.add('plain');
    }
    const main = el('div', 'job-main');
    const top = el('div', 'job-top');
    top.append(el('span', 'job-model', j.model));
    const state = el('span', 'job-state ' + (settled ? 'ok' : 'run'), settled ? '✓ verified · paid' : 'running on GPU');
    top.append(state);
    const io = el('div', 'job-io');
    if (j.kind === 'gen') {
      const nn = el('span', 'nn', '→ ');
      nn.append(el('b', '', j.io[1]));
      io.append(el('span', 'q', j.io[0]), nn);
    } else if (j.kind === 'txt') {
      const nn = el('span', 'nn', '→ closest: ');
      nn.append(el('b', '', `“${j.io[1]}”`));
      io.append(el('span', 'q', `“${j.io[0]}”`), nn);
    } else {
      const line = el('span');
      line.append(document.createTextNode(j.io[0]), el('b', '', j.io[1]));
      io.append(line);
    }
    const bar = el('div', 'job-bar'); bar.append(el('i'));
    main.append(top, io, bar);
    if (j.kind === 'img') li.append(ico, main); else li.append(main);
    if (settled) li.classList.add('done');
    return { li, state };
  }
  let jobIndex = 0;
  function pushJob(settled) {
    const j = replay[jobIndex++ % replay.length];
    const { li, state } = jobNode(j, settled);
    feed.prepend(li);
    while (feed.children.length > 5) feed.lastChild.remove();
    if (settled) return;
    li.classList.add('enter');
    requestAnimationFrame(() => requestAnimationFrame(() => { li.classList.remove('enter'); li.classList.add('done'); }));
    setTimeout(() => { state.className = 'job-state chk'; state.textContent = 'Freivalds check'; }, 1250);
    setTimeout(() => { state.className = 'job-state ok'; state.textContent = '✓ verified · paid'; }, 2050);
  }
  for (let i = 0; i < 4; i++) pushJob(true);
  if (!reduce) {
    let feedOn = true;
    if ('IntersectionObserver' in window) new IntersectionObserver((e) => { feedOn = e[0].isIntersecting; }).observe(feed);
    setInterval(() => { if (feedOn && !document.hidden) pushJob(false); }, 2800);
  }

  /* ---------------------------------------------------------------- Freivalds console */
  const pre = $('#gemm'), log = $('#vlog');
  const N = 6, rnd = () => Math.floor(Math.random() * 15) - 7, pad = (x, w) => String(x).padStart(w, ' ');
  let A, B, C, step = 0, phase = 0, jobs = 0, bad = -1, cheat = false;
  function newJob() {
    A = Array.from({ length: N }, () => Array.from({ length: N }, rnd));
    B = Array.from({ length: N }, () => Array.from({ length: N }, rnd));
    C = A.map((r) => B[0].map((_, j) => r.reduce((s, a, k) => s + a * B[k][j], 0)));
    cheat = jobs % 3 === 2; bad = -1;
    if (cheat) { const i = Math.floor(Math.random() * N), j = Math.floor(Math.random() * N); C[i][j] += 1; bad = i * N + j; }
    jobs++;
  }
  function draw(hr, hc) {
    const out = ['<span class="dim">  A · weights            B · input              C = A × B · from miner</span>'];
    for (let i = 0; i < N; i++) {
      const a = A[i].map((x) => pad(x, 3)).join(''), b = B[i].map((x) => pad(x, 3)).join('');
      const c = C[i].map((x, j) => {
        const s = pad(x, 4);
        if (bad === i * N + j && phase >= 3) return `<span class="bad">${s}</span>`;
        return (i === hr || j === hc) ? `<span class="hl">${s}</span>` : s;
      }).join('');
      out.push(`${a}   ${i === 2 ? '×' : ' '}  ${b}   ${i === 2 ? '=' : ' '} ${c}`);
    }
    pre.innerHTML = out.join('\n');
  }
  const hex = () => Array.from({ length: 8 }, () => '0123456789abcdef'[Math.floor(Math.random() * 16)]).join('');
  function tick() {
    if (phase === 0) {
      newJob(); step = 0; phase = 1;
      log.innerHTML = `<span class="dim">job 0x${hex().slice(0, 6)} · ${N}×${N}×${N} · escrow locked</span>\n`;
    }
    if (phase === 1) {
      draw(step, -1);
      if (++step >= N) { phase = 2; step = 0; log.innerHTML += `miner  : result committed, root <span class="hl">${hex()}</span>\n`; }
      return;
    }
    if (phase === 2) {
      draw(-1, step % N);
      if (++step < 3) return;
      phase = 3;
      const r = Array.from({ length: N }, () => (Math.random() < 0.5 ? 1 : 0));
      if (!r.some(Boolean)) r[0] = 1;
      const Br = B.map((row) => row.reduce((s, x, j) => s + x * r[j], 0));
      const ABr = A.map((row) => row.reduce((s, x, k) => s + x * Br[k], 0));
      const Cr = C.map((row) => row.reduce((s, x, j) => s + x * r[j], 0));
      const ok = ABr.every((v, i) => v === Cr[i]);
      log.innerHTML += `client : r = [${r.join(' ')}]\n         A(Br) = [${ABr.join(' ')}]\n         Cr    = [${Cr.join(' ')}]\n`;
      if (cheat && ok) {                      // this random vector missed the altered cell: draw another one
        log.innerHTML += '<span class="dim">         vector missed the altered cell, drawing another</span>\n';
        phase = 2; step = 0; return;
      }
      log.innerHTML += ok ? '<span class="ok">✓ match · escrow released to the miner</span>'
                          : '<span class="bad">✗ mismatch · fraud proof published, miner bond slashed</span>';
      draw(-1, -1);
      return;
    }
    if (++phase > 10) phase = 0;
  }
  newJob(); jobs = 0; draw(-1, -1);
  if (reduce) { phase = 0; tick(); }
  else if ('IntersectionObserver' in window) {
    let on = false;
    new IntersectionObserver((e) => { on = e[0].isIntersecting; }).observe(pre);
    setInterval(() => { if (on && !document.hidden) tick(); }, 420);
  }

  // steps light up as they cross the middle of the screen
  if ('IntersectionObserver' in window) {
    const so = new IntersectionObserver((es) => es.forEach((e) => e.target.classList.toggle('on', e.isIntersecting)), { rootMargin: '-35% 0px -45% 0px' });
    document.querySelectorAll('#steps li').forEach((li) => so.observe(li));
  }

  /* ---------------------------------------------------------------- closing ripples */
  const dc = $('#drops'), dx = dc.getContext('2d');
  let drops = [], DW = 0, DH = 0, dOn = false, dRun = false;
  function dsize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    DW = dc.clientWidth; DH = dc.clientHeight; dc.width = DW * dpr; dc.height = DH * dpr; dx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function dframe() {
    dx.clearRect(0, 0, DW, DH);
    if (Math.random() < 0.03 && drops.length < 7) drops.push({ x: Math.random() * DW, y: Math.random() * DH, r: 0, h: Math.random() });
    drops = drops.filter((d) => d.r < 260);
    for (const d of drops) {
      d.r += 0.9;
      const a = Math.max(0, 1 - d.r / 260);
      for (let k = 0; k < 3; k++) {
        const rr = d.r - k * 22; if (rr <= 0) continue;
        const c = palette[Math.floor(((d.h + k * 0.15) % 1) * HUES)];
        dx.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},${a * (0.5 - k * 0.14)})`;
        dx.lineWidth = 1; dx.beginPath(); dx.arc(d.x, d.y, rr, 0, Math.PI * 2); dx.stroke();
      }
    }
    if (dOn) requestAnimationFrame(dframe); else dRun = false;
  }
  window.addEventListener('resize', dsize);
  dsize();
  if (!reduce && 'IntersectionObserver' in window) {
    new IntersectionObserver((e) => { dOn = e[0].isIntersecting; if (dOn && !dRun) { dRun = true; requestAnimationFrame(dframe); } }).observe(dc);
  }
})();
