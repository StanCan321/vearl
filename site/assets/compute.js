/* Vearl Compute: create a job, follow it, get a verified result. A single-page reader/writer of the job service (/gw/*).
   No libraries, no third-party requests, no innerHTML: every node is built with textContent, which is what the strict CSP expects. */
(function () {
  'use strict';

  function meta(n) { return (document.querySelector('meta[name="' + n + '"]') || {}).content || ''; }
  var GW = meta('vearl-gw') || '/gw';
  var EXPLORER = meta('vearl-explorer') || '/explorer/';
  var app = document.getElementById('app');

  /* ------------------------------------------------------------------ small DOM helpers */
  function h(tag, props) {
    var el = document.createElement(tag);
    if (props) {
      Object.keys(props).forEach(function (k) {
        var v = props[k];
        if (v == null || v === false) return;
        if (k === 'class') el.className = v;
        else if (k === 'text') el.textContent = v;
        else if (k.slice(0, 2) === 'on') el.addEventListener(k.slice(2), v);
        else el.setAttribute(k, v === true ? '' : v);
      });
    }
    for (var i = 2; i < arguments.length; i++) append(el, arguments[i]);
    return el;
  }
  function append(el, c) {
    if (c == null || c === false) return;
    if (Array.isArray(c)) { c.forEach(function (x) { append(el, x); }); return; }
    el.appendChild(c.nodeType ? c : document.createTextNode(String(c)));
  }
  var SVGNS = 'http://www.w3.org/2000/svg';
  function sv(tag, props) {
    var el = document.createElementNS(SVGNS, tag);
    if (props) Object.keys(props).forEach(function (k) { if (props[k] != null) el.setAttribute(k, props[k]); });
    for (var i = 2; i < arguments.length; i++) append(el, arguments[i]);
    return el;
  }
  document.querySelectorAll('[data-ext="explorer"]').forEach(function (a) { a.setAttribute('href', EXPLORER); });

  /* ------------------------------------------------------------------ service */
  function api(path, opts) {
    opts = opts || {};
    var ctl = new AbortController();
    var timer = setTimeout(function () { ctl.abort(); }, opts.timeout || 12000);
    var init = { signal: ctl.signal, headers: { Accept: 'application/json' } };
    if (opts.body) { init.method = 'POST'; init.body = JSON.stringify(opts.body); init.headers['Content-Type'] = 'application/json'; }
    return fetch(GW + path, init)
      .then(function (r) {
        return r.json().catch(function () { return null; }).then(function (body) {
          if (!r.ok) { var e = new Error((body && body.error) || ('HTTP ' + r.status)); e.status = r.status; throw e; }
          return body;
        });
      })
      .catch(function (e) {
        if (e && e.name === 'AbortError') e = new Error('The job service did not answer in time.');
        if (e && e.status == null) e.status = 0;
        throw e;
      })
      .then(function (v) { clearTimeout(timer); return v; }, function (e) { clearTimeout(timer); throw e; });
  }

  /* ------------------------------------------------------------------ state, storage */
  var S = { info: null, online: null, refresh: null, sig: null, form: null };
  var STORE = 'vearl.compute.jobs';
  function mine() { try { var v = JSON.parse(localStorage.getItem(STORE) || '[]'); return Array.isArray(v) ? v.filter(function (x) { return /^[0-9a-f]{12}$/.test(x); }) : []; } catch (e) { return []; } }
  function remember(ref) { try { var l = mine().filter(function (x) { return x !== ref; }); l.unshift(ref); localStorage.setItem(STORE, JSON.stringify(l.slice(0, 40))); } catch (e) { /* private mode: the job page link still works */ } }

  /* ------------------------------------------------------------------ formatting */
  var nf = new Intl.NumberFormat('en-US');
  function num(n) { return nf.format(n); }
  function short(x, a, b) { x = String(x || ''); a = a || 10; b = b || 8; return x.length > a + b + 1 ? x.slice(0, a) + '…' + x.slice(-b) : x; }
  function utc(ts) { return new Date(ts * 1000).toISOString().replace('T', ' ').slice(0, 19) + ' UTC'; }
  function ago(ts) {
    var d = Math.max(0, Math.round(Date.now() / 1000 - ts));
    if (d < 10) return 'just now';
    if (d < 60) return d + ' s ago';
    if (d < 3600) return Math.floor(d / 60) + ' min ago';
    if (d < 86400) return Math.floor(d / 3600) + ' h ago';
    return Math.floor(d / 86400) + ' d ago';
  }
  function clock(s) { s = Math.max(0, Math.round(s)); var m = Math.floor(s / 60); return m ? m + ' min ' + String(s % 60).padStart(2, '0') + ' s' : s + ' s'; }
  function cap(s) { s = String(s || ''); return s.charAt(0).toUpperCase() + s.slice(1); }
  function a(href, text, cls, title) { return h('a', { href: href, class: cls || 'lnk', title: title || null }, text); }

  var KIND_ORDER = ['render3', 'render2', 'render', 'pi', 'digits', 'matmul', 'gpt', 'vit', 'embed'];
  function isPath(k) { return k === 'render3' || k === 'render2' || k === 'render'; }   /* the three renderers share one engine (path tracing on the GPU) */
  var R3_PASS_SPP = 32; /* samples per pixel in one pass: the quality is a multiple of it */
  var R3_RATE = { watch: 2.0e6, cocktail: 0.95e6, mini: 2.0e6 }; /* samples per second measured on the project's RTX 4070 SUPER */
  var R3_QUALITY_NAME = { 128: 'Good', 256: 'Fine', 512: 'Very fine', 1024: 'Studio' };
  function r3h(P) { return P.shape === 'wide' ? Math.floor(P.size * 9 / 16 / 16) * 16 : P.size; }
  function r3secs(P) { return Math.max(1, Math.round(P.size * r3h(P) * P.quality / (R3_RATE[P.scene] || 1e6))); }
  function minsec(s) { return s < 90 ? s + ' s' : Math.round(s / 60 * 10) / 10 + ' min'; }
  function sceneLabel(k, s) { var K = S.info && S.info.kinds[k], L = K && K.params.scene && K.params.scene.labels; return (L && L[s]) || cap(s); }
  function isMl(kind) { return kind === 'gpt' || kind === 'vit' || kind === 'embed'; }
  function isExact(kind) { return kind === 'digits' || kind === 'matmul' || isMl(kind); }
  var EXAMPLE_PROMPTS = ['Once upon a time, there was a little cat', 'One day, a girl named Lily found a shiny red ball', 'The old dragon was very sad because'];
  var EXAMPLE_TEXT_SETS = [
    ['The striker scored twice in the final.', 'Her goals won the championship for the team.', 'The new processor is faster and uses less power.'],
    ['Shares fell sharply as oil prices rose.', 'Investors worried about inflation and interest rates.', 'The goalkeeper saved a penalty in the shootout.'],
    ['How do I reset my password?', 'I forgot my login and cannot sign in.', 'What is the capital of France?']
  ];
  function textLines(v) { return (Array.isArray(v) ? v : String(v || '').split('\n')).map(function (t) { return String(t).replace(/\s+/g, ' ').trim(); }).filter(function (t) { return t; }); }
  var PHOTO_SIDE = 224;
  function photoToDataUrl(img) {
    /* centre-crop to a square and shrink to 224 × 224, in the browser: only a small JPEG leaves the page */
    var c = document.createElement('canvas'), s = Math.min(img.naturalWidth || img.width, img.naturalHeight || img.height);
    c.width = c.height = PHOTO_SIDE;
    var g = c.getContext('2d');
    g.drawImage(img, ((img.naturalWidth || img.width) - s) / 2, ((img.naturalHeight || img.height) - s) / 2, s, s, 0, 0, PHOTO_SIDE, PHOTO_SIDE);
    return c.toDataURL('image/jpeg', 0.9);
  }
  /* before the testnet starts, /info says prelaunch: the page is complete, only the creation of a job is closed */
  var MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  function launchLong(iso) { var d = new Date(iso); return isNaN(d) ? 'the launch' : d.getUTCDate() + ' ' + MONTHS[d.getUTCMonth()] + ' ' + d.getUTCFullYear() + ' at ' + String(d.getUTCHours()).padStart(2, '0') + ':' + String(d.getUTCMinutes()).padStart(2, '0') + ' UTC'; }
  function launchShort(iso) { var d = new Date(iso); return isNaN(d) ? 'launch' : d.getUTCDate() + ' ' + MONTHS[d.getUTCMonth()].slice(0, 3); }
  function kindTitle(k) { var K = S.info && S.info.kinds[k]; return K ? K.title : cap(k); }
  var STATUS_TAG = { queued: 't-native', submitting: 't-native', open: 't-job', claimed: 't-layer', posted: 't-layer', verifying: 't-op', running: 't-layer', done: 't-ok', rejected: 't-bad', failed: 't-bad' };
  var STATUS_TEXT = { queued: 'queued', submitting: 'sending', open: 'waiting for a miner', claimed: 'miner computing', posted: 'result posted', verifying: 'auditing', running: 'miner computing, checked exactly', done: 'done', rejected: 'rejected', failed: 'failed' };
  function statusTag(s) { return h('span', { class: 'tag ' + (STATUS_TAG[s] || 't-native'), text: STATUS_TEXT[s] || s }); }
  function isFinal(s) { return s === 'done' || s === 'rejected' || s === 'failed'; }
  function summary(j) {
    var p = j.params || {};
    if (j.kind === 'gpt') return p.prompt ? '“' + short(p.prompt, 34, 0).replace('…', '') + (p.prompt.length > 34 ? '…' : '') + '” · ' + p.words + ' words' : 'story';
    if (j.kind === 'vit') return 'one photo · 224 × 224 · 1,000 classes';
    if (j.kind === 'embed') return (p.texts ? p.texts.length : '?') + ' text' + (p.texts && p.texts.length > 1 ? 's' : '') + ' · 1,024 dimensions · 10 chained jobs';
    if (j.kind === 'digits') return '2 handwritten digits · sample ' + (p.sample + 1) + ' of 12';
    if (j.kind === 'matmul') return p.size + ' × ' + p.size + ' matrices · ' + num(j.chunks) + ' block products';
    if (j.kind === 'pi') return num(p.samples * p.chunks) + ' random points · ' + p.chunks + ' chunks';
    if (isPath(j.kind)) return sceneLabel('render3', p.scene) + ' · ' + p.size + ' × ' + (p.height || p.size) + ' px · ' + p.quality + ' samples per pixel · ' + num(j.chunks) + ' chunks';
    return p.size + ' × ' + p.size + ' px · ' + p.spp + ' sample' + (p.spp > 1 ? 's' : '') + ' per pixel · ' + j.chunks + ' tiles';
  }

  /* ------------------------------------------------------------------ components */
  function panel(title, body, more, flush) {
    return h('section', { class: 'panel' + (flush ? ' flush' : '') },
      h('div', { class: 'panel-head' }, h('h2', { text: title }), more ? a(more.href, more.text, 'more') : null),
      h('div', { class: 'panel-body' }, body));
  }
  function kv(rows) {
    var el = h('div', { class: 'kv' });
    rows.forEach(function (r) {
      if (!r || r[1] == null) return;
      el.appendChild(h('div', { class: 'k', text: r[0] }));
      el.appendChild(h('div', { class: 'v' + (r[2] ? ' ' + r[2] : '') }, r[1]));
    });
    return el;
  }
  function crumbs(parts) {
    var el = h('div', { class: 'crumbs' });
    parts.forEach(function (p, i) { if (i) el.appendChild(h('i', { text: '/' })); el.appendChild(p.href ? a(p.href, p.text, '') : h('span', { text: p.text })); });
    return el;
  }
  function head(parts, title, lede) {
    return h('div', { class: 'x-page-head' }, h('div', { class: 'wrap' }, crumbs(parts), h('h1', null, title), lede ? h('p', { text: lede }) : null));
  }
  function stateCard(title, text, bad, retry) {
    return h('div', { class: 'wrap' }, h('div', { class: 'state' + (bad ? ' bad' : '') }, h('h2', null, title), h('p', { text: text }),
      retry ? h('button', { class: 'btn', type: 'button', onclick: function () { route(); } }, 'Try again') : null));
  }
  function loading() { return h('div', { class: 'wrap loading' }, h('span', { class: 'skel' }), h('span', { class: 'skel' }), h('span', { class: 'skel' })); }
  function copyBtn(text) {
    var b = h('button', { class: 'copy', type: 'button', 'aria-label': 'Copy to clipboard', title: 'Copy', text: '⧉' });
    b.addEventListener('click', function () {
      function done() { b.classList.add('done'); setTimeout(function () { b.classList.remove('done'); }, 1200); }
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, function () {});
    });
    return b;
  }

  /* ------------------------------------------------------------------ illustrations (SVG, attributes only) */
  function tilesSvg(side) {
    var n = side / 16, W = 240, c = W / n;
    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + W, role: 'img', 'aria-label': n * n + ' tiles of 16 by 16 pixels' }, sv('rect', { x: 0, y: 0, width: W, height: W, class: 'pv-bg' }));
    for (var y = 0; y < n; y++) for (var x = 0; x < n; x++) svg.appendChild(sv('rect', { x: (x * c + .4).toFixed(2), y: (y * c + .4).toFixed(2), width: (c - .8).toFixed(2), height: (c - .8).toFixed(2), class: 'pv-tile' }));
    return svg;
  }
  function digitSvg(px, label) {
    var svg = sv('svg', { viewBox: '0 0 80 80', role: 'img', 'aria-label': label || 'handwritten digit' }, sv('rect', { x: 0, y: 0, width: 80, height: 80, class: 'pv-bg' }));
    px.forEach(function (v, i) {
      if (!v) return;
      var g = Math.round(46 + v / 16 * 209);
      svg.appendChild(sv('rect', { x: (i % 8) * 10, y: Math.floor(i / 8) * 10, width: 10, height: 10, fill: 'rgb(' + g + ',' + g + ',' + g + ')' }));
    });
    return svg;
  }
  function blocksSvg(b) {
    var W = 240, c = W / b, svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + W, role: 'img', 'aria-label': b * b + ' blocks of 128 by 128 in the product' }, sv('rect', { x: 0, y: 0, width: W, height: W, class: 'pv-bg' }));
    for (var y = 0; y < b; y++) for (var x = 0; x < b; x++) {
      svg.appendChild(sv('rect', { x: (x * c + 1).toFixed(1), y: (y * c + 1).toFixed(1), width: (c - 2).toFixed(1), height: (c - 2).toFixed(1), class: 'pv-tile' }));
      svg.appendChild(sv('text', { x: (x * c + c / 2).toFixed(1), y: (y * c + c / 2 + 4).toFixed(1), 'text-anchor': 'middle', class: 'ax-t' }, b + ' products'));
    }
    return svg;
  }
  function logitBars(logits, predicted, label) {
    var W = 200, H = 54, lo = Math.min.apply(null, logits), hi = Math.max.apply(null, logits), span = Math.max(1, hi - lo), bw = W / 10;
    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + (H + 12), role: 'img', 'aria-label': 'Score of each digit from 0 to 9' });
    logits.forEach(function (v, i) {
      var bh = 3 + (H - 3) * (v - lo) / span;
      svg.appendChild(sv('rect', { x: (i * bw + 2).toFixed(1), y: (H - bh).toFixed(1), width: (bw - 4).toFixed(1), height: bh.toFixed(1), class: i === predicted ? 'bar-a' : 'bar-b dim-bar' }));
      svg.appendChild(sv('text', { x: (i * bw + bw / 2).toFixed(1), y: H + 10, 'text-anchor': 'middle', class: i === label ? 'ax-l tick-true' : 'ax-l' }, String(i)));
    });
    return svg;
  }
  function piSvg(points) {
    var W = 240, svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + W, role: 'img', 'aria-label': 'Random points in a square, some inside the quarter circle' }, sv('rect', { x: 0, y: 0, width: W, height: W, class: 'pv-bg' }));
    var seed = 12345, rnd = function () { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
    var n = Math.max(30, Math.min(700, Math.round(points / 40)));
    for (var i = 0; i < n; i++) {
      var x = rnd(), y = rnd(), inside = x * x + y * y <= 1;
      svg.appendChild(sv('circle', { cx: (x * W).toFixed(1), cy: ((1 - y) * W).toFixed(1), r: 1.6, class: inside ? 'pv-in' : 'pv-out' }));
    }
    svg.appendChild(sv('path', { d: 'M0 0 A' + W + ' ' + W + ' 0 0 1 ' + W + ' ' + W, class: 'pv-arc', transform: 'translate(0 ' + W + ') scale(1 -1)' }));
    return svg;
  }

  /* ------------------------------------------------------------------ home: the job form */
  function defaults() {
    var f = { kind: 'render3', seed: String(Math.floor(Math.random() * 1e9)) };
    Object.keys(S.info.kinds).forEach(function (k) {
      var P = S.info.kinds[k].params;
      Object.keys(P).forEach(function (p) { if (f[p] == null) f[p] = P[p].default; });
    });
    return f;
  }
  function pageHome() {
    var info = S.info, K = info.kinds, F = S.form || (S.form = defaults());
    if (!K[F.kind]) F.kind = KIND_ORDER[0];
    var root = h('div', null);
    document.title = 'Vearl Compute · Create a verified compute job, free on the testnet';
    root.appendChild(h('section', { class: 'x-hero cp-hero' }, h('div', { class: 'wrap' },
      h('div', { class: 'eyebrow' }, 'Vearl compute'),
      h('h1', null, 'Create a job. ', h('span', { class: 'serif' }, 'A miner computes it, the network checks it.')),
      h('p', { class: 'lede', text: 'Pick a computation, press a button, and get back a result that was audited before anyone was paid. Nothing to install, no wallet, no cost while Vearl is a testnet.' }),
      h('div', { class: 'free-badge' }, h('i'), 'Free on the testnet'),
      h('div', { class: 'steps3' },
        h('div', null, h('b', { text: '1 · You choose' }), h('span', { text: 'A 3D scene, π, handwritten digits or a matrix product, and how big.' })),
        h('div', null, h('b', { text: '2 · A miner computes' }), h('span', { text: 'A worker node takes the job and computes it on its GPU.' })),
        h('div', null, h('b', { text: '3 · It is audited' }), h('span', { text: 'Checked exactly, or by recomputing random pieces. A wrong one is provable and costs the miner its bond.' }))))));

    var wrap = h('div', { class: 'wrap' });
    root.appendChild(wrap);
    var err = h('div', { class: 'formerr', role: 'alert' });
    var paramsBox = h('div', { class: 'form' });
    var previewBox = h('div', null);
    var estBox = h('div', null);
    var PRE = !!info.prelaunch;
    var submit = h('button', { class: 'btn primary', type: 'submit' }, PRE ? 'Opens at the testnet launch' : 'Create the job, free');
    if (PRE) submit.disabled = true;

    function tilesOf(side) { return (side / 16) * (side / 16); }
    function chunksOf() { return F.kind === 'pi' ? F.chunks : F.kind === 'digits' ? 1 : F.kind === 'matmul' ? F.blocks * F.blocks * F.blocks : isPath(F.kind) ? (F.size / 16) * (r3h(F) / 16) * (F.quality / R3_PASS_SPP) : (F.size / 16) * (F.size / 16); }
    function drawPreviewExact() {
      var rows;
      if (F.kind === 'gpt') {
        previewBox.replaceChildren(h('div', { class: 'preview-box' }, h('div', { class: 'story-prev' }, h('span', { class: 'sp-prompt', text: F.prompt || '…' }), h('span', { class: 'sp-new', text: ' … ' + F.words + ' more word' + (F.words > 1 ? 's' : '') + ', chosen one at a time' })),
          h('p', { class: 'cap', text: 'The model continues your sentence. For every word, two jobs run on the network and are checked before the miner is paid.' })));
        rows = [['Model', 'TinyStories-33M, 4 layers'], ['Words to write', String(F.words)], ['Verified jobs', num(2 * F.words)], ['Checked', 'every operation, exactly'], ['Escrow locked by the service', 'testnet tokens'], ['Cost to you', 'Free'], ['Typical wait', 'about ' + Math.max(1, Math.round(F.words * 25 / 60)) + ' min']];
      } else if (F.kind === 'embed') {
        var tl = textLines(F.texts);
        previewBox.replaceChildren(h('div', { class: 'preview-box' }, tl.length ? h('ol', { class: 'emb-prev' }, tl.map(function (t) { return h('li', { text: t }); })) : h('p', { class: 'mute', text: 'Write one to four sentences on the left.' }),
          h('p', { class: 'cap', text: 'Each sentence becomes a list of 1,024 numbers. You will get the table of similarities between your sentences: the closer two meanings, the closer to 1.' })));
        rows = [['Model', 'Qwen3-Embedding-0.6B, 28 layers'], ['Texts', String(tl.length)], ['Verified jobs', '10, chained (all texts together)'], ['Operations checked', '≈ 2,200, exactly'], ['Escrow locked by the service', 'testnet tokens'], ['Cost to you', 'Free'], ['Typical wait', 'a few minutes']];
      } else if (F.kind === 'vit') {
        previewBox.replaceChildren(h('div', { class: 'preview-box' }, F.photo ? h('img', { class: 'photo-prev', src: F.photo.url, alt: 'The photo you picked, cropped to a square' }) : h('p', { class: 'mute', text: 'Pick a photo on the left.' }),
          h('p', { class: 'cap', text: 'Your photo is cropped to a square and shrunk to 224 × 224 in your browser. Only that small picture is sent, and it is deleted as soon as it has been read.' })));
        rows = [['Model', 'DeiT-Tiny, 12 layers'], ['Classes', '1,000 (ImageNet)'], ['Verified jobs', '2, chained'], ['Checked', 'every operation, exactly'], ['Escrow locked by the service', 'testnet tokens'], ['Cost to you', 'Free'], ['Typical wait', 'about a minute']];
      } else if (F.kind === 'digits') {
        var smp = S.digitSamples && S.digitSamples[F.sample];
        previewBox.replaceChildren(h('div', { class: 'preview-box' }, h('div', { class: 'digit-pair' }, smp ? smp.images.map(function (im) { return digitSvg(im, 'a handwritten digit'); }) : h('p', { class: 'mute', text: 'Loading the digits…' })),
          h('p', { class: 'cap', text: 'The two handwritten digits (8 × 8 pixels) the network will read. It is not told what they are.' })));
        rows = [['Network', 'small CNN, 8 operations'], ['Digits read', '2'], ['Checked', 'every operation, exactly'], ['Escrow locked by the service', '3 VRL (testnet tokens)'], ['Cost to you', 'Free'], ['Typical wait', 'under a minute']];
      } else {
        var side = F.blocks * 128, prods = F.blocks * F.blocks * F.blocks;
        previewBox.replaceChildren(h('div', { class: 'preview-box' }, blocksSvg(F.blocks), h('p', { class: 'cap', text: 'The product of two ' + side + ' × ' + side + ' matrices is cut in ' + prods + ' products of 128 × 128 blocks. Miners compute them, the client checks each one.' })));
        rows = [['Matrices', side + ' × ' + side + ' (int8)'], ['Block products', num(prods)], ['Multiply-adds', num(2 * side * side * side)], ['Checked', 'every block, exactly'], ['Escrow locked by the service', prods + ' VRL (testnet tokens)'], ['Cost to you', 'Free'], ['Typical wait', prods > 30 ? 'about a minute' : 'under a minute']];
      }
      estBox.replaceChildren(h('div', { class: 'est' }, rows.map(function (r) { return [h('div', { text: r[0] }), h('div', { text: r[1] })]; })));
    }
    function drawPreview() {
      if (isExact(F.kind)) { submit.disabled = PRE; if (err.textContent.indexOf('too large') >= 0) err.classList.remove('show'); return drawPreviewExact(); }
      var K1 = K[F.kind], n = chunksOf(), audit = Math.min(info.audit_chunks, n);
      previewBox.replaceChildren(h('div', { class: 'preview-box' }, F.kind === 'pi' ? piSvg(F.samples * F.chunks) : tilesSvg(Math.min(F.size, 256)),
        h('p', { class: 'cap', text: F.kind === 'pi' ? 'Illustration: random points, mint inside the quarter circle, rose outside. Your job throws ' + num(F.samples * F.chunks) + ' of them.' : isPath(F.kind) ? 'Illustration: your image is cut in ' + num((F.size / 16) * (r3h(F) / 16)) + ' tiles of 16 × 16 pixels and each tile is rendered ' + (F.quality / R3_PASS_SPP) + ' time' + (F.quality > R3_PASS_SPP ? 's' : '') + ', 32 samples per pixel at a time. The passes are averaged: more of them, less noise.' : 'Illustration: your image is cut in ' + n + ' tiles of 16 × 16 pixels. Each tile is one chunk a miner computes on its own.' })));
      var rows = [
        isPath(F.kind) ? ['Scene', sceneLabel('render3', F.scene)] : null,
        [F.kind === 'pi' ? 'Random points' : 'Image', F.kind === 'pi' ? num(F.samples * F.chunks) : F.size + ' × ' + (isPath(F.kind) ? r3h(F) : F.size) + ' px, ' + (isPath(F.kind) ? F.quality : F.spp) + ' spp'],
        isPath(F.kind) ? ['Rays traced', num(F.size * r3h(F) * F.quality) + ' paths, up to 14 bounces each'] : null,
        isPath(F.kind) ? ['GPU compute', 'about ' + minsec(r3secs(F))] : null,
        ['Chunks to compute', num(n)],
        ['Audited by sampling', audit + ' of ' + n + ' chunks (' + Math.round(100 * audit / n) + '%)'],
        ['Escrow locked by the service', info.escrow + ' VRL (testnet tokens)'],
        ['Cost to you', 'Free'],
        ['Typical wait', isPath(F.kind) ? 'about ' + minsec(r3secs(F) + 90 + Math.round(n / 60)) + ' in all' : 'a few minutes']
      ].filter(Boolean);
      estBox.replaceChildren(h('div', { class: 'est' }, rows.map(function (r) { return [h('div', { text: r[0] }), h('div', { text: r[1] })]; })));
      var tooBig = isPath(F.kind) && F.size * r3h(F) * F.quality > K[F.kind].params.max_samples;
      submit.disabled = PRE || tooBig;
      if (tooBig) { err.textContent = 'This image is too large for this quality: pick a smaller width or fewer samples.'; err.classList.add('show'); }
      else if (err.textContent.indexOf('too large') >= 0) err.classList.remove('show');
      void K1;
    }
    function range(label, key, P, fmt) {
      var out = h('output', { for: 'f-' + key, text: fmt ? fmt(F[key]) : String(F[key]) });
      var input = h('input', { type: 'range', id: 'f-' + key, min: P.min, max: P.max, step: P.step || 1, value: F[key] });
      input.addEventListener('input', function () { F[key] = parseInt(input.value, 10); out.textContent = fmt ? fmt(F[key]) : String(F[key]); drawPreview(); });
      return h('div', { class: 'field' }, h('label', { for: 'f-' + key }, label, out), input);
    }
    function drawParams() {
      var P = K[F.kind].params, fields = [];
      if (F.kind === 'digits') {
        var grid = h('div', { class: 'digit-grid', role: 'group', 'aria-label': 'Pick a pair of handwritten digits' }, h('p', { class: 'mute', text: 'Loading the digits…' }));
        function fillGrid() {
          grid.replaceChildren.apply(grid, S.digitSamples.map(function (sm, i) {
            var b = h('button', { class: 'dpick', type: 'button', 'aria-pressed': String(F.sample === i), 'aria-label': 'Pair ' + (i + 1) }, sm.images.map(function (im) { return digitSvg(im, ''); }));
            b.addEventListener('click', function () { F.sample = i; grid.querySelectorAll('button').forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); }); drawPreview(); });
            return b;
          }));
        }
        if (S.digitSamples) fillGrid();
        else api('/samples/digits').then(function (r) { S.digitSamples = r.samples; fillGrid(); drawPreview(); }).catch(function () { grid.replaceChildren(h('p', { class: 'mute', text: 'The digits could not be loaded.' })); });
        fields.push(h('div', { class: 'field' }, h('div', { class: 'lbl' }, 'Pick a pair of digits'), grid, h('div', { class: 'hint', text: 'These are test images that were not used to train the network. Pick one and the network reads both digits.' })));
        paramsBox.replaceChildren.apply(paramsBox, fields);
        return;
      }
      if (F.kind === 'gpt') {
        var ta = h('textarea', { id: 'f-prompt', rows: '3', maxlength: '200', 'aria-label': 'Start of the story' });
        ta.value = F.prompt;
        ta.addEventListener('input', function () { F.prompt = ta.value; drawPreview(); });
        var ex = h('div', { class: 'sizes', role: 'group', 'aria-label': 'Example beginnings' }, EXAMPLE_PROMPTS.map(function (t) {
          var b = h('button', { class: 'chip', type: 'button' }, t.split(' ').slice(0, 4).join(' ') + '…');
          b.addEventListener('click', function () { ta.value = t; F.prompt = t; drawPreview(); });
          return b;
        }));
        fields.push(h('div', { class: 'field' }, h('label', { for: 'f-prompt' }, 'Start of the story'), ta, ex, h('div', { class: 'hint', text: 'English works best: the model only knows children’s stories. Up to 200 characters; a story holds 64 words at most, prompt included.' })));
        fields.push(range('Words to write', 'words', P.words, function (v) { return v + (v > 1 ? ' words' : ' word'); }));
        paramsBox.replaceChildren.apply(paramsBox, fields);
        return;
      }
      if (F.kind === 'embed') {
        var te = h('textarea', { id: 'f-texts', rows: '5', maxlength: String(4 * 141), 'aria-label': 'Sentences, one per line' });
        te.value = textLines(F.texts).join('\n');
        te.addEventListener('input', function () { F.texts = te.value.split('\n'); drawPreview(); });
        var exs = h('div', { class: 'sizes', role: 'group', 'aria-label': 'Example sets' }, EXAMPLE_TEXT_SETS.map(function (set, i) {
          var b = h('button', { class: 'chip', type: 'button' }, ['Sport and tech', 'Markets and sport', 'Questions'][i]);
          b.addEventListener('click', function () { F.texts = set.slice(); te.value = set.join('\n'); drawPreview(); });
          return b;
        }));
        fields.push(h('div', { class: 'field' }, h('label', { for: 'f-texts' }, 'Your sentences (one per line)'), te, exs, h('div', { class: 'hint', text: 'One to four sentences, 140 characters each at most. The model understands many languages. Each text is cut to 127 word pieces.' })));
        paramsBox.replaceChildren.apply(paramsBox, fields);
        return;
      }
      if (F.kind === 'vit') {
        var photos = window.VEARL_PHOTOS || [];
        var gal = h('div', { class: 'photo-grid', role: 'group', 'aria-label': 'Pick a photo' });
        var thumbs = [];
        function mark(on) { thumbs.forEach(function (t) { t.setAttribute('aria-pressed', String(t === on)); }); }
        function use(src, label, on) {
          var im = new Image();
          im.onload = function () { F.photo = { url: photoToDataUrl(im), label: label }; mark(on); drawPreview(); };
          im.onerror = function () { err.textContent = 'That photo could not be read.'; err.classList.add('show'); };
          im.src = src;
        }
        photos.forEach(function (ph, i) {
          var b = h('button', { class: 'pphoto', type: 'button', 'aria-pressed': 'false', 'aria-label': 'Photo of a ' + ph.label, title: ph.label }, h('img', { src: ph.src, alt: '', loading: 'lazy', width: '96', height: '96' }));
          b.addEventListener('click', function () { err.classList.remove('show'); use(ph.src, ph.label, b); });
          thumbs.push(b); gal.appendChild(b);
        });
        var fileIn = h('input', { type: 'file', accept: 'image/jpeg,image/png', class: 'sr', id: 'f-upload' });
        fileIn.addEventListener('change', function () {
          var f = fileIn.files && fileIn.files[0];
          if (!f) return;
          err.classList.remove('show');
          if (f.size > 12 * 1024 * 1024) { err.textContent = 'That file is too large (12 MB at most).'; err.classList.add('show'); return; }
          var rd = new FileReader();
          rd.onload = function () { use(String(rd.result), 'your photo', null); mark(null); };
          rd.readAsDataURL(f);
        });
        fields.push(h('div', { class: 'field' }, h('div', { class: 'lbl' }, 'Pick a photo'), gal,
          h('div', { class: 'row' }, h('label', { class: 'btn mini', for: 'f-upload' }, 'Use my own photo'), fileIn),
          h('div', { class: 'hint', text: 'The samples are free-licence photos (credits are on the website’s Credits page). If you use your own, it is shrunk in your browser, read once by the network and deleted. Do not send photos of people who have not agreed.' })));
        paramsBox.replaceChildren.apply(paramsBox, fields);
        return;
      }
      if (F.kind === 'matmul') {
        var bchips = h('div', { class: 'sizes', role: 'group', 'aria-label': 'Matrix size' }, P.blocks.choices.map(function (b) {
          var btn = h('button', { class: 'chip', type: 'button', 'aria-pressed': String(F.blocks === b) }, (b * 128) + ' × ' + (b * 128));
          btn.addEventListener('click', function () { F.blocks = b; bchips.querySelectorAll('button').forEach(function (x) { x.setAttribute('aria-pressed', String(x === btn)); }); drawPreview(); });
          return btn;
        }));
        fields.push(h('div', { class: 'field' }, h('div', { class: 'lbl' }, 'Matrix size'), bchips, h('div', { class: 'hint', text: 'Bigger matrices are more block products for the miners. The public network goes up to 8192 × 8192.' })));
      } else if (F.kind === 'pi') {
        fields.push(range('Points per chunk', 'samples', P.samples, num));
        fields.push(range('Chunks', 'chunks', P.chunks, num));
      } else if (isPath(F.kind)) {
        if (F.kind === 'render2') F.scene = 'orb'; else if (F.kind === 'render') F.scene = 'gallery'; else if (F.scene === 'orb' || F.scene === 'gallery') F.scene = 'watch';
        var pick = function (label, aria, choices, key, text, hint) {
          var box = h('div', { class: 'sizes', role: 'group', 'aria-label': aria }, choices.map(function (c) {
            var b = h('button', { class: 'chip', type: 'button', 'aria-pressed': String(F[key] === c) }, text(c));
            b.addEventListener('click', function () { F[key] = c; box.querySelectorAll('button').forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); }); drawPreview(); });
            return b;
          }));
          return h('div', { class: 'field' }, h('div', { class: 'lbl' }, label), box, hint ? h('div', { class: 'hint', text: hint }) : null);
        };
        if (P.scene) fields.push(pick('Scene', 'Scene', P.scene.choices, 'scene', function (c) { return P.scene.labels[c] || c; }, 'Three scenes built from solid shapes only: glass, metal, liquid and soft studio lights.'));
        if (P.size.choices.indexOf(F.size) < 0) F.size = P.size.default;
        fields.push(pick('Image shape', 'Image shape', P.shape.choices, 'shape', function (c) { return P.shape.labels[c] || c; }, 'The cocktail and the watch were framed for a wide picture; square keeps the object closer.'));
        fields.push(pick('Image width', 'Image width', P.size.choices, 'size', function (s) { return s + ' px'; }, 'Bigger images are more tiles for the miner and take longer.'));
        fields.push(pick('Quality', 'Samples per pixel', P.quality.choices, 'quality', function (q) { return R3_QUALITY_NAME[q] + ' · ' + q + ' samples'; }, 'Light is simulated by random rays: with few samples the picture is grainy, with hundreds it is smooth and the glass, the metal and the soft shadows become exact. Doubling the samples doubles the GPU time.'));
      } else {
        if (P.size.choices.indexOf(F.size) < 0) F.size = P.size.default;
        var chips = h('div', { class: 'sizes', role: 'group', 'aria-label': 'Image size' }, P.size.choices.map(function (s) {
          var b = h('button', { class: 'chip', type: 'button', 'aria-pressed': String(F.size === s) }, s + ' × ' + s);
          b.addEventListener('click', function () { F.size = s; chips.querySelectorAll('button').forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); }); drawPreview(); });
          return b;
        }));
        fields.push(h('div', { class: 'field' }, h('div', { class: 'lbl' }, 'Image size'), chips, h('div', { class: 'hint', text: 'Bigger images are more tiles for the miner and take longer.' })));
        fields.push(range('Samples per pixel', 'spp', P.spp, function (v) { return v + (v > 1 ? ' samples' : ' sample'); }));
      }
      var seedIn = h('input', { type: 'text', id: 'f-seed', inputmode: 'numeric', autocomplete: 'off', value: F.seed, 'aria-label': 'Seed' });
      seedIn.addEventListener('input', function () { F.seed = seedIn.value.replace(/[^0-9]/g, '').slice(0, 15); });
      var dice = h('button', { class: 'btn mini', type: 'button' }, 'Random');
      dice.addEventListener('click', function () { F.seed = String(Math.floor(Math.random() * 1e9)); seedIn.value = F.seed; });
      fields.push(h('div', { class: 'field' }, h('label', { for: 'f-seed' }, 'Seed'), h('div', { class: 'row' }, seedIn, dice), h('div', { class: 'hint', text: 'The same seed always gives the same result, on any machine: the computation is in integers only.' })));
      paramsBox.replaceChildren.apply(paramsBox, fields);
    }
    var kinds = h('div', { class: 'kinds', role: 'radiogroup', 'aria-label': 'Job type' }, KIND_ORDER.filter(function (k) { return K[k]; }).map(function (k) {
      var input = h('input', { type: 'radio', name: 'kind', value: k, class: 'sr' });
      if (F.kind === k) input.checked = true;
      input.addEventListener('change', function () { F.kind = k; drawParams(); drawPreview(); });
      return h('label', { class: 'kind' }, input, h('span', { class: 'kind-body' }, h('span', { class: 'k-eyebrow', text: K[k].short }), h('span', { class: 'k-title', text: K[k].title }), h('span', { class: 'k-sum', text: K[k].summary })));
    }));
    var form = h('form', { novalidate: true }, kinds, h('div', { class: 'panel', 'aria-label': 'Job settings' }, paramsBox,
      h('div', { class: 'submit' }, err, submit, h('p', { class: 'fine', text: PRE ? 'Job creation opens when the testnet starts, on ' + launchLong(info.launch) + '. Everything else on this page is already live: look around, set up a job, see what it costs and how it is checked.' : 'Free limit: ' + info.limits.per_hour + ' jobs an hour per visitor, ' + info.limits.per_visitor_active + ' at a time. You do not need an account: keep the link of your job, or find it again under “My jobs”.' }))));
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      if (PRE) return;
      err.classList.remove('show');
      if (F.kind === 'vit' && !F.photo) { err.textContent = 'Pick a photo first.'; err.classList.add('show'); return; }
      if (F.kind === 'embed') { var tt = textLines(F.texts); if (!tt.length || tt.length > 4 || tt.some(function (t) { return t.length > 140; })) { err.textContent = 'Write one to four sentences of at most 140 characters.'; err.classList.add('show'); return; } }
      if (F.kind === 'gpt' && !String(F.prompt || '').trim()) { err.textContent = 'Write the beginning of the story first.'; err.classList.add('show'); return; }
      submit.disabled = true; submit.textContent = 'Sending…';
      var body = { kind: F.kind, seed: parseInt(F.seed || '0', 10) || 0 };
      if (F.kind === 'pi') { body.samples = F.samples; body.chunks = F.chunks; }
      else if (F.kind === 'gpt') { body = { kind: 'gpt', prompt: F.prompt, words: F.words }; }
      else if (F.kind === 'vit') { body = { kind: 'vit', image: F.photo.url }; }
      else if (F.kind === 'embed') { body = { kind: 'embed', texts: textLines(F.texts) }; }
      else if (F.kind === 'digits') { body = { kind: 'digits', sample: F.sample }; }
      else if (F.kind === 'matmul') { body.blocks = F.blocks; }
      else if (isPath(F.kind)) { body.scene = F.scene; body.size = F.size; body.shape = F.shape; body.quality = F.quality; }
      else { body.size = F.size; body.spp = F.spp; }
      api('/jobs', { body: body, timeout: 30000 }).then(function (j) {
        remember(j.ref);
        if (F.kind === 'vit') { S.photos = S.photos || {}; S.photos[j.ref] = F.photo.url; }
        location.hash = '#/job/' + j.ref;
      }).catch(function (ex) {
        submit.disabled = false; submit.textContent = 'Create the job, free';
        err.textContent = ex.status === 0 ? 'The job service could not be reached. Try again in a moment.' : ex.message;
        err.classList.add('show');
      });
    });
    drawParams(); drawPreview();
    wrap.appendChild(h('div', { class: 'studio' }, form, h('div', null, h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', { text: 'What you are about to run' })), previewBox, estBox))));

    if (info.catalog && info.catalog.length) wrap.appendChild(h('div', { class: 'x-grid' }, catalogPanel(info.catalog)));
    var recent = h('div', null, h('div', { class: 'muted-note', text: 'Loading…' }));
    wrap.appendChild(h('div', { class: 'x-grid' }, panel('Latest jobs from everyone', recent, null, true)));
    api('/recent').then(function (r) { recent.replaceChildren(jobList(r.jobs, 'No job has been created yet. Yours can be the first.')); }).catch(function () { recent.replaceChildren(h('div', { class: 'muted-note', text: 'The list is unavailable right now.' })); });
    return Promise.resolve(root);
  }

  function catalogPanel(cat) {
    var cols = [
      { k: 'Workload', cls: 'strong', cell: function (c) { return h('span', null, h('b', { text: c.title }), h('span', { class: 'cat-sub', text: c.model })); } },
      { k: 'Size', cell: function (c) { return c.size; }, cls: 'm' },
      { k: 'Measured on Vearl', cell: function (c) { return c.measure; } },
      { k: 'Here', cell: function (c) { return c.available ? h('span', null, h('span', { class: 'tag t-ok', text: 'available' }), c.note ? h('span', { class: 'cat-sub', text: c.note }) : null) : h('span', null, h('span', { class: 'tag t-native', text: 'not available yet' }), h('span', { class: 'cat-sub', text: c.reason || '' })); } }
    ];
    var rows = cat.map(function (c) { return c; });
    var tbl = h('div', { class: 'xtable-scroll' }, h('table', { class: 'xtable stack cat' },
      h('thead', null, h('tr', null, cols.map(function (c) { return h('th', { text: c.k }); }))),
      h('tbody', null, rows.map(function (r) { return h('tr', { class: r.available ? null : 'cat-off' }, cols.map(function (c) { return h('td', { class: c.cls || null, 'data-k': c.k }, c.cell(r)); })); }))));
    return h('section', { class: 'panel flush' }, h('div', { class: 'panel-head' }, h('h2', { text: 'Everything Vearl can compute' })), h('div', { class: 'panel-body' }, tbl),
      h('p', { class: 'note', text: 'The figures are the ones measured by the project on a GPU server with its own quantized models. Only the jobs marked available can be created here; the others need model files that are not installed yet.' }));
  }

  function jobList(jobs, empty) {
    if (!jobs.length) return h('div', { class: 'muted-note', text: empty });
    return h('ul', { class: 'jlist' }, jobs.map(function (j) {
      var th = j.status === 'done' && (j.kind === 'render' || j.kind === 'render2' || j.kind === 'render3') ? h('img', { src: GW + '/jobs/' + j.ref + '/image.png', alt: '', loading: 'lazy', width: '56', height: '56' }) : h('span', { text: j.kind === 'pi' ? 'π' : j.kind === 'digits' ? '0-9' : j.kind === 'matmul' ? 'A×B' : j.kind === 'gpt' ? 'Aa' : j.kind === 'vit' ? 'IMG' : '…' });
      return h('li', null, h('div', { class: 'th' }, th),
        h('div', { class: 't' }, a('#/job/' + j.ref, kindTitle(j.kind), ''), h('span', { text: summary(j) + ' · ' + ago(j.created) })),
        statusTag(j.status));
    }));
  }

  /* ------------------------------------------------------------------ job page */
  var STEPS = [
    ['Submitted', 'The service locked the escrow and sent the job to the chain.', 'open'],
    ['Taken by a miner', 'A worker node claimed it and locked a bond.', 'claimed'],
    ['Result posted', 'The miner committed to every chunk with a Merkle root.', 'posted'],
    ['Audited', 'Random chunks are recomputed and opened against that root.', 'verifying'],
    ['Accepted', 'The audit passed: the miner is paid and you get the output.', 'done']
  ];
  function stepIndex(s) { return { queued: 0, submitting: 0, open: 1, claimed: 2, posted: 3, verifying: 3, done: 5, rejected: 3, failed: 1 }[s]; }
  var STEPS_EXACT = [
    ['Submitted', 'The service sent the job to the chain and locked the escrow.'],
    ['Computed and checked', 'A miner computes it. The client checks every operation exactly: each answer must match its Merkle commitment and pass 24 rounds of Freivalds’ test.'],
    ['Accepted', 'Every check passed: the miner is paid and you get the output.']
  ];
  function lifecycleExact(j) {
    var idx = { queued: 0, running: 1, done: 3, failed: 1, rejected: 1 }[j.status], bad = j.status === 'failed' || j.status === 'rejected', t = j.times || {};
    return h('ul', { class: 'life three' }, STEPS_EXACT.map(function (s, i) {
      var cls = i < idx ? 'done' : i === idx ? 'now' : '';
      if (bad && i === idx) cls = 'done end-bad';
      var at = i === 0 ? (t.running || t.queued) : t.done;
      return h('li', { class: cls }, h('div', { class: 'n', text: '0' + (i + 1) }),
        h('h3', null, s[0], i === idx && !bad && j.status !== 'done' ? h('span', { class: 'spin', 'aria-hidden': 'true' }) : null),
        h('p', { text: s[1] }), at && i < idx ? h('div', { class: 'when', title: utc(at), text: t.queued ? '+ ' + clock(at - t.queued) : ago(at) }) : null);
    }));
  }
  function lifecycle(j) {
    if (isExact(j.kind)) return lifecycleExact(j);
    var idx = stepIndex(j.status), bad = j.status === 'failed' || j.status === 'rejected', t = j.times || {};
    return h('ul', { class: 'life five' }, STEPS.map(function (s, i) {
      var cls = i < idx ? 'done' : i === idx ? 'now' : '';
      if (bad && i === idx) cls = 'done end-bad';
      var at = i === 0 ? (t.open || t.submitting) : t[s[2]];
      return h('li', { class: cls }, h('div', { class: 'n', text: '0' + (i + 1) }),
        h('h3', null, s[0], i === idx && !bad && j.status !== 'done' ? h('span', { class: 'spin', 'aria-hidden': 'true' }) : null),
        h('p', { text: s[1] }), at && i < idx + (bad ? 1 : 0) ? h('div', { class: 'when', title: utc(at), text: t.queued ? '+ ' + clock(at - t.queued) : ago(at) }) : null);
    }));
  }
  function waitingFrame(j) {
    if (isMl(j.kind)) {
      var mnotes = (j.progress || []).slice(-4);
      return h('div', { class: 'waiting' }, j.kind === 'vit' && S.photos && S.photos[j.ref] ? h('img', { class: 'photo-prev', src: S.photos[j.ref], alt: 'Your photo' }) : null,
        h('div', { text: j.status === 'queued' ? 'Waiting in line: the GPU worker takes one text or photo job at a time.' : j.kind === 'embed' ? 'The network is reading your sentences. Ten chained jobs must each be verified before the next one can start; this takes a few minutes.' : j.kind === 'gpt' ? 'The model is writing. Each word is two jobs that must be verified before the next word can start.' : 'The network is looking at your photo. Two chained jobs, every operation verified.' }),
        mnotes.length ? h('ul', { class: 'prog' }, mnotes.map(function (n) { return h('li', { text: n }); })) : null);
    }
    if (isExact(j.kind)) {
      var notes = (j.progress || []).slice(-4);
      return h('div', { class: 'waiting' }, j.kind === 'matmul' ? blocksSvg(j.params.blocks) : h('div', { class: 'digit-pair' }, ((S.digitSamples || [])[j.params.sample] || { images: [] }).images.map(function (im) { return digitSvg(im, 'a handwritten digit'); })),
        h('div', { text: j.status === 'queued' ? 'Queued.' : 'A miner is computing, and the client checks every operation as it comes back. This usually takes under a minute.' }),
        notes.length ? h('ul', { class: 'prog' }, notes.map(function (n) { return h('li', { text: n }); })) : null);
    }
    var side = j.kind === 'pi' ? 0 : Math.min(j.params.size, 256);
    var msg = { queued: 'Queued.', submitting: 'Sending the job to the chain…', open: 'On the chain. Waiting for a miner to take it. This is the longest step: a block comes about every 10 seconds.', claimed: 'A miner is computing your job.', posted: 'The miner posted its result. The audit starts now.', verifying: 'Recomputing random chunks and opening them against the miner’s root…' }[j.status] || '';
    return h('div', { class: 'waiting' }, side ? tilesSvg(side) : piSvg(j.params.samples * j.params.chunks), h('div', { text: msg }));
  }
  function resultPanel(j) {
    if (j.result && j.result.type === 'text') {
      var tr = j.result, live = !isFinal(j.status), cut = tr.text.indexOf(tr.prompt) === 0 ? tr.prompt.length : 0;
      return h('div', null, h('div', { class: 'story' }, h('span', { class: 'sp-prompt', text: tr.text.slice(0, cut) }), h('span', { class: 'sp-new', text: tr.text.slice(cut) }), live ? h('span', { class: 'caret', 'aria-hidden': 'true' }) : null),
        h('div', { class: 'panel-body pad' }, h('p', { class: 'mute', text: tr.words + ' of ' + tr.wanted + ' words · ' + (tr.words ? (tr.seconds / tr.words).toFixed(1) : '0') + ' s per word. Each word was chosen from 50,257 integer scores after two verified jobs. The model is small and only knows children’s stories: expect simple, sometimes odd sentences.' }),
          j.status === 'done' ? h('div', { class: 'actions' }, runAgain(j)) : null));
    }
    if (j.status === 'done' && j.result && j.result.type === 'embedding') {
      var er = j.result, n = er.texts.length, best = null;
      for (var x = 0; x < n; x++) for (var y = x + 1; y < n; y++) if (!best || er.similarity[x][y] > best[0]) best = [er.similarity[x][y], x, y];
      var tbl = h('table', { class: 'mtab simtab' }, h('thead', null, h('tr', null, [h('th', { text: '' })].concat(er.texts.map(function (t, i) { return h('th', { title: t, text: String(i + 1) }); })))),
        h('tbody', null, er.similarity.map(function (row, i) {
          return h('tr', null, [h('th', { title: er.texts[i], text: String(i + 1) })].concat(row.map(function (v, k) {
            var td = h('td', { text: v.toFixed(2) });
            td.style.background = 'rgba(94,234,212,' + (Math.max(0, Math.min(1, v)) * 0.55).toFixed(2) + ')';
            if (i === k) td.style.opacity = '0.55';
            return td;
          })));
        })));
      return h('div', null, h('ol', { class: 'emb-prev emb-res' }, er.texts.map(function (t) { return h('li', { text: t }); })),
        h('div', { class: 'xtable-scroll' }, tbl),
        h('div', { class: 'panel-body pad' }, best ? h('p', { class: 'ink', text: 'Closest pair: ' + (best[1] + 1) + ' and ' + (best[2] + 1) + ' (similarity ' + best[0].toFixed(2) + ').' }) : null,
          h('p', { class: 'mute', text: 'Each text became ' + er.dim + ' numbers; the table shows their cosine similarity (1 = same meaning, near 0 = unrelated). First values of text 1: ' + er.head[0].join(', ') + '. The numbers are an 8-bit approximation of the original model (cosine about 0.77 with it): good for finding texts on the same topic, not an exact copy. What is verified is that the miner ran every operation exactly. Took ' + er.seconds + ' s.' }),
          h('div', { class: 'actions' }, runAgain(j))));
    }
    if (j.status === 'done' && j.result && j.result.type === 'photo') {
      var pr = j.result;
      return h('div', null, h('div', { class: 'photo-res' }, S.photos && S.photos[j.ref] ? h('img', { class: 'photo-prev', src: S.photos[j.ref], alt: 'The photo that was classified' }) : null,
        h('ul', { class: 'topk' }, pr.top.map(function (t, i) {
          var fill = h('i');
          setTimeout(function () { fill.style.width = Math.max(1, Math.round(t.prob * 100)) + '%'; }, 40);
          return h('li', { class: i === 0 ? 'first' : null }, h('span', { class: 'nm', text: t.name }), h('span', { class: 'bar' }, fill), h('span', { class: 'pc', text: (t.prob * 100).toFixed(1) + '%' }));
        }))),
        h('div', { class: 'panel-body pad' }, h('p', { class: 'mute', text: 'The network reads 1,000 everyday categories (ImageNet) and can be wrong, especially outside them. What is verified is that the miner ran it exactly. Took ' + pr.seconds + ' s.' }), h('div', { class: 'actions' }, runAgain(j))));
    }
    if (j.status === 'done' && j.result && j.result.type === 'digits') {
      var good = j.result.items.filter(function (it) { return it.predicted === it.label; }).length;
      return h('div', null, h('div', { class: 'digits-res' }, j.result.items.map(function (it) {
        var ok = it.predicted === it.label;
        return h('div', { class: 'dcard ' + (ok ? 'good' : 'bad') }, digitSvg(it.pixels, 'handwritten digit written as ' + it.label),
          h('div', { class: 'pred' }, 'Read as ', h('b', { text: String(it.predicted) })),
          h('div', { class: ok ? 'verdict ok' : 'verdict bad', text: ok ? 'Correct: it was written as ' + it.label : 'Wrong: it was written as ' + it.label }),
          logitBars(it.logits, it.predicted, it.label));
      })), h('div', { class: 'panel-body pad' }, h('p', { class: 'mute', text: good + ' of 2 read correctly. The network is small and can be wrong: what is verified is that the miner ran it exactly, not that its answer is right. The tall bar is the digit it chose; the underlined digit is the true one.' }), h('div', { class: 'actions' }, h('button', { class: 'btn', type: 'button', onclick: function () { var f = S.form || defaults(); f.kind = 'digits'; f.sample = (j.params.sample + 1) % 12; S.form = f; location.hash = '#/'; } }, 'Try the next pair'))));
    }
    if (j.status === 'done' && j.result && j.result.type === 'matrix') {
      var m = j.result, ct = h('table', { class: 'mtab' }, h('tbody', null, m.corner.map(function (r) { return h('tr', null, r.map(function (v) { return h('td', { text: num(v) }); })); })));
      return h('div', { class: 'panel-body' }, kv([
        ['Product', m.side + ' × ' + m.side + ' (int8 × int8 → int32)'], ['Block products', num(m.products) + ' of 128 × 128'], ['Multiply-adds', num(m.ops)],
        ['Time', (m.seconds != null ? m.seconds.toFixed(1) + ' s in all' : '') + (m.miner_seconds != null ? ', of which the miner needed ' + m.miner_seconds.toFixed(1) + ' s' : '')],
        ['Independent check', m.matching + ' of ' + m.checked + ' random entries recomputed by the service itself match']]),
        h('div', { class: 'lbl-s', text: 'Top-left corner of the product' }), h('div', { class: 'xtable-scroll' }, ct),
        h('div', { class: 'actions' }, runAgain(j)));
    }
    if (j.status === 'done' && j.result && j.result.type === 'image') {
      var url = GW + '/jobs/' + j.ref + '/image.png';
      return h('div', null, h('div', { class: 'frame' }, h('img', { src: url, class: isPath(j.kind) ? 'smooth' : null, alt: kindTitle(j.kind) + ', ' + j.result.width + ' by ' + j.result.height + ' pixels, computed by a Vearl miner', width: j.result.width, height: j.result.height })),
        h('div', { class: 'panel-body' }, h('div', { class: 'actions' }, h('a', { class: 'btn primary', href: url, download: 'vearl-job-' + j.ref + '.png' }, 'Download PNG'), runAgain(j))));
    }
    if (j.status === 'done' && j.result && j.result.type === 'number') {
      var r = j.result, err = Math.abs(r.pi - Math.PI), lo = r.pi - r.stderr, hi = r.pi + r.stderr, L = 3.0, R = 3.3;
      var pos = function (v) { return Math.max(0, Math.min(100, (v - L) / (R - L) * 100)); };
      var band = h('b'), mark = h('i');
      setTimeout(function () { band.style.left = pos(lo) + '%'; band.style.width = Math.max(1, pos(hi) - pos(lo)) + '%'; mark.style.left = pos(Math.PI) + '%'; }, 30);
      return h('div', { class: 'pi-big' }, h('div', { class: 'v' }, r.pi.toFixed(5), h('small', { text: '± ' + r.stderr.toFixed(4) })),
        h('p', { text: 'Estimated from ' + num(r.samples) + ' random points. The true value is 3.14159…: your estimate is off by ' + err.toFixed(5) + ', which ' + (err <= r.stderr * 2 ? 'is within' : 'is a bit outside') + ' the statistical margin.' }),
        h('div', { class: 'pi-bar', role: 'img', 'aria-label': 'Estimate with its margin, and the true value of pi' }, band, mark),
        h('p', { text: 'Lavender band: the estimate and its margin. White tick: the true value of π.' }),
        h('div', { class: 'actions' }, runAgain(j)));
    }
    return h('div', { class: 'frame' }, waitingFrame(j));
  }
  function runAgain(j) {
    return h('button', { class: 'btn', type: 'button', onclick: function () {
      var f = S.form || defaults(); f.kind = j.kind;
      Object.keys(j.params).forEach(function (k) { if (k !== 'kind' && k !== 'seed') f[k] = j.params[k]; });
      f.seed = String(Math.floor(Math.random() * 1e9)); S.form = f; location.hash = '#/';
    } }, isMl(j.kind) ? 'Run it again' : 'Run it again with a new seed');
  }
  function pageJob(ref) {
    return api('/jobs/' + encodeURIComponent(ref)).then(function (j) {
      remember(j.ref);
      document.title = kindTitle(j.kind) + ' · Vearl Compute';
      var root = h('div', null, head([{ text: 'Compute', href: '#/' }, { text: 'Job' }, { text: j.ref }], h('span', null, kindTitle(j.kind), ' ', h('span', { class: 'serif', text: 'job' })), summary(j)));
      var wrap = h('div', { class: 'wrap' });
      var since = j.times && j.times.queued;
      var final = isFinal(j.status), end = j.times && j.times[j.status];
      wrap.appendChild(h('div', { class: 'bighead' }, h('div', null, statusTag(j.status), ' ', since ? h('span', { class: 'mute', 'data-since': final ? null : since, text: ' ' + (final ? 'took ' + clock((end || since) - since) : clock(Date.now() / 1000 - since)) }) : null),
        h('div', { class: 'mute', text: final ? '' : 'This page updates by itself.' })));
      wrap.appendChild(h('div', { class: 'x-grid' }, lifecycle(j)));
      if (j.status === 'failed' || j.status === 'rejected') wrap.appendChild(h('div', { class: 'notice bad', role: 'alert', text: j.message || 'The job did not complete.' }));
      else if (j.status === 'done') wrap.appendChild(h('div', { class: 'notice ok', text: isExact(j.kind) ? 'Checked exactly and accepted: every operation matched the miner’s commitment.' : 'Audited and accepted. ' + (j.audit ? j.audit.checked + ' of ' + j.audit.of + ' chunks were recomputed and matched the miner’s commitment.' : '') }));
      var p = j.params, rows = [
        ['Job', withCopyRef(j.ref)], ['Type', kindTitle(j.kind)],
        j.kind === 'gpt' ? ['Story', p.prompt ? p.prompt + ' (+ ' + p.words + ' words)' : null] : j.kind === 'vit' ? ['Photo', 'one image, 224 × 224 (deleted after reading)'] : j.kind === 'embed' ? ['Texts', p.texts ? p.texts.length + ' (' + short(p.texts.join(' | '), 60, 0) + ')' : null] : j.kind === 'digits' ? ['Digits', 'pair ' + (p.sample + 1) + ' of 12 (test images)'] : j.kind === 'matmul' ? ['Matrices', p.size + ' × ' + p.size] : isPath(j.kind) ? ['Scene', sceneLabel('render3', p.scene) + ', ' + p.size + ' × ' + (p.height || p.size) + ' px, ' + p.quality + ' samples per pixel'] : [j.kind === 'pi' ? 'Random points' : 'Image', j.kind === 'pi' ? num(p.samples * p.chunks) + ' (' + p.samples + ' per chunk)' : p.size + ' × ' + p.size + ' px, ' + p.spp + ' samples per pixel'],
        isMl(j.kind) ? null : [isExact(j.kind) ? 'Block products / batches' : 'Chunks', num(j.chunks)], ['Seed', p.seed != null ? String(p.seed) : null],
        ['Created', since ? utc(since) + '  ·  ' + ago(since) : null],
        ['Progress', j.progress && j.progress.length ? j.progress[j.progress.length - 1] : null],
        isExact(j.kind) && !j.chain_id ? null : ['On the chain', j.chain_id ? h('span', null, a(EXPLORER + '#/job/' + j.chain_id, short(j.chain_id, 14, 10), 'lnk mono', 'See this job in the explorer'), copyBtn(j.chain_id)) : h('span', { class: 'mute', text: 'not sent yet' }), 'mono']
        ['Audit', j.audit ? j.audit.checked + ' of ' + j.audit.of + ' chunks recomputed' : null]
      ];
      wrap.appendChild(h('div', { class: 'result' }, h('section', { class: 'panel flush' }, h('div', { class: 'panel-head' }, h('h2', { text: j.status === 'done' ? 'Result' : 'Result (waiting)' })), resultPanel(j)),
        h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', { text: 'Details' })), h('div', { class: 'panel-body' }, kv(rows)))));
      wrap.appendChild(h('p', { class: 'sub-note', text: isExact(j.kind) ? 'Checked exactly: the client downloads every result, matches it to the miner’s Merkle commitment and tests it with 24 rounds of Freivalds’ algorithm, so a wrong entry slips through with probability below one in sixteen million. A single wrong entry is provable on chain and costs the miner its bond.' : 'The audit is by sampling: a miner that falsified one chunk among many could slip through, which is why the chance of catching it depends on how many chunks are recomputed. A wrong chunk that is checked is provable on chain with a proof of about a kilobyte, and the miner loses its bond.' }));
      root.appendChild(wrap);
      root._sig = j.status + '|' + (j.chain_id || '') + '|' + JSON.stringify(j.times) + '|' + (j.message || '') + '|' + (j.progress || []).length + '|' + (j.result && j.result.words != null ? j.result.words : '');
      if (!final) S.refresh = function () { route(true); };
      return root;
    });
  }
  function withCopyRef(ref) { return h('span', { class: 'mono', text: ref }); }

  /* ------------------------------------------------------------------ my jobs */
  function pageMine() {
    var refs = mine();
    return Promise.all(refs.map(function (r) { return api('/jobs/' + r).catch(function () { return null; }); })).then(function (jobs) {
      jobs = jobs.filter(Boolean);
      document.title = 'My jobs · Vearl Compute';
      var root = h('div', null, head([{ text: 'Compute', href: '#/' }, { text: 'My jobs' }], h('span', null, 'My ', h('span', { class: 'serif', text: 'jobs' })), 'The jobs this browser created. They are remembered here only, in this browser: there is no account.'));
      var wrap = h('div', { class: 'wrap' });
      wrap.appendChild(h('div', { class: 'x-grid' }, jobs.length ? h('section', { class: 'panel flush' }, h('div', { class: 'panel-body' }, jobList(jobs))) :
        h('div', { class: 'state' }, h('h2', null, 'No job ', h('span', { class: 'serif', text: 'yet' })), h('p', { text: 'Create one: it takes a few clicks and a few minutes.' }), h('a', { class: 'btn primary', href: '#/' }, 'Create a job'))));
      root.appendChild(wrap);
      root._sig = jobs.map(function (j) { return j.ref + j.status; }).join(',');
      if (jobs.some(function (j) { return !isFinal(j.status); })) S.refresh = function () { route(true); };
      return root;
    });
  }

  /* ------------------------------------------------------------------ how it works */
  function pageHow() {
    var info = S.info, L = info.limits;
    document.title = 'How it works · Vearl Compute';
    var root = h('div', null, head([{ text: 'Compute', href: '#/' }, { text: 'How it works' }], h('span', null, 'How a job ', h('span', { class: 'serif', text: 'works' })), 'What happens between your click and your result, and what is, and is not, guaranteed.'));
    var wrap = h('div', { class: 'wrap' });
    var items = [
      ['What a job is', 'A job is one of a few computations frozen into the Vearl protocol (a 3D render, a Monte Carlo estimate), plus its parameters, cut into independent chunks. Nobody runs arbitrary code: the chunks are integer-only, so they give the same bytes on any machine.'],
      ['Who pays', 'Nobody, for now. Vearl is a testnet and the tokens have no value. The service sends your job from its own funded account and locks ' + info.escrow + ' VRL for it. About a tenth of that is burned by the protocol, the rest goes to the miner when the audit passes. You never touch a key.'],
      ['Who computes', 'A miner running a worker node takes the job, locks a bond of ten percent of the escrow, computes every chunk and posts one Merkle root that commits to all of them. The service currently audits against one worker node; supporting several miners is the next step.'],
      ['How it is checked', 'Two ways. Renders and the π estimate are audited by sampling: the service recomputes ' + info.audit_chunks + ' random chunks itself and asks the miner to open each against its root; a chunk that does not match is proven wrong on chain, the miner loses its bond and is not paid. Matrix products and the digit network are checked exactly: the client verifies every result with a Merkle proof and 24 rounds of Freivalds’ test (an error slips through with probability below one in sixteen million), and a single wrong entry is enough to prove fraud.'],
      ['What the check does not do', 'A sampling audit is not a full proof: a miner that cheats on a single chunk among many has a fair chance of not being checked. The exact checks prove the miner ran the computation correctly, not that the model is right: the digit network, for instance, is small and sometimes reads a digit wrongly. The whitepaper explains the difference and why we say so.']
    ];
    wrap.appendChild(h('div', { class: 'how' }, items.map(function (it, i) { return h('div', null, h('div', { class: 'n', text: '0' + (i + 1) }), h('h3', { text: it[0] }), h('p', { text: it[1] })); })));
    wrap.appendChild(h('div', { class: 'x-grid two limits' },
      panel('Free limits', kv([['Jobs per visitor', L.per_hour + ' an hour'], ['At the same time', String(L.per_visitor_active) + ' per visitor'], ['On the whole service', L.max_active + ' in progress'], ['Chunks audited', String(info.audit_chunks) + ' per job']]), null, true),
      panel('What is on offer', h('ul', { class: 'dn-legend panel-body' }, KIND_ORDER.filter(function (k) { return info.kinds[k]; }).map(function (k) { return h('li', null, h('b', { text: info.kinds[k].title }), h('span', { class: 'n', text: info.kinds[k].short })); })), null, true)));
    wrap.appendChild(h('div', { class: 'actions' }, h('a', { class: 'btn primary', href: '#/' }, 'Create a job'), h('a', { class: 'btn', href: EXPLORER }, 'Open the explorer')));
    root.appendChild(wrap);
    return Promise.resolve(root);
  }

  /* ------------------------------------------------------------------ router and live status */
  function parseHash() {
    var raw = location.hash.replace(/^#/, '') || '/';
    var qi = raw.indexOf('?'), path = qi < 0 ? raw : raw.slice(0, qi);
    return path.split('/').filter(Boolean).map(decodeURIComponent);
  }
  function offline() {
    return h('div', { class: 'wrap' }, h('div', { class: 'state bad' }, h('h2', null, 'The job service is ', h('span', { class: 'serif', text: 'offline' })),
      h('p', { text: 'The page could not reach the Vearl job service. The public testnet is planned for 23 October 2026; until a worker network is online there is nothing to run jobs on.' }),
      h('button', { class: 'btn', type: 'button', onclick: function () { route(); } }, 'Try again')));
  }
  var navSeq = 0;
  function route(soft) {
    var my = ++navSeq, seg = parseHash(), s0 = seg[0] || '';
    if (!soft) { S.refresh = null; app.replaceChildren(loading()); window.scrollTo(0, 0); }
    document.querySelectorAll('[data-nav]').forEach(function (el) { el.classList.toggle('active', el.getAttribute('data-nav') === (s0 === 'job' ? 'home' : s0 || 'home')); });
    var go = S.info ? Promise.resolve() : loadInfo();
    return go.then(function () {
      if (my !== navSeq) return;
      var page;
      if (!s0) page = pageHome();
      else if (s0 === 'job' && seg[1]) page = pageJob(seg[1]);
      else if (s0 === 'mine') page = pageMine();
      else if (s0 === 'how') page = pageHow();
      else page = Promise.resolve(stateCard('Page not found', 'This address does not exist here.', false, false));
      return page;
    }).then(function (node) {
      if (!node || my !== navSeq) return;
      if (soft && node._sig && node._sig === S.sig) return;
      S.sig = node._sig || null;
      app.replaceChildren(node);
    }).catch(function (e) {
      if (my !== navSeq || soft) return;
      if (e && e.status == null) console.error(e);
      app.replaceChildren(e && (e.status === 404 || e.status === 400) ? stateCard('Nothing found', e.message, false, false) : offline());
    });
  }
  var pill = document.getElementById('pill'), pillText = document.getElementById('pill-text');
  var banner = document.getElementById('banner'), bannerText = document.getElementById('banner-text');
  function setPill() {
    if (S.online === false) { pill.className = 'status-pill bad'; pillText.textContent = 'Service offline'; return; }
    if (!S.info) { pill.className = 'status-pill warn'; pillText.textContent = 'Connecting'; return; }
    if (S.info.prelaunch) { pill.className = 'status-pill warn'; pillText.textContent = 'Opens ' + launchShort(S.info.launch); return; }
    pill.className = 'status-pill ok';
    pillText.textContent = 'Free · ' + S.info.active + ' in progress';
  }
  function loadInfo() {
    return api('/info').then(function (i) {
      S.info = i; if (S.online !== true) { S.online = true; banner.classList.remove('show'); }
      if (i.prelaunch) { banner.classList.add('show'); banner.classList.remove('bad'); bannerText.textContent = 'Preview. Job creation opens when the testnet starts, on ' + launchLong(i.launch) + '. You can explore everything now.'; }
      setPill();
    }, function (e) {
      S.online = false; banner.classList.add('show', 'bad'); bannerText.textContent = 'The job service cannot be reached. Retrying every few seconds.'; setPill(); throw e;
    });
  }
  setInterval(function () { loadInfo().catch(function () {}); }, 15000);
  setInterval(function () { if (S.refresh) S.refresh(); }, 4000);
  setInterval(function () {
    document.querySelectorAll('[data-since]').forEach(function (el) { var s = parseInt(el.getAttribute('data-since'), 10); if (s) el.textContent = ' ' + clock(Date.now() / 1000 - s); });
  }, 1000);

  var nav = document.getElementById('nav'), burger = document.getElementById('burger');
  burger.addEventListener('click', function () { var open = nav.classList.toggle('open'); burger.setAttribute('aria-expanded', String(open)); });
  document.getElementById('menu').addEventListener('click', function (e) { if (e.target.closest('a')) { nav.classList.remove('open'); burger.setAttribute('aria-expanded', 'false'); } });
  window.addEventListener('hashchange', function () { route(false); });
  route(false);
})();
