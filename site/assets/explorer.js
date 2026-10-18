/* Vearl testnet explorer: a single-page reader of the node's JSON API (/api/*).
   No libraries, no third-party requests, no innerHTML: every node is built with textContent, which is what the strict CSP expects. */
(function () {
  'use strict';

  var API = (document.querySelector('meta[name="vearl-api"]') || {}).content || '/api';
  var app = document.getElementById('app');
  var BLOCK_TARGET_S = 10;                  // blocks every 10 s (ASERT target)

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
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  /* ------------------------------------------------------------------ API */
  function api(path, opts) {
    var ctl = new AbortController();
    var timer = setTimeout(function () { ctl.abort(); }, (opts && opts.timeout) || 9000);
    return fetch(API + path, { signal: ctl.signal, headers: { Accept: 'application/json' } })
      .then(function (r) {
        return r.json().catch(function () { return null; }).then(function (body) {
          if (!r.ok) {
            var e = new Error((body && body.error) || ('HTTP ' + r.status));
            e.status = r.status;
            throw e;
          }
          return body;
        });
      })
      .catch(function (e) {
        if (e && e.name === 'AbortError') { e = new Error('The node did not answer in time.'); }
        if (e && e.status == null) e.status = 0;
        throw e;
      })
      .then(function (v) { clearTimeout(timer); return v; }, function (e) { clearTimeout(timer); throw e; });
  }

  /* ------------------------------------------------------------------ formatting */
  var nf = new Intl.NumberFormat('en-US');
  function num(n) { return nf.format(n); }
  function amt(s) {
    s = String(s == null ? '0' : s);
    var p = s.split('.');
    var frac = (p[1] || '').replace(/0+$/, '');
    var whole;
    try { whole = nf.format(BigInt(p[0] || '0')); } catch (e) { whole = p[0]; }
    return whole + (frac ? '.' + frac : '');
  }
  function units(s) {
    var p = String(s || '0').split('.');
    return BigInt(p[0] || '0') * 100000000n + BigInt(((p[1] || '') + '00000000').slice(0, 8));
  }
  function fromUnits(u) {
    var neg = u < 0n; if (neg) u = -u;
    var w = u / 100000000n, f = String(u % 100000000n).padStart(8, '0');
    return (neg ? '-' : '') + w.toString() + '.' + f;
  }
  function short(x, a, b) {
    x = String(x || ''); a = a || 10; b = b || 8;
    return x.length > a + b + 1 ? x.slice(0, a) + '…' + x.slice(-b) : x;
  }
  function human(s) { return String(s || '').replace(/_/g, ' '); }
  function cap(s) { s = human(s); return s.charAt(0).toUpperCase() + s.slice(1); }
  function si(v, unit) {
    var u = ['', 'k', 'M', 'G', 'T', 'P', 'E'], i = 0;
    while (v >= 1000 && i < u.length - 1) { v /= 1000; i++; }
    return (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)) + u[i] + (unit ? ' ' + unit : '');
  }
  var S = { status: null, skew: 0, online: null, refresh: null, ribbonTop: -1 };
  function nowSrv() { return Date.now() / 1000 + S.skew; }
  function ago(ts) {
    var d = Math.max(0, Math.round(nowSrv() - ts));
    if (d < 5) return 'just now';
    if (d < 60) return d + ' s ago';
    if (d < 3600) return Math.floor(d / 60) + ' min ago';
    if (d < 86400) return Math.floor(d / 3600) + ' h ago';
    return Math.floor(d / 86400) + ' d ago';
  }
  function utc(ts) { return new Date(ts * 1000).toISOString().replace('T', ' ').slice(0, 19) + ' UTC'; }
  function dur(s) {
    s = Math.max(0, Math.round(s));
    if (s < 90) return s + ' s';
    if (s < 5400) return Math.round(s / 60) + ' min';
    if (s < 172800) return (s / 3600).toFixed(1).replace(/\.0$/, '') + ' h';
    return (s / 86400).toFixed(1).replace(/\.0$/, '') + ' d';
  }
  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  function dayShort(d) { var p = String(d).split('-'); return (MONTHS[parseInt(p[1], 10) - 1] || p[1]) + ' ' + parseInt(p[2], 10); }
  var EVENT_LABEL = {
    submitted: 'Submitted', bid: 'Bid placed', claimed: 'Claimed', posted: 'Result posted', accepted: 'Accepted', settled: 'Settled',
    fraud_proof: 'Fraud proof', data_requested: 'Data requested', data_served: 'Data served', slash_unserved: 'Slashed, data not served', work_block: 'Block mined for it'
  };
  var EVENT_TAG = {
    submitted: 't-new', bid: 't-native', claimed: 't-job', posted: 't-layer', accepted: 't-ok', settled: 't-ok',
    fraud_proof: 't-bad', data_requested: 't-native', data_served: 't-native', slash_unserved: 't-bad', work_block: 't-product'
  };
  function eventTag(w) { return h('span', { class: 'tag ' + (EVENT_TAG[w] || 't-native'), text: EVENT_LABEL[w] || human(w) }); }

  /* ------------------------------------------------------------------ work classes, statuses, kinds */
  var CLASSES = ['native', 'job', 'layer', 'program_op', 'product'];
  var CLASS_CSS = ['c-native', 'c-job', 'c-layer', 'c-op', 'c-product'];
  var CLASS_TAG = ['t-native', 't-job', 't-layer', 't-op', 't-product'];
  var CLASS_LABEL = ['Native', 'Paid job', 'Inference layer', 'Program op', 'Tiled product'];
  var CLASS_DESC = {
    native: 'Native work: the matrices come from the epoch, no client is paying for this block.',
    job: 'Paid work: the matrices of a client job were mined, the block also advances that job.',
    layer: 'Paid work: one layer of a verified inference job.',
    program_op: 'Paid work: one operation of a verified tensor program.',
    product: 'Paid work: one block product of a tiled matrix job.'
  };
  function classIndex(type) { var i = CLASSES.indexOf(type); return i < 0 ? 0 : i; }
  var KIND_LABEL = {
    matrix_product: 'Matrix product', inference: 'Inference', tensor_program: 'Tensor program',
    tiled_matrix_product: 'Tiled matrix product', level2: 'Level 2'
  };
  var KERNEL_LABEL = { monte_carlo: 'Monte Carlo', render: '3D render', render2: 'Render2 · GPU ray marching', render3: 'Render3 · GPU path tracer' };
  function statusTag(st) {
    var m = { settled: 't-ok', slashed: 't-bad', expired: 't-bad', posted: 't-layer', complete: 't-layer', claimed: 't-job', working: 't-job', open: 't-native' };
    return h('span', { class: 'tag ' + (m[st] || 't-native'), text: st });
  }

  /* ------------------------------------------------------------------ components */
  function a(href, text, cls, title) { return h('a', { href: href, class: cls || 'lnk', title: title || null }, text); }
  function blockLink(id, text) { return a('#/block/' + id, text == null ? '#' + id : text); }
  function addrLink(addr, full) { return a('#/address/' + addr, full ? addr : short(addr, 12, 6), 'lnk mono', addr); }
  function jobLink(id, full) { return a('#/job/' + id, full ? id : short(id, 10, 6), 'lnk mono', id); }
  function txLink(id, full) { return a('#/tx/' + id, full ? id : short(id, 10, 6), 'lnk mono', id); }
  function icon() {
    return sv('svg', { viewBox: '0 0 16 16', width: '12', height: '12', 'aria-hidden': 'true' },
      sv('rect', { x: '5.5', y: '5.5', width: '8', height: '8', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.3' }),
      sv('path', { d: 'M3 10.5V3h7.5', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.3' }));
  }
  function copyBtn(text) {
    var b = h('button', { class: 'copy', type: 'button', 'aria-label': 'Copy to clipboard', title: 'Copy' }, icon());
    b.addEventListener('click', function () {
      function done() { b.classList.add('done'); setTimeout(function () { b.classList.remove('done'); }, 1200); }
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, function () {});
      else {
        var t = h('textarea', { 'aria-hidden': 'true' }); t.value = text; document.body.appendChild(t); t.select();
        try { document.execCommand('copy'); done(); } catch (e) { /* nothing to do */ }
        document.body.removeChild(t);
      }
    });
    return b;
  }
  function withCopy(node, text) { return h('span', null, node, copyBtn(text)); }

  function table(cols, rows, opts) {
    opts = opts || {};
    var thead = h('thead', null, h('tr', null, cols.map(function (c) { return h('th', { class: c.cls || null, text: c.k }); })));
    var tbody = h('tbody', null, rows.map(function (r) {
      var tr = h('tr', { class: opts.rowClass ? opts.rowClass(r) : null });
      cols.forEach(function (c) {
        var cell = c.cell(r);
        tr.appendChild(h('td', { class: c.cls || null, 'data-k': c.k }, cell));
      });
      return tr;
    }));
    return h('div', { class: 'xtable-scroll' }, h('table', { class: 'xtable stack' + (opts.compact ? ' compact' : '') }, thead, tbody));
  }
  function panel(title, body, more, flush) {
    return h('section', { class: 'panel' + (flush ? ' flush' : '') },
      h('div', { class: 'panel-head' }, h('h2', { text: title }), more ? a(more.href, more.text, 'more') : null),
      h('div', { class: 'panel-body' }, body));
  }
  function kv(rows) {
    var el = h('div', { class: 'kv' });
    rows.forEach(function (r) {
      if (r[1] == null) return;
      el.appendChild(h('div', { class: 'k', text: r[0] }));
      el.appendChild(h('div', { class: 'v' + (r[2] ? ' ' + r[2] : '') }, r[1]));
    });
    return el;
  }
  function crumbs(parts) {
    var el = h('div', { class: 'crumbs' });
    parts.forEach(function (p, i) {
      if (i) el.appendChild(h('i', { text: '/' }));
      el.appendChild(p.href ? a(p.href, p.text, '') : h('span', { text: p.text }));
    });
    return el;
  }
  function searchForm(compact) {
    var input = h('input', { type: 'text', id: compact ? null : 'q', name: 'q', autocomplete: 'off', spellcheck: 'false', 'aria-label': 'Search by block height, block hash, transaction, job or address', placeholder: 'Block height, hash, transaction, job or vrl1… address' });
    var msg = h('div', { class: 'x-search-msg', role: 'status' });
    var form = h('form', { class: 'x-search' + (compact ? ' compact' : ''), role: 'search' }, input, compact ? null : h('kbd', { text: '/' }), h('button', { type: 'submit', text: 'Search' }));
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var q = input.value.trim();
      if (!q) return;
      msg.textContent = '';
      api('/search?q=' + encodeURIComponent(q)).then(function (r) {
        location.hash = '#/' + (r.type === 'job' ? 'job' : r.type) + '/' + r.id;
      }).catch(function (err) {
        msg.textContent = err.status === 404 || err.status === 400 ? err.message : 'The node could not be reached.';
      });
    });
    return h('div', null, form, msg);
  }
  function head(parts, title, lede, withSearch) {
    return h('div', { class: 'x-page-head' }, h('div', { class: 'wrap' }, crumbs(parts), h('h1', null, title), lede ? h('p', { text: lede }) : null, withSearch === false ? null : searchForm(true)));
  }
  function stateCard(title, text, bad, retry) {
    return h('div', { class: 'wrap' }, h('div', { class: 'state' + (bad ? ' bad' : '') },
      h('h2', null, title), h('p', { text: text }),
      retry ? h('button', { class: 'btn', type: 'button', onclick: function () { route(); } }, 'Try again') : null));
  }
  function loading() {
    return h('div', { class: 'wrap loading' }, h('span', { class: 'skel' }), h('span', { class: 'skel' }), h('span', { class: 'skel' }));
  }
  function pager(prev, next) {
    function btn(o, label) { return o ? h('a', { class: 'btn', href: o }, label) : h('span', { class: 'btn', 'aria-disabled': 'true' }, label); }
    return h('div', { class: 'pager' }, btn(prev, '← Newer'), btn(next, 'Older →'));
  }

  /* ------------------------------------------------------------------ charts (inline SVG, no styles attributes) */
  function barChart(values, o) {
    o = o || {};
    var W = 600, H = 150, L = 40, T = 8, B = 18, n = values.length;
    var max = Math.max.apply(null, values.concat([o.min || 1e-9]));
    if (o.ref && o.ref > max) max = o.ref;
    max = max * 1.1;
    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': o.label || 'chart', preserveAspectRatio: 'xMidYMid meet' });
    [0, 0.5, 1].forEach(function (f) {
      var y = T + (H - T - B) * (1 - f);
      svg.appendChild(sv('line', { x1: L, x2: W, y1: y, y2: y, class: 'ax' }));
      svg.appendChild(sv('text', { x: L - 6, y: y + 3, 'text-anchor': 'end', class: 'ax-t' }, (o.fmt || String)(Math.round(max * f * 10) / 10)));
    });
    var bw = (W - L) / Math.max(n, 1);
    values.forEach(function (v, i) {
      var bh = Math.max(1, (H - T - B) * (v / max));
      var r = sv('rect', { x: L + i * bw + 1, y: H - B - bh, width: Math.max(bw - 2, 1), height: bh, class: (o.cls || 'bar-a') + (o.partialLast && i === n - 1 ? ' bar-today' : '') });
      r.appendChild(sv('title', null, (o.titles && o.titles[i]) || String(v)));
      svg.appendChild(r);
      if (o.labels && o.labels[i] != null) svg.appendChild(sv('text', { x: L + i * bw + bw / 2, y: H - 5, class: 'ax-l' }, o.labels[i]));
    });
    if (o.ref) {
      var ry = H - B - (H - T - B) * (o.ref / max);
      svg.appendChild(sv('line', { x1: L, x2: W, y1: ry, y2: ry, class: 'ref' }));
    }
    if (o.x0 != null) {
      svg.appendChild(sv('text', { x: L, y: H - 3, class: 'ax-t' }, o.x0));
      svg.appendChild(sv('text', { x: W, y: H - 3, 'text-anchor': 'end', class: 'ax-t' }, o.x1));
    }
    return svg;
  }
  function lineChart(values, o) {
    o = o || {};
    var W = 600, H = 150, L = 40, T = 8, B = 18, n = values.length;
    var max = Math.max.apply(null, values.concat([1e-9])) * 1.1, min = 0;
    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': o.label || 'chart', preserveAspectRatio: 'xMidYMid meet' });
    var defs = sv('defs', null, sv('linearGradient', { id: 'gradA', x1: '0', y1: '0', x2: '0', y2: '1' },
      sv('stop', { offset: '0', 'stop-color': '#9ff0e2', 'stop-opacity': '.35' }), sv('stop', { offset: '1', 'stop-color': '#9ff0e2', 'stop-opacity': '0' })));
    svg.appendChild(defs);
    [0, 0.5, 1].forEach(function (f) {
      var y = T + (H - T - B) * (1 - f);
      svg.appendChild(sv('line', { x1: L, x2: W, y1: y, y2: y, class: 'ax' }));
      svg.appendChild(sv('text', { x: L - 6, y: y + 3, 'text-anchor': 'end', class: 'ax-t' }, (o.fmt || String)(max * f)));
    });
    if (n > 1) {
      var pts = values.map(function (v, i) { return [L + (W - L) * i / (n - 1), H - B - (H - T - B) * ((v - min) / (max - min))]; });
      var d = pts.map(function (p, i) { return (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' ');
      svg.appendChild(sv('path', { d: d + ' L' + W + ' ' + (H - B) + ' L' + L + ' ' + (H - B) + ' Z', class: 'area-a' }));
      svg.appendChild(sv('path', { d: d, class: 'line-a' }));
    }
    if (o.x0 != null) {
      svg.appendChild(sv('text', { x: L, y: H - 3, class: 'ax-t' }, o.x0));
      svg.appendChild(sv('text', { x: W, y: H - 3, 'text-anchor': 'end', class: 'ax-t' }, o.x1));
    }
    return svg;
  }
  /* two series side by side (a = first colour, b = second) */
  function groupChart(a, b, o) {
    o = o || {};
    var W = 600, H = 150, L = 40, T = 8, B = 18, n = a.length;
    var max = Math.max.apply(null, a.concat(b, [o.min || 1])) * 1.1;
    var svg = sv('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': o.label || 'chart', preserveAspectRatio: 'xMidYMid meet' });
    [0, 0.5, 1].forEach(function (f) {
      var y = T + (H - T - B) * (1 - f);
      svg.appendChild(sv('line', { x1: L, x2: W, y1: y, y2: y, class: 'ax' }));
      svg.appendChild(sv('text', { x: L - 6, y: y + 3, 'text-anchor': 'end', class: 'ax-t' }, (o.fmt || String)(Math.round(max * f * 10) / 10)));
    });
    var gw = (W - L) / Math.max(n, 1), bw = Math.max((gw - 3) / 2, 1);
    [a, b].forEach(function (vals, k) {
      vals.forEach(function (v, i) {
        var bh = v > 0 ? Math.max(1, (H - T - B) * (v / max)) : 0;
        if (!bh) return;
        var r = sv('rect', { x: L + i * gw + 1 + k * bw, y: H - B - bh, width: bw, height: bh, class: k ? 'bar-b' : 'bar-a' });
        r.appendChild(sv('title', null, (o.titles && o.titles[i]) || String(v)));
        svg.appendChild(r);
      });
    });
    if (o.x0 != null) {
      svg.appendChild(sv('text', { x: L, y: H - 3, class: 'ax-t' }, o.x0));
      svg.appendChild(sv('text', { x: W, y: H - 3, 'text-anchor': 'end', class: 'ax-t' }, o.x1));
    }
    return svg;
  }
  function donut(segs, centre, sub) {
    var R = 70, C = 2 * Math.PI * R, total = segs.reduce(function (s, x) { return s + x.value; }, 0) || 1, off = 0;
    var svg = sv('svg', { viewBox: '0 0 180 180', role: 'img', 'aria-label': sub || 'distribution' });
    svg.appendChild(sv('circle', { cx: 90, cy: 90, r: R, class: 'dn-bg' }));
    segs.forEach(function (s, i) {
      var len = C * s.value / total;
      svg.appendChild(sv('circle', { cx: 90, cy: 90, r: R, class: 'dn-seg dn-' + (i % 7), 'stroke-dasharray': len.toFixed(2) + ' ' + (C - len).toFixed(2), 'stroke-dashoffset': (-off).toFixed(2), transform: 'rotate(-90 90 90)' }));
      off += len;
    });
    svg.appendChild(sv('text', { x: 90, y: 92, class: 'dn-c' }, centre));
    svg.appendChild(sv('text', { x: 90, y: 110, class: 'dn-s' }, (sub || '').toUpperCase()));
    return svg;
  }
  function legend(segs, fmt) {
    return h('ul', { class: 'dn-legend' }, segs.map(function (s, i) {
      return h('li', null, h('i', { class: 'sw-' + (i % 7) }), s.node || h('b', { text: s.label }), h('span', { class: 'n', text: fmt ? fmt(s) : String(s.value) }));
    }));
  }

  /* ------------------------------------------------------------------ derived numbers */
  var TWO64 = 1n << 64n;
  function attempts(target) { try { return Number(TWO64 / (BigInt(target) + 1n)); } catch (e) { return 0; } }
  function seriesStats(ch) {
    var n = ch.heights.length;
    var o = { n: n, blockTime: null, rate: null, rates: [], dts: [], useful: 0 };
    if (n < 2) { if (n === 1) o.useful = ch.classes[0] ? 1 : 0; return o; }
    var sumA = 0;
    for (var i = 1; i < n; i++) {
      var dt = Math.max(1, ch.timestamps[i] - ch.timestamps[i - 1]);
      o.dts.push(dt);
      var at = attempts(ch.targets[i]);
      sumA += at;
      o.rates.push(at / dt);
    }
    var span = ch.timestamps[n - 1] - ch.timestamps[0];
    o.blockTime = span / (n - 1);
    o.rate = span > 0 ? sumA / span : null;
    var u = 0; ch.classes.forEach(function (c) { if (c) u++; });
    o.useful = u / n;
    return o;
  }
  function latestBlock(st) { return st.height - 1; }

  /* ------------------------------------------------------------------ pages */
  function pageHome() {
    return Promise.all([api('/status'), api('/chart?blocks=400'), api('/blocks?limit=10'), api('/mempool'), api('/miners?window=1000'), api('/jobs?limit=6')]).then(function (r) {
      var st = r[0], ch = r[1], bl = r[2], mp = r[3], mi = r[4], jb = r[5];
      var ss = seriesStats(ch);
      var root = h('div', null);
      document.title = 'Vearl Explorer · Testnet blocks, jobs and miners';

      // hero
      root.appendChild(h('section', { class: 'x-hero' }, h('div', { class: 'wrap' },
        h('div', { class: 'eyebrow' }, 'Testnet explorer'),
        h('h1', null, 'Every block, ', h('span', { class: 'serif' }, 'every job, in the open.')),
        h('p', { class: 'lede', text: 'Blocks, transactions, compute jobs, miners and balances, read straight from a Vearl node. Nothing here is estimated by us: if the node says it, you can check it.' }),
        searchForm(false))));

      // ribbon
      var wrap = h('div', { class: 'wrap' });
      root.appendChild(wrap);
      wrap.appendChild(ribbon(ch));

      // stats
      var jobsOpen = (jb.by_status.open || 0) + (jb.by_status.claimed || 0) + (jb.by_status.working || 0) + (jb.by_status.posted || 0) + (jb.by_status.complete || 0);
      var rateTxt = ss.rate ? si(ss.rate, '') : 'n/a';
      wrap.appendChild(h('div', { class: 'xstats' },
        stat('Latest block', st.height > 0 ? '#' + num(latestBlock(st)) : '—', st.height > 0 ? ago(st.tip_timestamp) : 'waiting for the first block'),
        stat('Block time', ss.blockTime ? ss.blockTime.toFixed(1) : '—', 'seconds · target ' + BLOCK_TARGET_S, ss.blockTime ? 's' : null),
        stat('Network rate', rateTxt, 'lottery tickets per second, from the difficulty targets', ss.rate ? 'tickets/s' : null),
        stat('Miners', num(mi.distinct_miners), 'in the last ' + num(mi.blocks) + ' blocks'),
        stat('Jobs', num(jb.total), num(jobsOpen) + ' in progress'),
        stat('Supply', amt(st.supply.minted), 'VRL minted · ' + amt(st.supply.burned) + ' burned')));

      // activity
      var latest = latestBlocksPanel(bl);
      var activity = h('div', null);
      wrap.appendChild(h('div', { class: 'x-grid wide-left' }, latest, panel('Transactions', activity, { href: '#/mempool', text: 'Pending (' + mp.count + ')' }, true)));
      recentTxs(ch, mp).then(function (rows) { activity.appendChild(rows); });

      // charts
      var titles = ss.dts.map(function (d, i) { return '#' + ch.heights[i + 1] + ': ' + d + ' s'; });
      var dtv = ss.dts.slice(-80), dtt = titles.slice(-80);
      var cw = h('div', { class: 'x-grid three' });
      cw.appendChild(panel('Block time (s)', h('div', { class: 'chart' }, dtv.length ? barChart(dtv, { ref: BLOCK_TARGET_S, titles: dtt, label: 'Seconds between blocks', fmt: function (v) { return String(Math.round(v)); }, x0: dtv.length ? '#' + ch.heights[ch.heights.length - dtv.length] : '', x1: '#' + ch.heights[ch.heights.length - 1] }) : emptyChart(), h('div', { class: 'legend' }, h('span', { text: 'dashed line: ' + BLOCK_TARGET_S + ' s target' })))));
      var rv = ss.rates.slice(-80);
      cw.appendChild(panel('Network rate (tickets/s)', h('div', { class: 'chart' }, rv.length > 1 ? lineChart(rv, { label: 'Lottery tickets per second', fmt: function (v) { return si(v, ''); }, x0: '#' + ch.heights[ch.heights.length - rv.length], x1: '#' + ch.heights[ch.heights.length - 1] }) : emptyChart(), h('div', { class: 'legend' }, h('span', { text: 'expected tickets per block ÷ block time' })))));
      var tx = ch.txs.slice(-80), txt = ch.heights.slice(-80).map(function (x, i) { return '#' + x + ': ' + tx[i] + ' tx'; });
      cw.appendChild(panel('Transactions per block', h('div', { class: 'chart' }, tx.length ? barChart(tx, { cls: 'bar-l', min: 4, titles: txt, label: 'Transactions per block', fmt: function (v) { return String(Math.round(v)); }, x0: '#' + ch.heights[ch.heights.length - tx.length], x1: '#' + ch.heights[ch.heights.length - 1] }) : emptyChart(), h('div', { class: 'legend' }, h('span', { text: 'last ' + tx.length + ' blocks' })))));
      wrap.appendChild(cw);

      // miners + useful work + jobs
      var segs = minerSegments(mi);
      var minersBody = mi.blocks ? h('div', { class: 'donut-wrap' }, donut(segs, String(mi.distinct_miners), 'miners'), legend(segs, function (s) { return s.value + ' blocks'; })) : h('div', { class: 'panel-body' }, h('p', { class: 'mute', text: 'No blocks yet.' }));
      var fill = h('div', { class: 'fill' });
      var pct = Math.round(ss.useful * 1000) / 10;
      setTimeout(function () { fill.style.width = Math.min(100, pct) + '%'; }, 60);
      var meter = h('div', { class: 'meter' },
        h('div', { class: 'big' }, pct.toFixed(1) + '%', h('small', { text: 'of the last ' + ss.n + ' blocks carried paid work' })),
        h('div', { class: 'track' }, fill),
        h('p', { text: 'A native block secures the chain with epoch matrices. A paid block mines the matrices of a real client job, so the energy that secures the network also produces a result someone needs.' }));
      wrap.appendChild(h('div', { class: 'x-grid three' },
        panel('Who mines', minersBody, { href: '#/miners', text: 'Leaderboard' }, true),
        panel('Useful work', meter, null, true),
        panel('Latest jobs', jobsMini(jb), { href: '#/jobs', text: 'All jobs' }, true)));

      // job activity + last two weeks: these need the node's index (newer nodes); the section stays out when the node does not have it
      var feedBody = h('div', null, h('div', { class: 'muted-note', text: 'Loading…' }));
      var weeksBody = h('div', null, h('div', { class: 'muted-note', text: 'Loading…' }));
      var extra = h('div', { class: 'x-grid two' }, panel('Job activity', feedBody, { href: '#/stats', text: 'Daily stats' }, true), panel('Blocks per day', weeksBody, { href: '#/stats', text: 'More charts' }, true));
      wrap.appendChild(extra);
      Promise.all([api('/events?limit=8'), api('/stats/daily?days=14')]).then(function (x) {
        feedBody.replaceChildren(x[0].events.length ? eventsTable(x[0].events, true) : h('div', { class: 'muted-note feed-empty', text: 'No job has moved yet: submissions, claims, results and proofs will appear here as they are mined.' }));
        var days = x[1].days.slice().reverse();
        weeksBody.replaceChildren(days.length ? h('div', { class: 'chart' }, barChart(days.map(function (d) { return d.blocks; }), {
          label: 'Blocks per day', partialLast: true, labels: days.length <= 14 ? days.map(function (d) { return dayShort(d.day).split(' ')[1]; }) : null,
          titles: days.map(function (d) { return d.day + ': ' + d.blocks + ' blocks, ' + d.miners + ' miners'; }), fmt: function (v) { return String(Math.round(v)); }, min: 4
        }), h('div', { class: 'legend' }, h('span', { text: 'UTC days · the faded bar is the day in progress' })),
        h('div', { class: 'chart-sep' }),
        groupChart(days.map(function (d) { return d.jobs_submitted; }), days.map(function (d) { return d.jobs_closed; }), {
          label: 'Jobs submitted and closed per day', min: 3, fmt: function (v) { return String(Math.round(v)); },
          x0: dayShort(days[0].day), x1: dayShort(days[days.length - 1].day),
          titles: days.map(function (d) { return d.day + ': ' + d.jobs_submitted + ' submitted, ' + d.jobs_closed + ' closed'; })
        }), h('div', { class: 'legend' }, h('span', null, h('i', { class: 'k-a' }), 'jobs submitted'), h('span', null, h('i', { class: 'k-b' }), 'jobs closed'))) : h('div', { class: 'muted-note', text: 'No block yet.' }));
      }).catch(function () { extra.remove(); });
      return root;
    });
  }
  /* ------------------------------------------------------------------ before the launch: no node yet, so no data and nothing invented */
  var LAUNCH = (function () { var m = document.querySelector('meta[name="vearl-launch"]'); var t = m ? Date.parse(m.content) : NaN; return isNaN(t) ? 0 : t; })();
  var MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  /* A page may be published BEFORE the launch with a demo network behind it: it then says so on every screen (a banner that cannot be closed).
     The banner stays for as long as the node's genesis is the demo one, even after the launch date, so demo numbers are never shown as the testnet. */
  var DEMO = (document.querySelector('meta[name="vearl-demo-genesis"]') || {}).content || '';
  function preLaunch() { return LAUNCH > 0 && Date.now() < LAUNCH && !DEMO; }
  function sampleActive() { return !!DEMO && ((LAUNCH > 0 && Date.now() < LAUNCH) || (S.status && S.status.genesis_hash === DEMO)); }
  function launchDate() { var d = new Date(LAUNCH); return d.getUTCDate() + ' ' + MONTHS[d.getUTCMonth()]; }
  function launchLong() { var d = new Date(LAUNCH); return launchDate() + ' ' + d.getUTCFullYear() + ' at ' + String(d.getUTCHours()).padStart(2, '0') + ':' + String(d.getUTCMinutes()).padStart(2, '0') + ' UTC'; }
  function countdown() {
    var s = Math.max(0, Math.floor((LAUNCH - Date.now()) / 1000));
    return { d: Math.floor(s / 86400), h: Math.floor(s % 86400 / 3600), m: Math.floor(s % 3600 / 60), s: s % 60 };
  }
  function pad2(n) { return String(n).padStart(2, '0'); }
  function tickCountdown() {
    var c = countdown();
    document.querySelectorAll('[data-cd]').forEach(function (el) { el.textContent = c[el.getAttribute('data-cd')] < 100 ? pad2(c[el.getAttribute('data-cd')]) : String(c[el.getAttribute('data-cd')]); });
  }
  function pagePrelaunch() {
    document.title = 'Vearl Explorer · The testnet starts on ' + launchDate();
    var root = h('div', null);
    function cell(k, label) { return h('div', null, h('b', { 'data-cd': k, text: '00' }), h('span', { text: label })); }
    root.appendChild(h('section', { class: 'x-hero' }, h('div', { class: 'wrap' },
      h('div', { class: 'eyebrow' }, 'Testnet explorer'),
      h('h1', null, 'The network starts ', h('span', { class: 'serif', text: 'on ' + launchDate() + '.' })),
      h('p', { class: 'lede', text: 'Blocks, transactions, compute jobs, miners and balances will appear here as soon as the first block is mined, on ' + launchLong() + '. Until then there is nothing to read: no demo data and no placeholder.' }),
      h('div', { class: 'cd', role: 'timer', 'aria-label': 'Time left before the testnet starts' }, cell('d', 'days'), cell('h', 'hours'), cell('m', 'minutes'), cell('s', 'seconds')))));
    var wrap = h('div', { class: 'wrap' });
    root.appendChild(wrap);
    wrap.appendChild(h('div', { class: 'xstats' },
      stat('Latest block', '—', 'waiting for the first block'), stat('Block time', '—', 'seconds · target ' + BLOCK_TARGET_S),
      stat('Network rate', '—', 'lottery tickets per second'), stat('Miners', '—', 'none yet'), stat('Jobs', '—', 'none yet'), stat('Supply', '—', 'VRL minted · none yet')));
    wrap.appendChild(h('div', { class: 'x-grid' }, h('div', { class: 'state' }, h('h2', null, 'Nothing to show ', h('span', { class: 'serif', text: 'yet' })),
      h('p', { text: 'The explorer reads a Vearl node and shows exactly what it says. The public testnet has not started, so this page stays empty and switches to live data by itself at the start time.' }))));
    tickCountdown();
    return Promise.resolve(root);
  }
  function emptyChart() { return h('p', { class: 'mute', text: 'Not enough blocks yet.' }); }
  function stat(k, v, s, unit) {
    return h('div', { class: 'xstat' }, h('div', { class: 'k', text: k }), h('div', { class: 'v' }, v, unit ? h('small', { text: unit }) : null), h('div', { class: 's', text: s }));
  }
  function ribbon(ch) {
    var n = ch.heights.length, show = Math.min(n, 60);
    var el = h('div', { class: 'ribbon', role: 'list', 'aria-label': 'Most recent blocks, newest first' });
    var top = n ? ch.heights[n - 1] : -1;
    for (var i = n - 1; i >= n - show; i--) {
      var c = ch.classes[i], ht = ch.heights[i];
      var cls = 'rb ' + CLASS_CSS[c] + (ch.txs[i] ? ' has-tx' : '') + (S.ribbonTop >= 0 && ht > S.ribbonTop ? ' fresh' : '');
      el.appendChild(h('a', { class: cls, href: '#/block/' + ht, role: 'listitem', title: 'Block #' + ht + ' · ' + CLASS_LABEL[c] + ' · ' + ch.txs[i] + ' tx', 'aria-label': 'Block ' + ht }));
    }
    S.ribbonTop = top;
    var legendEl = h('div', { class: 'ribbon-legend' }, CLASS_LABEL.map(function (l, i) { return h('span', null, h('i', { class: CLASS_CSS[i] }), l); }), h('span', { text: '· a dot means the block carries transactions' }));
    return h('div', { class: 'ribbon-wrap' },
      h('div', { class: 'ribbon-head' }, h('div', { class: 'ribbon-title', text: 'Latest blocks · newest on the left' }), legendEl),
      n ? el : h('p', { class: 'mute', text: 'The chain is empty: the first block has not been mined yet.' }),
      n ? h('div', { class: 'ribbon-foot' }, h('span', { text: '#' + top }), h('span', { text: '#' + ch.heights[n - show] })) : null);
  }
  function workTag(work) {
    var i = classIndex(work.type);
    return h('span', { class: 'tag ' + CLASS_TAG[i], text: CLASS_LABEL[i] });
  }
  function latestBlocksPanel(bl) {
    var cols = [
      { k: 'Block', cell: function (b) { return blockLink(b.height, '#' + num(b.height)); }, cls: 'strong' },
      { k: 'Age', cell: function (b) { return h('span', { class: 'ago', text: ago(b.timestamp) }); }, cls: 'm' },
      { k: 'Work', cell: function (b) { return workTag(b.work); } },
      { k: 'Miner', cell: function (b) { return addrLink(b.miner); } },
      { k: 'Txs', cell: function (b) { return String(b.tx_count); }, cls: 'num m' }
    ];
    var body = bl.blocks.length ? table(cols, bl.blocks) : h('div', { class: 'panel-body' }, h('p', { class: 'mute', text: 'No blocks yet.' }));
    return panel('Latest blocks', body, { href: '#/blocks', text: 'All blocks' }, true);
  }
  function txSummary(k) {
    switch (k.type) {
      case 'transfer': return h('span', null, amt(k.amount) + ' VRL → ', addrLink(k.to));
      case 'submit_job': case 'submit_job_auction': case 'program_submit': case 'tiled_submit': case 'level2_submit': case 'inference_submit':
        return h('span', { text: 'escrow ' + amt(k.escrow) + ' VRL' });
      default: return h('span', { class: 'mute', text: k.job ? 'job ' + short(k.job, 8, 4) : '' });
    }
  }
  function txRows(list, compact) {
    if (compact) {
      return table([
        { k: 'Transaction', cell: function (t) { return a('#/tx/' + t.id, short(t.id, 8, 4), 'lnk mono', t.id); } },
        { k: 'Type', cell: function (t) { return h('span', { class: 'tag t-native', text: human(t.kind.type) }); } },
        { k: 'Details', cell: function (t) { return txSummary(t.kind); } },
        { k: 'Block', cell: function (t) { return t._pending ? h('span', { class: 'tag t-op', text: 'pending' }) : blockLink(t._height, '#' + t._height); }, cls: 'm' }
      ], list, { compact: true });
    }
    return table([
      { k: 'Transaction', cell: function (t) { return txLink(t.id); } },
      { k: 'Type', cell: function (t) { return h('span', { class: 'tag t-native', text: human(t.kind.type) }); } },
      { k: 'From', cell: function (t) { return addrLink(t.from); } },
      { k: 'Details', cell: function (t) { return txSummary(t.kind); } },
      { k: 'Status', cell: function (t) { return h('span', { class: 'tag ' + (t._pending ? 't-op' : 't-ok'), text: t._pending ? 'pending' : '#' + t._height }); } }
    ], list);
  }
  function recentTxs(ch, mp) {
    var heights = [];
    for (var i = ch.heights.length - 1; i >= 0 && heights.length < 5; i--) if (ch.txs[i] > 0) heights.push(ch.heights[i]);
    return Promise.all(heights.map(function (x) { return api('/block/' + x).catch(function () { return null; }); })).then(function (blocks) {
      var list = [];
      mp.txs.forEach(function (t) { t._pending = true; list.push(t); });
      blocks.forEach(function (b) { if (b) b.txs.slice().reverse().forEach(function (t) { t._height = b.height; list.push(t); }); });
      list = list.slice(0, 9);
      return list.length ? txRows(list, true) : h('div', { class: 'panel-body' }, h('p', { class: 'mute', text: 'No transactions in the recent blocks.' }));
    });
  }
  function minerSegments(mi) {
    var top = mi.miners.slice(0, 5), rest = mi.blocks - top.reduce(function (s, m) { return s + m.blocks; }, 0);
    var segs = top.map(function (m) { return { value: m.blocks, node: a('#/address/' + m.address, short(m.address, 8, 4), 'lnk mono', m.address) }; });
    if (rest > 0) segs.push({ value: rest, label: 'Others' });
    return segs;
  }
  function jobsMini(jb) {
    if (!jb.jobs.length) return h('div', { class: 'panel-body' }, h('p', { class: 'mute', text: 'No job has been submitted yet.' }));
    return h('ul', { class: 'dn-legend panel-body' }, jb.jobs.map(function (j) {
      return h('li', null, statusTag(j.status), jobLink(j.id), h('span', { class: 'n', text: amt(j.escrow_net) + ' VRL' }));
    }));
  }

  function eventsTable(evs, compact) {
    var cols = [
      { k: 'When', cell: function (e) { return h('span', { class: 'ago', title: utc(e.time), text: ago(e.time) }); }, cls: 'm' },
      { k: 'Event', cell: function (e) { return eventTag(e.what); } },
      { k: 'Job', cell: function (e) { return h('span', null, jobLink(e.job), compact ? null : h('span', { class: 'mute', text: '  ' + (KIND_LABEL[e.kind] || cap(e.kind)) })); } },
      { k: 'By', cell: function (e) { return addrLink(e.by); } },
      { k: 'Block', cell: function (e) { return blockLink(e.height, '#' + num(e.height)); }, cls: 'm' }
    ];
    if (compact) cols.splice(3, 2);
    return table(cols, evs, { compact: !!compact });
  }

  /* ---------------- daily statistics */
  function histogram(dts) {
    var edges = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, Infinity], labels = ['<5', '5', '10', '15', '20', '30', '45', '60', '90', '120+'], c = [];
    edges.slice(0, -1).forEach(function () { c.push(0); });
    dts.forEach(function (d) { for (var i = 0; i < c.length; i++) if (d >= edges[i] && d < edges[i + 1]) { c[i]++; break; } });
    return { counts: c, labels: labels };
  }
  function pageStats(q) {
    var days = [7, 30, 90, 365].indexOf(parseInt(q.days, 10)) >= 0 ? parseInt(q.days, 10) : 30;
    var daily = api('/stats/daily?days=' + days).catch(function (e) { if (e.status === 404 || e.status === 400) return null; throw e; });
    return Promise.all([daily, api('/chart?blocks=400'), api('/status')]).then(function (r) {
      var dl = r[0], ch = r[1], st = r[2], ss = seriesStats(ch);
      document.title = 'Daily statistics · Vearl Explorer';
      var root = h('div', null, head([{ text: 'Explorer', href: '#/' }, { text: 'Stats' }], h('span', null, 'Daily ', h('span', { class: 'serif', text: 'statistics' })), 'What the chain did, day by day (UTC), computed from the blocks themselves.'));
      var wrap = h('div', { class: 'wrap' });
      if (!dl) {
        wrap.appendChild(h('div', { class: 'x-grid' }, h('div', { class: 'state' }, h('h2', null, 'Daily statistics ', h('span', { class: 'serif', text: 'unavailable' })), h('p', { text: 'The node this explorer reads does not have the statistics index yet. The charts below only use the latest blocks.' }))));
      } else {
        var chips = h('div', { class: 'chips', role: 'group', 'aria-label': 'Range' });
        [7, 30, 90, 365].forEach(function (d) { chips.appendChild(h('a', { class: 'chip' + (d === days ? ' on' : ''), href: '#/stats?days=' + d }, 'Last ' + d + ' days')); });
        wrap.appendChild(h('div', { class: 'x-grid' }, chips));
        var rows = dl.days.slice().reverse();
        if (!rows.length) {
          wrap.appendChild(h('div', { class: 'state' }, h('h2', { text: 'No block yet' }), h('p', { text: 'The first block has not been mined.' })));
        } else {
          var tot = { blocks: 0, paid: 0, txs: 0, sub: 0, closed: 0, fraud: 0, vol: 0n, fees: 0n, miners: 0 };
          rows.forEach(function (d) { tot.blocks += d.blocks; tot.paid += d.paid_blocks; tot.txs += d.txs; tot.sub += d.jobs_submitted; tot.closed += d.jobs_closed; tot.fraud += d.fraud_proofs; tot.vol += units(d.transfer_volume); tot.fees += units(d.fees); tot.miners = Math.max(tot.miners, d.miners); });
          var avg = tot.blocks / rows.length;
          wrap.appendChild(h('div', { class: 'xstats' },
            stat('Blocks', num(tot.blocks), 'over ' + rows.length + ' day' + (rows.length > 1 ? 's' : '') + ' · ' + avg.toFixed(0) + ' a day on average'),
            stat('Paid-work blocks', tot.blocks ? (100 * tot.paid / tot.blocks).toFixed(1) : '0', num(tot.paid) + ' blocks mined a client job', '%'),
            stat('Transactions', num(tot.txs), 'fees ' + amt(fromUnits(tot.fees)) + ' VRL'),
            stat('Jobs', num(tot.sub), 'submitted · ' + num(tot.closed) + ' closed'),
            stat('Most miners in a day', num(tot.miners), 'distinct block winners'),
            stat('Transferred', amt(fromUnits(tot.vol)), 'VRL, transfers only')));
          var lab = rows.map(function (d) { return d.day; });
          var many = rows.length > 31;
          var x0 = dayShort(rows[0].day), x1 = dayShort(rows[rows.length - 1].day);
          function ch1(title, vals, o, note) {
            o = o || {};
            o.partialLast = true; o.x0 = x0; o.x1 = x1; o.label = title;
            o.titles = rows.map(function (d, i) { return d.day + ': ' + (o.fmtT ? o.fmtT(vals[i]) : vals[i]); });
            return panel(title, h('div', { class: 'chart' }, barChart(vals, o), h('div', { class: 'legend' }, h('span', { text: note || 'UTC days · the faded bar is the day in progress' }))));
          }
          var cw = h('div', { class: 'x-grid two' });
          cw.appendChild(ch1('Blocks per day', rows.map(function (d) { return d.blocks; }), { ref: 8640, fmt: function (v) { return num(Math.round(v)); }, min: 10 }, 'dashed line: 8,640 blocks, a day at exactly 10 s per block'));
          cw.appendChild(ch1('Active miners per day', rows.map(function (d) { return d.miners; }), { cls: 'bar-b', fmt: function (v) { return String(Math.round(v)); }, min: 3 }, 'distinct addresses that won at least one block'));
          wrap.appendChild(cw);
          var cw2 = h('div', { class: 'x-grid two' });
          cw2.appendChild(ch1('Transactions per day', rows.map(function (d) { return d.txs; }), { cls: 'bar-w', fmt: function (v) { return num(Math.round(v)); }, min: 5 }));
          var share = rows.map(function (d) { return d.blocks ? Math.round(1000 * d.paid_blocks / d.blocks) / 10 : 0; });
          cw2.appendChild(panel('Paid-work share per day', h('div', { class: 'chart' }, share.length > 1 ? lineChart(share, { label: 'Share of blocks that carried paid work', fmt: function (v) { return Math.round(v) + '%'; }, x0: x0, x1: x1 }) : h('p', { class: 'mute', text: 'Needs two days of blocks.' }), h('div', { class: 'legend' }, h('span', { text: 'blocks that mined a client job ÷ all blocks' })))));
          wrap.appendChild(cw2);
          var cw3 = h('div', { class: 'x-grid two' });
          cw3.appendChild(panel('Jobs per day', h('div', { class: 'chart' }, groupChart(rows.map(function (d) { return d.jobs_submitted; }), rows.map(function (d) { return d.jobs_closed; }), { label: 'Jobs submitted and closed per day', min: 3, fmt: function (v) { return String(Math.round(v)); }, x0: x0, x1: x1, titles: rows.map(function (d) { return d.day + ': ' + d.jobs_submitted + ' submitted, ' + d.jobs_closed + ' closed' + (d.fraud_proofs ? ', ' + d.fraud_proofs + ' fraud proofs' : ''); }) }),
            h('div', { class: 'legend' }, h('span', null, h('i', { class: 'k-a' }), 'submitted'), h('span', null, h('i', { class: 'k-b' }), 'closed (accepted, settled, slashed)')))));
          cw3.appendChild(ch1('Transfer volume per day (VRL)', rows.map(function (d) { return parseFloat(d.transfer_volume); }), { cls: 'bar-o', fmt: function (v) { return si(v, ''); }, min: 1, fmtT: function (v) { return amt(v) + ' VRL'; } }, 'only transfers between accounts, no job escrows'));
          wrap.appendChild(cw3);
          var cols = [
            { k: 'Day (UTC)', cell: function (d) { return d.day; }, cls: 'strong' },
            { k: 'Blocks', cell: function (d) { return num(d.blocks); }, cls: 'num m' },
            { k: 'Paid work', cell: function (d) { return num(d.paid_blocks); }, cls: 'num m' },
            { k: 'Miners', cell: function (d) { return String(d.miners); }, cls: 'num m' },
            { k: 'Txs', cell: function (d) { return num(d.txs); }, cls: 'num m' },
            { k: 'Jobs', cell: function (d) { return d.jobs_submitted + ' in · ' + d.jobs_closed + ' out' + (d.fraud_proofs ? ' · ' + d.fraud_proofs + (d.fraud_proofs > 1 ? ' fraud proofs' : ' fraud proof') : ''); }, cls: 'm' },
            { k: 'Volume', cell: function (d) { return amt(d.transfer_volume) + ' VRL'; }, cls: 'num m' },
            { k: 'Fees', cell: function (d) { return amt(d.fees) + ' VRL'; }, cls: 'num m' },
            { k: 'Block rewards', cell: function (d) { return amt(d.base_rewards) + ' VRL'; }, cls: 'num m' }
          ];
          wrap.appendChild(h('div', { class: 'x-grid' }, panel('Day by day', table(cols, dl.days), null, true)));
          if (many) wrap.appendChild(h('p', { class: 'sub-note', text: 'Charts show every day of the range; hover a bar for its value.' }));
        }
      }
      // from the latest blocks: block-time distribution, so a healthy difficulty adjustment is visible
      var hs = histogram(ss.dts);
      var tbs = h('div', { class: 'x-grid two' });
      tbs.appendChild(panel('Block time distribution', ss.dts.length > 3 ? h('div', { class: 'chart' }, barChart(hs.counts, { cls: 'bar-p', labels: hs.labels, label: 'Number of blocks by seconds since the previous one', min: 3, fmt: function (v) { return String(Math.round(v)); }, titles: hs.counts.map(function (c, i) { return c + ' blocks · ' + hs.labels[i] + ' s'; }) }), h('div', { class: 'legend' }, h('span', { text: 'seconds since the previous block, last ' + (ss.dts.length + 1) + ' blocks' }))) : emptyChart()));
      var classes = [0, 0, 0, 0, 0]; ch.classes.forEach(function (c) { classes[c] = (classes[c] || 0) + 1; });
      var segs = CLASS_LABEL.map(function (l, i) { return { value: classes[i], label: l }; }).filter(function (s) { return s.value > 0; });
      tbs.appendChild(panel('What the last blocks carried', segs.length ? h('div', { class: 'donut-wrap' }, donut(segs, String(ch.classes.length), 'blocks'), legend(segs, function (s) { return s.value + ' blocks'; })) : h('div', { class: 'muted-note', text: 'No block yet.' }), null, true));
      wrap.appendChild(tbs);
      wrap.appendChild(h('p', { class: 'sub-note', text: 'Days are UTC calendar days. The node recomputes these totals from the blocks it holds, so two honest nodes show the same numbers. Nothing here is reported by miners.' }));
      root.appendChild(wrap);
      return root;
    });
  }

  /* ---------------- blocks list */
  function pageBlocks(q) {
    var from = q.from != null ? parseInt(q.from, 10) : null;
    var per = 25;
    return Promise.all([api('/status'), api('/blocks?limit=' + per + (from != null && !isNaN(from) ? '&from=' + from : ''))]).then(function (r) {
      var st = r[0], bl = r[1], blocks = bl.blocks;
      document.title = 'Blocks · Vearl Explorer';
      var root = h('div', null, head([{ text: 'Explorer', href: '#/' }, { text: 'Blocks' }], h('span', null, 'All ', h('span', { class: 'serif', text: 'blocks' })), 'Newest first. Every block shows the work it carried and who mined it.'));
      var wrap = h('div', { class: 'wrap' });
      if (!blocks.length) { wrap.appendChild(h('div', { class: 'state' }, h('h2', { text: 'No blocks yet' }), h('p', { text: 'The chain is empty.' }))); root.appendChild(wrap); return root; }
      var cols = [
        { k: 'Block', cell: function (b) { return blockLink(b.height, '#' + num(b.height)); }, cls: 'strong' },
        { k: 'Age', cell: function (b) { return h('span', { class: 'ago', title: utc(b.timestamp), text: ago(b.timestamp) }); }, cls: 'm' },
        { k: 'Work', cell: function (b) { return workTag(b.work); } },
        { k: 'Miner', cell: function (b) { return addrLink(b.miner); } },
        { k: 'Txs', cell: function (b) { return String(b.tx_count); }, cls: 'num m' },
        { k: 'Reward', cell: function (b) { return amt(b.reward) + ' VRL'; }, cls: 'num m' },
        { k: 'Hash', cell: function (b) { return blockLink(b.hash, short(b.hash, 10, 6)); }, cls: 'm' }
      ];
      var first = blocks[0].height, last = blocks[blocks.length - 1].height, top = latestBlock(st);
      var newer = first < top ? '#/blocks' + (first + per < top ? '?from=' + (first + per) : '') : null;
      var older = last > 0 ? '#/blocks?from=' + (last - 1) : null;
      wrap.appendChild(h('div', { class: 'bighead' }, h('div', { class: 'mute', text: 'Showing #' + num(last) + ' to #' + num(first) + ' of ' + num(st.height) + ' blocks' }), pager(newer, older)));
      wrap.appendChild(h('div', { class: 'x-grid' }, h('section', { class: 'panel flush' }, h('div', { class: 'panel-body' }, table(cols, blocks)))));
      wrap.appendChild(h('div', { class: 'bighead' }, h('span'), pager(newer, older)));
      root.appendChild(wrap);
      return root;
    });
  }

  /* ---------------- block detail */
  function pageBlock(id) {
    return Promise.all([api('/status'), api('/block/' + encodeURIComponent(id))]).then(function (r) {
      var st = r[0], b = r[1], top = latestBlock(st);
      document.title = 'Block #' + b.height + ' · Vearl Explorer';
      var ci = classIndex(b.work.type);
      var root = h('div', null, head([{ text: 'Explorer', href: '#/' }, { text: 'Blocks', href: '#/blocks' }, { text: '#' + num(b.height) }],
        h('span', null, 'Block ', h('span', { class: 'serif', text: '#' + num(b.height) }))));
      var wrap = h('div', { class: 'wrap' });
      var nav = h('div', { class: 'pager' },
        b.height > 0 ? h('a', { class: 'btn', href: '#/block/' + (b.height - 1) }, '← Previous') : h('span', { class: 'btn', 'aria-disabled': 'true' }, '← Previous'),
        b.height < top ? h('a', { class: 'btn', href: '#/block/' + (b.height + 1) }, 'Next →') : h('span', { class: 'btn', 'aria-disabled': 'true' }, 'Next →'));
      wrap.appendChild(h('div', { class: 'bighead' }, h('div', null, workTag(b.work), ' ', h('span', { class: 'mute', text: ' ' + (top - b.height) + ' confirmations' })), nav));
      var workRow = h('span', null, workTag(b.work), h('span', { class: 'mute', text: '  ' + CLASS_DESC[b.work.type] }));
      var jobRef = b.work.job ? jobLink(b.work.job, true) : null;
      var rows = [
        ['Hash', withCopy(h('span', { class: 'mono wrap-any', text: b.hash }), b.hash), 'mono'],
        ['Previous block', b.height > 0 ? blockLink(b.prev, b.prev) : h('span', { class: 'mute', text: 'genesis' }), 'mono'],
        ['Time', h('span', null, utc(b.timestamp), h('span', { class: 'mute', text: '  ·  ' + ago(b.timestamp) }))],
        ['Miner', withCopy(addrLink(b.miner, true), b.miner)],
        ['Work', workRow],
        ['Job', jobRef],
        ['Layer / operation', b.work.layer != null ? String(b.work.layer) : b.work.op != null ? String(b.work.op) : null],
        ['Product block', b.work.block ? b.work.block.join(' · ') : null],
        ['Winning tile', 'row ' + b.tile[0] + ', column ' + b.tile[1], 'mono'],
        ['Jackpot', h('span', { class: 'mono wrap-any', text: b.jackpot })],
        ['Target', h('span', { class: 'mono', text: b.target })],
        ['Block reward', amt(b.reward) + ' VRL'],
        ['Transactions', String(b.tx_count)]
      ];
      wrap.appendChild(h('div', { class: 'x-grid' }, h('section', { class: 'panel' }, h('div', { class: 'panel-body' }, kv(rows)))));
      if (b.txs && b.txs.length) {
        b.txs.forEach(function (t) { t._height = b.height; });
        wrap.appendChild(h('div', { class: 'x-grid' }, panel('Transactions (' + b.txs.length + ')', txRows(b.txs), null, true)));
      }
      wrap.appendChild(h('p', { class: 'sub-note', text: 'The jackpot is the hash of the winning 16 × 16 tile of the noised matrix product; it is below the target, which is what makes the block valid. Any node recomputes that one tile to check it.' }));
      root.appendChild(wrap);
      return root;
    });
  }

  /* ---------------- transaction detail */
  function renderValue(key, v) {
    if (v == null) return '';
    if (key === 'to' || key === 'client' || key === 'worker') return addrLink(String(v), true);
    if (key === 'job') return jobLink(String(v), true);
    if (key === 'amount' || key === 'escrow' || key === 'price') return amt(v) + ' VRL';
    if (Array.isArray(v)) return v.join(' · ');
    return String(v);
  }
  function pageTx(id) {
    return api('/tx/' + encodeURIComponent(id)).then(function (t) {
      document.title = 'Transaction ' + short(t.id, 8, 6) + ' · Vearl Explorer';
      var pending = t.status === 'pending';
      var root = h('div', null, head([{ text: 'Explorer', href: '#/' }, { text: 'Transaction' }], h('span', null, 'Transaction ', h('span', { class: 'serif', text: human(t.kind.type) }))));
      var wrap = h('div', { class: 'wrap' });
      wrap.appendChild(h('div', { class: 'bighead' }, h('span', { class: 'tag ' + (pending ? 't-op' : 't-ok'), text: pending ? 'pending in the mempool' : 'confirmed · ' + num(t.confirmations) + ' confirmations' })));
      var rows = [
        ['Transaction id', withCopy(h('span', { class: 'mono wrap-any', text: t.id }), t.id), 'mono'],
        ['Status', pending ? 'Waiting for a miner to include it in a block' : 'Included in block'],
        ['Block', pending ? null : blockLink(t.height, '#' + num(t.height))],
        ['Type', h('span', { class: 'tag t-native', text: human(t.kind.type) })],
        ['From', withCopy(addrLink(t.from, true), t.from)]
      ];
      Object.keys(t.kind).forEach(function (k) { if (k !== 'type') rows.push([cap(k), renderValue(k, t.kind[k]), typeof t.kind[k] === 'string' && t.kind[k].length > 40 ? 'mono' : null]); });
      rows.push(['Fee', amt(t.fee) + ' VRL'], ['Account nonce', String(t.nonce)]);
      wrap.appendChild(h('div', { class: 'x-grid' }, h('section', { class: 'panel' }, h('div', { class: 'panel-body' }, kv(rows)))));
      if (pending) S.refresh = function () { route(true); };
      root.appendChild(wrap);
      return root;
    });
  }

  /* ---------------- address */
  function pageAddress(addr) {
    return Promise.all([api('/account/' + encodeURIComponent(addr)), api('/address/' + encodeURIComponent(addr) + '/txs?limit=50')]).then(function (r) {
      var ac = r[0], hx = r[1];
      document.title = short(ac.address, 10, 6) + ' · Vearl Explorer';
      var root = h('div', null, head([{ text: 'Explorer', href: '#/' }, { text: 'Address' }], h('span', null, 'Account ', h('span', { class: 'serif', text: short(ac.address, 8, 6) }))));
      var wrap = h('div', { class: 'wrap' });
      wrap.appendChild(h('div', { class: 'bighead' }, h('div', { class: 'bignum' }, amt(ac.balance), h('small', { text: 'VRL' })), h('div', { class: 'mute mono wrap-any', text: '' })));
      var rep = ac.reputation;
      var rows = [
        ['Address', withCopy(h('span', { class: 'mono wrap-any', text: ac.address }), ac.address), 'mono'],
        ['Balance', amt(ac.balance) + ' VRL'],
        ['Transactions sent', num(ac.nonce)],
        ['Pending', ac.next_nonce > ac.nonce ? String(ac.next_nonce - ac.nonce) + ' in the mempool' : 'none'],
        ['Compute reputation', rep.done + ' jobs done · ' + rep.slashed + ' slashed · ' + rep.expired + ' expired']
      ];
      wrap.appendChild(h('div', { class: 'x-grid' }, h('section', { class: 'panel' }, h('div', { class: 'panel-body' }, kv(rows)))));
      var list = hx.txs;
      var cols = [
        { k: 'Transaction', cell: function (t) { return txLink(t.id); } },
        { k: 'Direction', cell: function (t) { return h('span', { class: 'tag ' + (t.direction === 'in' ? 't-in' : 't-out'), text: t.direction === 'in' ? 'received' : 'sent' }); } },
        { k: 'Counterparty', cell: function (t) { return t.direction === 'in' ? addrLink(t.from) : (t.kind.to ? addrLink(t.kind.to) : h('span', { class: 'mute', text: human(t.kind.type) })); } },
        { k: 'Amount', cell: function (t) { return t.kind.amount ? (t.direction === 'in' ? '+' : '−') + amt(t.kind.amount) + ' VRL' : ''; }, cls: 'num m' },
        { k: 'Status', cell: function (t) { return t.status === 'pending' ? h('span', { class: 'tag t-op', text: 'pending' }) : blockLink(t.height, '#' + num(t.height)); }, cls: 'm' }
      ];
      wrap.appendChild(h('div', { class: 'x-grid' }, panel('Transactions (latest ' + list.length + ')', list.length ? table(cols, list) : h('div', { class: 'panel-body' }, h('p', { class: 'mute', text: 'No transaction involves this address yet.' })), null, true)));
      wrap.appendChild(h('p', { class: 'sub-note', text: 'The history lists the transactions this address sent and the transfers it received, newest first, up to 50.' }));
      root.appendChild(wrap);
      return root;
    });
  }

  /* ---------------- jobs */
  function pageJobs(q) {
    var offset = parseInt(q.offset || '0', 10) || 0, per = 25;
    var qs = '?limit=' + per + '&offset=' + offset + (q.kind ? '&kind=' + encodeURIComponent(q.kind) : '') + (q.status ? '&status=' + encodeURIComponent(q.status) : '');
    return Promise.all([api('/jobs' + qs), api('/jobs?limit=1')]).then(function (r) {
      var jb = r[0], all = r[1];
      document.title = 'Jobs · Vearl Explorer';
      var root = h('div', null, head([{ text: 'Explorer', href: '#/' }, { text: 'Jobs' }], h('span', null, 'Compute ', h('span', { class: 'serif', text: 'jobs' })), 'Clients lock an escrow, a worker computes, the result is verified, then the worker is paid or loses its bond.'));
      var wrap = h('div', { class: 'wrap' });
      function link(k, s) { return '#/jobs' + (k || s ? '?' + (k ? 'kind=' + k : '') + (k && s ? '&' : '') + (s ? 'status=' + s : '') : ''); }
      var kc = h('div', { class: 'chips', role: 'group', 'aria-label': 'Filter by kind' }, h('a', { class: 'chip' + (!q.kind ? ' on' : ''), href: link(null, q.status) }, 'All kinds ', h('b', { text: String(all.total) })));
      Object.keys(all.counts).sort().forEach(function (k) { kc.appendChild(h('a', { class: 'chip' + (q.kind === k ? ' on' : ''), href: link(k, q.status) }, KIND_LABEL[k] || cap(k), ' ', h('b', { text: String(all.counts[k]) }))); });
      var sc = h('div', { class: 'chips', role: 'group', 'aria-label': 'Filter by status' }, h('a', { class: 'chip' + (!q.status ? ' on' : ''), href: link(q.kind, null) }, 'Any status'));
      Object.keys(all.by_status).sort().forEach(function (s) { sc.appendChild(h('a', { class: 'chip' + (q.status === s ? ' on' : ''), href: link(q.kind, s) }, s, ' ', h('b', { text: String(all.by_status[s]) }))); });
      wrap.appendChild(kc); wrap.appendChild(sc);
      if (!jb.jobs.length) {
        wrap.appendChild(h('div', { class: 'state' }, h('h2', null, 'No job ', h('span', { class: 'serif', text: 'matches' })), h('p', { text: all.total ? 'Try another filter.' : 'Nobody has submitted a compute job on this network yet.' })));
      } else {
        var cols = [
          { k: 'Job', cell: function (j) { return jobLink(j.id); } },
          { k: 'Kind', cell: function (j) { return h('span', { text: (KIND_LABEL[j.kind] || cap(j.kind)) + (j.kernel ? ' · ' + (KERNEL_LABEL[j.kernel] || j.kernel) : '') }); } },
          { k: 'Status', cell: function (j) { return statusTag(j.status); } },
          { k: 'Client', cell: function (j) { return addrLink(j.client); } },
          { k: 'Worker', cell: function (j) { return j.worker ? addrLink(j.worker) : h('span', { class: 'dim', text: '—' }); } },
          { k: 'Escrow', cell: function (j) { return amt(j.escrow_net) + ' VRL'; }, cls: 'num m' },
          { k: 'Created', cell: function (j) { return j.created_time ? h('span', { class: 'ago', title: utc(j.created_time) + ' · block ' + num(j.created_height), text: ago(j.created_time) }) : h('span', { class: 'dim', text: '—' }); }, cls: 'm' },
          { k: 'Deadline', cell: function (j) { return 'block ' + num(j.deadline_height); }, cls: 'm' }
        ];
        var newer = offset > 0 ? '#/jobs?' + (q.kind ? 'kind=' + q.kind + '&' : '') + (q.status ? 'status=' + q.status + '&' : '') + 'offset=' + Math.max(0, offset - per) : null;
        var older = offset + per < jb.total ? '#/jobs?' + (q.kind ? 'kind=' + q.kind + '&' : '') + (q.status ? 'status=' + q.status + '&' : '') + 'offset=' + (offset + per) : null;
        wrap.appendChild(h('div', { class: 'x-grid' }, h('section', { class: 'panel flush' }, h('div', { class: 'panel-body' }, table(cols, jb.jobs)))));
        wrap.appendChild(h('div', { class: 'bighead' }, h('div', { class: 'mute', text: num(jb.total) + ' jobs' }), pager(newer, older)));
      }
      root.appendChild(wrap);
      return root;
    });
  }

  /* ---------------- job detail */
  function pageJob(id) {
    return api('/job/' + encodeURIComponent(id)).then(function (j) {
      document.title = 'Job ' + short(j.id, 8, 6) + ' · Vearl Explorer';
      var root = h('div', null, head([{ text: 'Explorer', href: '#/' }, { text: 'Jobs', href: '#/jobs' }, { text: short(j.id, 8, 6) }], h('span', null, (KIND_LABEL[j.kind] || cap(j.kind)), ' ', h('span', { class: 'serif', text: 'job' }))));
      var wrap = h('div', { class: 'wrap' });
      var order = { open: 1, claimed: 2, working: 2, posted: 3, complete: 3, settled: 4, slashed: 4, expired: 4 };
      var idx = order[j.status] != null ? order[j.status] : 1;
      var steps = [['Submitted', 'The client locked the escrow.'], ['Claimed', 'A worker took the job and locked a bond.'], ['Result posted', 'The worker committed to the result; the challenge window is open.'], [j.status === 'slashed' ? 'Slashed' : j.status === 'expired' ? 'Expired' : 'Settled', j.status === 'slashed' ? 'A fraud proof was accepted: the bond is forfeited.' : j.status === 'expired' ? 'The deadline passed.' : 'The result was accepted and the worker was paid.']];
      var hist = j.history || [];
      function firstOf(names) { for (var i = 0; i < hist.length; i++) if (names.indexOf(hist[i].what) >= 0) return hist[i]; return null; }
      function lastOf(names) { for (var i = hist.length - 1; i >= 0; i--) if (names.indexOf(hist[i].what) >= 0) return hist[i]; return null; }
      var stamps = [firstOf(['submitted']), firstOf(['claimed']), firstOf(['posted']), lastOf(['accepted', 'settled', 'fraud_proof', 'slash_unserved'])];
      var life = h('ul', { class: 'life' }, steps.map(function (s, i) {
        var cls = i < idx ? 'done' : i === idx ? 'now' : '';
        if (i === 3 && idx === 4 && (j.status === 'slashed' || j.status === 'expired')) cls = 'done end-bad';
        var at = stamps[i];
        return h('li', { class: cls }, h('div', { class: 'n', text: '0' + (i + 1) }), h('h3', { text: s[0] }), h('p', { text: s[1] }),
          at ? h('div', { class: 'when', title: utc(at.time), text: 'block ' + num(at.height) + ' · ' + ago(at.time) }) : null);
      }));
      var made = firstOf(['submitted']), closed = lastOf(['accepted', 'settled', 'fraud_proof', 'slash_unserved']);
      var worked = hist.filter(function (e) { return e.what === 'work_block'; });
      wrap.appendChild(h('div', { class: 'bighead' }, h('div', null, statusTag(j.status), h('span', { class: 'mute', text: '  ' + (KIND_LABEL[j.kind] || cap(j.kind)) }))));
      wrap.appendChild(h('div', { class: 'x-grid' }, life));
      var rows = [
        ['Job id', withCopy(h('span', { class: 'mono wrap-any', text: j.id }), j.id), 'mono'],
        ['Created', made ? h('span', null, utc(made.time), h('span', { class: 'mute', text: '  ·  block ' + num(made.height) + '  ·  ' + ago(made.time) })) : null],
        ['Duration', made && closed ? dur(closed.time - made.time) + ', from submission to the end' : null],
        ['Kind', KIND_LABEL[j.kind] || cap(j.kind)],
        ['Kernel', j.kernel ? (KERNEL_LABEL[j.kernel] || j.kernel) : null],
        ['Chunks', j.chunks != null ? num(j.chunks) : null],
        ['Client', addrLink(j.client, true)],
        ['Worker', j.worker ? addrLink(j.worker, true) : h('span', { class: 'mute', text: j.status === 'open' ? 'not claimed yet' : 'not kept by the chain once the job is closed' })],
        ['Escrow (net)', amt(j.escrow_net) + ' VRL'],
        ['Deadline', 'block ' + num(j.deadline_height)],
        ['Layers', j.layers_total != null ? j.layers_posted + ' of ' + j.layers_total + ' posted' : null],
        ['Operations', j.operations != null ? (j.operations_posted != null ? j.operations_posted + ' of ' + j.operations + ' posted' : String(j.operations)) + (j.gemms != null ? ' · ' + j.gemms + ' matrix products' : '') : null],
        ['Block products', j.blocks ? j.blocks.join(' × ') + ' blocks · ' + j.products + ' products' : null],
        ['Best bid', j.best_bid ? amt(j.best_bid.price) + ' VRL by ' + short(j.best_bid.worker, 10, 6) : null],
        ['Data requests', j.pending_data_requests != null ? String(j.pending_data_requests) + ' pending' : null],
        ['Entries served', j.served_entries != null ? String(j.served_entries) : null],
        ['Blocks mined for it', worked.length ? h('span', null, num(worked.length) + ' · first ', blockLink(worked[0].height, '#' + num(worked[0].height)), ' · last ', blockLink(worked[worked.length - 1].height, '#' + num(worked[worked.length - 1].height))) : null]
      ];
      wrap.appendChild(h('div', { class: 'x-grid' }, h('section', { class: 'panel' }, h('div', { class: 'panel-body' }, kv(rows)))));
      var steps2 = hist.filter(function (e) { return e.what !== 'work_block'; });
      if (steps2.length) {
        var hc = [
          { k: 'When', cell: function (e) { return h('span', { class: 'ago', title: utc(e.time), text: utc(e.time).slice(5, 19) + ' · ' + ago(e.time) }); }, cls: 'm' },
          { k: 'Event', cell: function (e) { return eventTag(e.what); } },
          { k: 'By', cell: function (e) { return addrLink(e.by); } },
          { k: 'Block', cell: function (e) { return blockLink(e.height, '#' + num(e.height)); }, cls: 'm' },
          { k: 'Transaction', cell: function (e) { return e.tx ? txLink(e.tx) : h('span', { class: 'dim', text: '—' }); }, cls: 'm' }
        ];
        wrap.appendChild(h('div', { class: 'x-grid' }, panel('History (' + steps2.length + ' event' + (steps2.length > 1 ? 's' : '') + ')', table(hc, steps2), null, true)));
      }
      if (j.kind === 'level2') wrap.appendChild(h('p', { class: 'sub-note', text: 'Level 2 jobs are verified by sampling: the client recomputes a few random chunks, and anyone can prove a wrong chunk on chain with a fraud proof of about a kilobyte. This is a weaker guarantee than the exact check of the matrix jobs, and the whitepaper explains why.' }));
      if (j.status !== 'settled' && j.status !== 'slashed' && j.status !== 'expired') S.refresh = function () { route(true); };
      root.appendChild(wrap);
      return root;
    });
  }

  /* ---------------- miners */
  function pageMiners(q) {
    var win = parseInt(q.window || '1000', 10) || 1000;
    return api('/miners?window=' + win).then(function (mi) {
      document.title = 'Miners · Vearl Explorer';
      var root = h('div', null, head([{ text: 'Explorer', href: '#/' }, { text: 'Miners' }], h('span', null, 'Who ', h('span', { class: 'serif', text: 'mines' })), 'Share of the most recent blocks, by miner. The share is blocks found, not hardware: nothing on chain reveals the machine.'));
      var wrap = h('div', { class: 'wrap' });
      var chips = h('div', { class: 'chips', role: 'group', 'aria-label': 'Window' });
      [100, 1000, 5000, 20000].forEach(function (w) { chips.appendChild(h('a', { class: 'chip' + (w === win ? ' on' : ''), href: '#/miners?window=' + w }, 'Last ' + num(w) + ' blocks')); });
      wrap.appendChild(chips);
      if (!mi.miners.length) { wrap.appendChild(h('div', { class: 'state' }, h('h2', { text: 'No miners yet' }), h('p', { text: 'No block has been mined in this window.' }))); root.appendChild(wrap); return root; }
      var segs = minerSegments(mi);
      wrap.appendChild(h('div', { class: 'x-grid two' },
        panel('Share of blocks', h('div', { class: 'donut-wrap' }, donut(segs, String(mi.distinct_miners), 'miners'), legend(segs, function (s) { return s.value + ' blocks'; })), null, true),
        panel('Window', kv([['Blocks counted', num(mi.blocks)], ['Distinct miners', num(mi.distinct_miners)], ['Paid-work blocks', num(mi.miners.reduce(function (s, m) { return s + m.useful_blocks; }, 0))]]), null, true)));
      var cols = [
        { k: '#', cell: function (m) { return String(mi.miners.indexOf(m) + 1); }, cls: 'rank' },
        { k: 'Miner', cell: function (m) { return addrLink(m.address); } },
        { k: 'Blocks', cell: function (m) { return num(m.blocks); }, cls: 'num m' },
        { k: 'Share', cell: function (m) { return shareBar(m.share); } },
        { k: 'Paid work', cell: function (m) { return num(m.useful_blocks); }, cls: 'num m' },
        { k: 'Rewards', cell: function (m) { return amt(m.reward) + ' VRL'; }, cls: 'num m' },
        { k: 'Jobs', cell: function (m) { return m.reputation.done + ' done · ' + m.reputation.slashed + ' slashed'; }, cls: 'm' },
        { k: 'Last block', cell: function (m) { return blockLink(m.last_height, '#' + num(m.last_height)); }, cls: 'm' }
      ];
      wrap.appendChild(h('div', { class: 'x-grid' }, h('section', { class: 'panel flush' }, h('div', { class: 'panel-body' }, table(cols, mi.miners)))));
      wrap.appendChild(h('p', { class: 'sub-note', text: 'Rewards are the base block reward at the height of each block, summed over the window. Paid-work blocks are blocks that mined the matrices of a client job.' }));
      root.appendChild(wrap);
      return root;
    });
  }
  function shareBar(f) {
    var i = h('i'); setTimeout(function () { i.style.width = Math.max(1, Math.min(100, f * 100)) + '%'; }, 40);
    return h('div', { class: 'sbar' }, h('div', { class: 'tr' }, i), h('span', { class: 'pc', text: (f * 100).toFixed(1) + '%' }));
  }

  /* ---------------- accounts */
  function pageAccounts() {
    return Promise.all([api('/accounts?limit=100'), api('/status')]).then(function (r) {
      var ac = r[0], st = r[1];
      document.title = 'Accounts · Vearl Explorer';
      var root = h('div', null, head([{ text: 'Explorer', href: '#/' }, { text: 'Accounts' }], h('span', null, 'Largest ', h('span', { class: 'serif', text: 'balances' })), 'Testnet balances have no value. This list is here to check that supply adds up.'));
      var wrap = h('div', { class: 'wrap' });
      var circ = units(st.supply.genesis) + units(st.supply.minted) - units(st.supply.burned);
      wrap.appendChild(h('div', { class: 'xstats' },
        stat('Accounts with a balance', num(ac.holders), 'out of ' + num(st.accounts) + ' live accounts'),
        stat('Held by accounts', amt(ac.held), 'VRL'),
        stat('Circulating supply', amt(fromUnits(circ)), 'VRL · genesis + minted − burned'),
        stat('Minted', amt(st.supply.minted), 'VRL'),
        stat('Burned', amt(st.supply.burned), 'VRL'),
        stat('Supply check', st.supply_invariant ? 'OK' : 'FAIL', st.supply_invariant ? 'the node verified it' : 'the node reports a mismatch')));
      var cols = [
        { k: '#', cell: function (x) { return String(ac.accounts.indexOf(x) + 1); }, cls: 'rank' },
        { k: 'Address', cell: function (x) { return addrLink(x.address, false); } },
        { k: 'Balance', cell: function (x) { return amt(x.balance) + ' VRL'; }, cls: 'num strong' },
        { k: 'Share of held', cell: function (x) { return shareBar(x.share); } },
        { k: 'Transactions sent', cell: function (x) { return num(x.nonce); }, cls: 'num m' }
      ];
      wrap.appendChild(h('div', { class: 'x-grid' }, h('section', { class: 'panel flush' }, h('div', { class: 'panel-body' }, ac.accounts.length ? table(cols, ac.accounts) : h('p', { class: 'mute', text: 'No account holds a balance.' })))));
      root.appendChild(wrap);
      return root;
    });
  }

  /* ---------------- mempool */
  function pageMempool() {
    return api('/mempool').then(function (mp) {
      document.title = 'Pending transactions · Vearl Explorer';
      var root = h('div', null, head([{ text: 'Explorer', href: '#/' }, { text: 'Mempool' }], h('span', null, 'Pending ', h('span', { class: 'serif', text: 'transactions' })), 'Signed transactions the node has seen and no block includes yet.'));
      var wrap = h('div', { class: 'wrap' });
      mp.txs.forEach(function (t) { t._pending = true; });
      wrap.appendChild(h('div', { class: 'x-grid' }, mp.txs.length ? h('section', { class: 'panel flush' }, h('div', { class: 'panel-body' }, txRows(mp.txs))) : h('div', { class: 'state' }, h('h2', null, 'The mempool is ', h('span', { class: 'serif', text: 'empty' })), h('p', { text: 'Every transaction the node knows about is already in a block.' }))));
      if (mp.count > mp.txs.length) wrap.appendChild(h('p', { class: 'sub-note', text: 'Showing the first ' + mp.txs.length + ' of ' + mp.count + '.' }));
      S.refresh = function () { route(true); };
      root.appendChild(wrap);
      return root;
    });
  }

  /* ---------------- network */
  var ENDPOINTS = [
    ['/api/status', 'Height, tip, supply, work parameters, counters'],
    ['/api/blocks?limit=25&from=H', 'Block summaries, newest first'],
    ['/api/block/{height|hash}', 'One block with its transactions'],
    ['/api/tx/{id}', 'A transaction, confirmed or pending'],
    ['/api/account/{address}', 'Balance, nonce, compute reputation'],
    ['/api/address/{address}/txs', 'Transfers sent or received'],
    ['/api/jobs?kind=&status=', 'Compute jobs of every kind'],
    ['/api/job/{id}', 'One compute job'],
    ['/api/miners?window=1000', 'Who mined the last blocks'],
    ['/api/accounts?limit=100', 'Largest balances'],
    ['/api/chart?blocks=120', 'Per-block series for charts'],
    ['/api/stats/daily?days=30', 'Totals per UTC day'],
    ['/api/events?what=&limit=50', 'Latest job events, filterable'],
    ['/api/mempool', 'Pending transactions'],
    ['/api/search?q=', 'Resolve a height, hash, id or address'],
    ['/api/wait?height=H', 'Long poll: answers when a new block arrives'],
    ['POST /api/tx', 'Relay a transaction already signed by a wallet']
  ];
  function pageNetwork() {
    return Promise.all([api('/status'), api('/chart?blocks=120')]).then(function (r) {
      var st = r[0], ch = r[1], ss = seriesStats(ch);
      document.title = 'Network · Vearl Explorer';
      var root = h('div', null, head([{ text: 'Explorer', href: '#/' }, { text: 'Network' }], h('span', null, 'The ', h('span', { class: 'serif', text: 'network' }), ' and its API'), 'What this node knows about the chain it follows.', false));
      var wrap = h('div', { class: 'wrap' });
      var circ = units(st.supply.genesis) + units(st.supply.minted) - units(st.supply.burned);
      var check = h('span', { class: 'check ' + (st.supply_invariant ? 'ok' : 'bad'), text: st.supply_invariant ? '✓ supply invariant holds' : '✗ supply invariant broken' });
      wrap.appendChild(h('div', { class: 'x-grid two' },
        panel('Chain', kv([
          ['Genesis hash', withCopy(h('span', { class: 'mono wrap-any', text: st.genesis_hash }), st.genesis_hash), 'mono'],
          ['Latest block', st.height ? blockLink(latestBlock(st), '#' + num(latestBlock(st))) : '—'],
          ['Tip hash', h('span', { class: 'mono wrap-any', text: st.tip }), 'mono'],
          ['Total work', st.total_work, 'mono'],
          ['Difficulty target', st.target, 'mono'],
          ['Difficulty rule', st.retarget === 'Asert' ? 'ASERT (aserti3-2d)' : st.retarget],
          ['Block time', ss.blockTime ? ss.blockTime.toFixed(1) + ' s measured, target ' + BLOCK_TARGET_S + ' s' : 'target ' + BLOCK_TARGET_S + ' s'],
          ['Block reward now', amt(st.reward) + ' VRL'],
          ['Node version', st.version]
        ]), null, true),
        panel('Work and supply', kv([
          ['Matrix size n', String(st.params.n)],
          ['Tile', st.params.tile + ' × ' + st.params.tile],
          ['Noise rank', String(st.params.rank)],
          ['Fold k', String(st.params.fold_k)],
          ['Genesis supply', amt(st.supply.genesis) + ' VRL'],
          ['Minted', amt(st.supply.minted) + ' VRL'],
          ['Burned', amt(st.supply.burned) + ' VRL'],
          ['Circulating', amt(fromUnits(circ)) + ' VRL'],
          ['Check', check]
        ]), null, true)));
      wrap.appendChild(h('div', { class: 'x-grid two' },
        panel('This node', kv([
          ['Peers', String(st.peers)],
          ['Mempool', String(st.mempool) + ' transactions'],
          ['Live accounts', num(st.accounts)],
          ['Matrix jobs', num(st.jobs)],
          ['Inference jobs', num(st.inference_jobs)],
          ['Tensor programs', num(st.tensor_programs)],
          ['Tiled jobs', num(st.tiled_jobs)],
          ['Level 2 jobs', num(st.level2_jobs)]
        ]), null, true),
        panel('Read it yourself', h('ul', { class: 'api-list' }, ENDPOINTS.map(function (e) {
          var plain = e[0].indexOf('{') < 0 && e[0].indexOf('=') < 0 && e[0].slice(0, 4) !== 'POST';
          return h('li', null, plain ? a(e[0], e[0], 'lnk plain') : h('code', { text: e[0] }), h('span', { text: e[1] }));
        })), null, true)));
      wrap.appendChild(h('p', { class: 'sub-note', text: 'The API is read-only. Transactions are signed by a wallet on your machine: neither this explorer nor the node ever sees a private key.' }));
      root.appendChild(wrap);
      return root;
    });
  }

  /* ------------------------------------------------------------------ router */
  function parseHash() {
    var raw = location.hash.replace(/^#/, '') || '/';
    var qi = raw.indexOf('?'), path = qi < 0 ? raw : raw.slice(0, qi), q = {};
    if (qi >= 0) raw.slice(qi + 1).split('&').forEach(function (kv) { var p = kv.split('='); if (p[0]) q[decodeURIComponent(p[0])] = decodeURIComponent(p[1] || ''); });
    var seg = path.split('/').filter(Boolean).map(decodeURIComponent);
    return { seg: seg, q: q };
  }
  function nodeError(e, what) {
    if (e && (e.status === 404 || e.status === 400)) {
      return stateCard('Nothing ' + 'found', e.message + (what ? '' : ''), false, false);
    }
    return h('div', { class: 'wrap' }, h('div', { class: 'state bad' },
      h('h2', null, 'No node ', h('span', { class: 'serif', text: 'reachable' })),
      h('p', { text: 'This explorer could not get an answer from a Vearl node. The public testnet is planned for 23 October 2026; until a public node is online there is nothing to show here.' }),
      h('button', { class: 'btn', type: 'button', onclick: function () { route(); } }, 'Try again')));
  }
  var navSeq = 0;
  function route(soft) {
    var my = ++navSeq;
    var p = parseHash(), s0 = p.seg[0] || '', page;
    if (!soft) S.refresh = null;
    document.querySelectorAll('[data-nav]').forEach(function (el) {
      var k = el.getAttribute('data-nav');
      var on = (k === 'home' && !s0) || (k === 'blocks' && (s0 === 'blocks' || s0 === 'block')) || (k === 'jobs' && (s0 === 'jobs' || s0 === 'job')) || k === s0 || (k === 'accounts' && s0 === 'address');
      el.classList.toggle('active', on);
    });
    var live = false;
    if (preLaunch()) page = pagePrelaunch();
    else if (!s0) { page = pageHome(); live = true; }
    else if (s0 === 'blocks') { page = pageBlocks(p.q); live = !p.q.from; }
    else if (s0 === 'block' && p.seg[1] != null) page = pageBlock(p.seg[1]);
    else if (s0 === 'tx' && p.seg[1]) page = pageTx(p.seg[1]);
    else if (s0 === 'address' && p.seg[1]) page = pageAddress(p.seg[1]);
    else if (s0 === 'jobs') { page = pageJobs(p.q); live = true; }
    else if (s0 === 'job' && p.seg[1]) page = pageJob(p.seg[1]);
    else if (s0 === 'miners') { page = pageMiners(p.q); live = true; }
    else if (s0 === 'accounts') { page = pageAccounts(); live = true; }
    else if (s0 === 'stats') { page = pageStats(p.q); live = true; }
    else if (s0 === 'mempool') page = pageMempool();
    else if (s0 === 'network') { page = pageNetwork(); live = true; }
    else page = Promise.resolve(stateCard('Page not found', 'This address does not exist in the explorer.', false, false));
    if (!soft) { app.replaceChildren(loading()); window.scrollTo(0, 0); }
    return page.then(function (node) {
      if (my !== navSeq) return;
      app.replaceChildren(node);
      if (live) S.refresh = function () { route(true); };
      var fresh = document.getElementById('q'); if (fresh && !soft && !location.hash.slice(2)) { /* the home search keeps its place */ }
    }).catch(function (e) {
      if (my !== navSeq) return;
      if (soft) return;
      app.replaceChildren(nodeError(e));
    });
  }

  /* ------------------------------------------------------------------ live status and long poll */
  var pill = document.getElementById('pill'), pillText = document.getElementById('pill-text');
  var banner = document.getElementById('banner'), bannerText = document.getElementById('banner-text');
  function setPill() {
    if (preLaunch()) { var c = countdown(); pill.className = 'status-pill warn'; pillText.textContent = 'Starts in ' + (c.d ? c.d + ' d ' + pad2(c.h) + ' h' : pad2(c.h) + ':' + pad2(c.m) + ':' + pad2(c.s)); return; }
    if (S.online === false) { pill.className = 'status-pill bad'; pillText.textContent = 'Node unreachable'; return; }
    if (!S.status) { pill.className = 'status-pill warn'; pillText.textContent = 'Connecting'; return; }
    var st = S.status;
    if (st.height === 0) { pill.className = 'status-pill warn'; pillText.textContent = 'Empty chain'; return; }
    var age = Math.max(0, nowSrv() - st.tip_timestamp);
    pill.className = 'status-pill ' + (age < BLOCK_TARGET_S * 6 ? 'ok' : 'warn');
    pillText.textContent = 'Block #' + num(latestBlock(st)) + ' · ' + ago(st.tip_timestamp);
  }
  function paintBanner() {
    var off = S.online === false && !preLaunch(), sample = !off && sampleActive();
    banner.classList.toggle('show', off || sample);
    banner.classList.toggle('bad', off);
    if (off) bannerText.textContent = 'The explorer cannot reach the node. Retrying every few seconds.';
    else if (sample) {
      var c = countdown(), left = Date.now() < LAUNCH ? ' The real testnet starts on ' + launchLong() + ' (in ' + (c.d ? c.d + ' d ' + pad2(c.h) + ' h' : pad2(c.h) + ':' + pad2(c.m) + ':' + pad2(c.s)) + ').' : '';
      bannerText.textContent = 'SAMPLE DATA. This explorer shows a demo network so you can see what it will look like: these numbers are not the Vearl testnet.' + left;
    } else bannerText.textContent = '';
  }
  function setOnline(v) {
    if (S.online === v) { paintBanner(); return; }
    S.online = v;
    paintBanner();
    setPill();
  }
  function pollStatus() {
    return api('/status').then(function (st) {
      S.status = st; S.skew = st.node_time - Date.now() / 1000; setOnline(true); setPill(); return st;
    }, function (e) { setOnline(false); throw e; });
  }
  /* Live updates: the status is asked every 4 to 6 seconds (a random part, so that a crowd does not ask at the same instant) and the page is refreshed when the
     height moved. It used to be a long poll (/api/wait), but a node answers only a few of those at a time, and a crowd of visitors (the launch) would have been
     refused; the status is a cheap read that the web server's micro-cache shares between everybody. */
  function liveLoop() {
    var height = null;
    function tick() {
      pollStatus().then(function (st) {
        if (height !== null && st.height !== height && S.refresh) S.refresh();
        height = st.height;
      }, function () {}).then(function () { setTimeout(tick, 4000 + Math.random() * 2000); });
    }
    tick();
  }

  /* ------------------------------------------------------------------ nav + keyboard */
  var nav = document.getElementById('nav'), burger = document.getElementById('burger');
  burger.addEventListener('click', function () {
    var open = nav.classList.toggle('open');
    burger.setAttribute('aria-expanded', String(open));
  });
  document.getElementById('menu').addEventListener('click', function (e) { if (e.target.closest('a')) { nav.classList.remove('open'); burger.setAttribute('aria-expanded', 'false'); } });
  document.addEventListener('keydown', function (e) {
    if (e.key === '/' && !/^(INPUT|TEXTAREA)$/.test((document.activeElement || {}).tagName || '')) {
      var q = document.querySelector('.x-search input');
      if (q) { e.preventDefault(); q.focus(); }
    }
  });
  window.addEventListener('hashchange', function () { route(false); });
  var started = false;
  function begin() { if (started) return; started = true; S.online = null; S.status = null; paintBanner(); route(false); liveLoop(); }
  setInterval(function () { setPill(); paintBanner(); if (preLaunch()) tickCountdown(); else if (!started) begin(); }, 1000);

  if (preLaunch()) route(false); else begin();
})();
