/* Vearl leaderboard: builds the ranking table from /rewards/leaderboard.json (written by vearl-core/tools/testnet_rewards.py).
   Text is always inserted with textContent: nothing from the file is ever interpreted as HTML. */
(function () {
  'use strict';
  var LIMIT = 200;
  var data = null;
  var $ = function (id) { return document.getElementById(id); };

  function el(tag, text, cls) {
    var e = document.createElement(tag);
    if (text != null) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }
  function short(a) { return a.length > 24 ? a.slice(0, 12) + '…' + a.slice(-8) : a; }
  function num(n) { return Number(n).toLocaleString('en-US'); }
  function pct(x) { return (x * 100).toFixed(x >= 0.1 ? 1 : 3) + ' %'; }
  function when(ts) { return new Date(ts * 1000).toISOString().replace('T', ' ').slice(0, 16) + ' UTC'; }

  function setStatus(text) { var s = $('status'); if (s) s.textContent = text; }

  function summary(d) {
    var box = $('summary');
    box.textContent = '';
    var rows = [
      ['Blocks counted', num(d.totals.blocks)],
      ['Miners', num(d.totals.miners)],
      ['Eligible so far', num(d.totals.eligible)],
      ['Last block', when(d.totals.last_block_time)],
      ['Rules in use', 'at least ' + d.params.min_active_days + ' active days and ' + (d.params.min_share * 100).toFixed(3) + ' % of all points; a block counts 1 point, a block with paid work ' + d.params.useful_mult + ', and the first ' + d.params.early_weeks + ' weeks ×' + d.params.early_mult],
      ['Data hash', d.data_hash]
    ];
    rows.forEach(function (r) { box.appendChild(el('div', r[0], 'k')); var v = el('div', r[1], 'v'); if (r[0] === 'Data hash') v.className = 'v mono'; box.appendChild(v); });
  }

  function render() {
    var q = ($('q').value || '').trim().toLowerCase();
    var list = data.miners.filter(function (m) { return !q || m.address.toLowerCase().indexOf(q) !== -1; });
    var shown = q ? list : list.slice(0, LIMIT);
    var body = $('rows');
    body.textContent = '';
    shown.forEach(function (m) {
      var tr = document.createElement('tr');
      var cells = [
        [String(m.rank), ''], [short(m.address), 'mono'], [num(m.blocks), ''], [num(m.useful_blocks), ''], [String(m.points), ''],
        [pct(m.share), ''], [String(m.active_days), ''], [m.eligible ? 'Eligible' : (m.reasons[0] || 'Not yet eligible'), m.eligible ? '' : 'dim']
      ];
      cells.forEach(function (c, i) {
        var td = el('td', c[0], c[1]);
        td.setAttribute('data-k', ['#', 'Address', 'Blocks', 'Useful', 'Points', 'Share', 'Active days', 'Status'][i]);
        if (i === 1) td.title = m.address;
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    var more = $('more');
    if (!shown.length) { more.hidden = false; more.textContent = q ? 'No miner with this address yet: it needs at least one block.' : 'No block yet.'; }
    else if (list.length > shown.length) { more.hidden = false; more.textContent = 'Showing the first ' + LIMIT + ' of ' + num(list.length) + ' miners. Type an address to find a specific one.'; }
    else more.hidden = true;
  }

  function load() {
    var url = '/rewards/leaderboard.json';
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.onload = function () {
      if (xhr.status !== 200) return setStatus('The leaderboard opens with the testnet, on October 23, 2026 at 16:00 UTC, and is refreshed regularly.');
      try { data = JSON.parse(xhr.responseText); } catch (e) { return setStatus('The leaderboard file could not be read. Try again in a moment.'); }
      if (!data.miners || !data.totals) return setStatus('The leaderboard file has an unexpected format.');
      if (!data.totals.blocks) return setStatus('No block counted yet. The leaderboard fills up from the start of the testnet, on October 23, 2026 at 16:00 UTC, and is refreshed every hour.');
      summary(data);
      $('table-panel').hidden = false;
      $('q').addEventListener('input', render);
      render();
    };
    xhr.onerror = function () { setStatus('The leaderboard could not be loaded. Try again in a moment.'); };
    xhr.send();
  }
  load();
}());
