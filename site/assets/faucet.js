/* Vearl faucet page: posts the address to the faucet (POST /drip) and shows the answer. No third party, no innerHTML. */
(function () {
  'use strict';
  var meta = document.querySelector('meta[name="vearl-faucet"]');
  var DRIP = (meta && meta.content) || '/drip';
  var form = document.getElementById('form'), addr = document.getElementById('addr'), err = document.getElementById('err');
  var go = document.getElementById('go'), panel = document.getElementById('result-panel'), out = document.getElementById('result');

  var nav = document.getElementById('nav'), burger = document.getElementById('burger');
  if (nav && burger) burger.addEventListener('click', function () {
    var open = nav.classList.toggle('open');
    burger.setAttribute('aria-expanded', String(open));
  });
  var exp = document.querySelector('meta[name="vearl-explorer"]');
  if (exp) document.querySelectorAll('a[data-ext="explorer"]').forEach(function (a) { a.setAttribute('href', exp.content); });

  function fail(msg) { err.textContent = msg; err.classList.add('show'); }
  function row(k, v, mono) {
    var a = document.createElement('div'), b = document.createElement('div');
    a.className = 'k'; a.textContent = k;
    b.className = 'v' + (mono ? ' mono' : ''); b.textContent = v;
    out.appendChild(a); out.appendChild(b);
  }
  function wait(s) { return s < 90 ? s + ' s' : s < 5400 ? Math.round(s / 60) + ' min' : Math.round(s / 3600) + ' h'; }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    err.classList.remove('show');
    var a = addr.value.trim();
    if (!/^vrlq?1[0-9a-z]{20,}$/.test(a)) { fail('That does not look like a Vearl address: it starts with vrl1.'); return; }
    go.disabled = true; go.textContent = 'Sending…';
    var ctl = typeof AbortController === 'function' ? new AbortController() : null;
    var timer = setTimeout(function () { if (ctl) ctl.abort(); }, 20000);
    fetch(DRIP, { method: 'POST', body: JSON.stringify({ address: a }), headers: { 'Content-Type': 'application/json' }, signal: ctl ? ctl.signal : undefined })
      .then(function (r) { return r.json().catch(function () { return {}; }).then(function (j) { return { status: r.status, body: j }; }); })
      .then(function (r) {
        if (r.status === 200 && r.body.ok) {
          out.replaceChildren();
          row('Amount', r.body.amount + ' VRL (test tokens)');
          row('To', r.body.to, true);
          row('Transaction', r.body.tx, true);
          panel.hidden = false;
        } else if (r.status === 429 && r.body.retry_in_secs) {
          fail('You already received tokens. Try again in ' + wait(r.body.retry_in_secs) + '.');
        } else if (r.status === 429) {
          fail('The faucet has reached its limit for today. Try again tomorrow.');
        } else if (r.status === 503) {
          fail('The faucet cannot reach the network right now. Try again in a few minutes.');
        } else {
          fail((r.body && r.body.error) || 'The request was refused.');
        }
      })
      .catch(function () { fail('The faucet could not be reached. Try again in a moment.'); })
      .then(function () { clearTimeout(timer); go.disabled = false; go.textContent = 'Get test tokens'; });
  });
})();
