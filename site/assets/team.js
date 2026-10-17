/* Team page: the cards, and the sheet that opens from the photo of a member (the photo flies from the card to the sheet).
   To add a member, add an object to TEAM and a photo in /assets/team/ (square, at least 800 px). */
(function () {
  'use strict';
  var TEAM = [
    {
      id: 'clovis-anicet',
      name: 'Clovis Anicet',
      role: 'Founder',
      photo: '/assets/team/clovis-anicet.jpg',
      short: 'Marketing, communication and partnerships.',
      about: [
        'Founder of Vearl. Born in Guadeloupe, started building video games at 16 after leaving school, and learned it all alone.',
        'Creates Fortnite maps. The numbers below come from the public creator profile, linked at the bottom.',
        'At Vearl: marketing, communication and partnerships.'
      ],
      facts: [
        ['Origin', 'Born in Guadeloupe'],
        ['Age', '18'],
        ['Background', 'Left school at 16 to build video games, self-taught'],
        ['Creations', 'Fortnite maps'],
        ['At Vearl', 'Marketing, communication, partnerships']
      ],
      tags: ['Founder', 'Guadeloupe', 'Self-taught', 'Fortnite creator', 'Marketing', 'Partnerships'],
      stats: [
        { to: 27.6, kind: 'millions', label: 'Minutes played' },
        { to: 5700, kind: 'int', label: 'Followers' },
        { to: 1800, kind: 'rank', label: 'Creator rank' }
      ],
      statsNote: 'Fortnite maps, as shown on the creator profile on fortnite.gg on 16 October 2026. 27.6 million minutes is about 52 years of total play time.',
      links: [{ label: 'Fortnite creator profile', href: 'https://fortnite.gg/creator/creativehe' }]
    }
  ];

  var grid = document.getElementById('grid'), dlg = document.getElementById('member'), big = document.getElementById('big');
  if (!grid || !dlg || typeof dlg.showModal !== 'function') return;
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var current = null, closing = false;

  function pad(n) { return (n < 10 ? '0' : '') + n; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  TEAM.forEach(function (m, i) {
    var li = document.createElement('li');
    li.innerHTML =
      '<button type="button" class="tm-card" data-id="' + esc(m.id) + '" aria-haspopup="dialog" aria-label="' + esc(m.name) + ', ' + esc(m.role) + ': show the details">' +
      '<span class="tm-idx">Member ' + pad(i + 1) + '</span>' +
      '<span class="tm-photo"><img src="' + esc(m.photo) + '" alt="Portrait of ' + esc(m.name) + '" width="900" height="900" ' + (i === 0 ? 'fetchpriority="high"' : 'loading="lazy"') + '></span>' +
      '<span class="tm-name">' + esc(m.name) + '</span>' +
      '<span class="tm-role">' + esc(m.role) + '</span>' +
      '<span class="tm-short">' + esc(m.short) + '</span>' +
      '<span class="tm-more">Details<i></i></span>' +
      '</button>';
    grid.appendChild(li);
  });
  var soon = document.createElement('li');
  soon.innerHTML = '<div class="tm-soon"><div class="plus"></div><b>The team is growing</b><span>More members will be introduced here.</span></div>';
  grid.appendChild(soon);

  function fmt(v, kind) {
    if (kind === 'millions') return v.toFixed(1) + 'M';
    if (kind === 'rank') return '#' + Math.round(v).toLocaleString('en-US');
    return Math.round(v).toLocaleString('en-US');
  }
  function countUp() {                                  // the numbers of the sheet count up once, after the lines have arrived
    var els = document.querySelectorAll('#d-stats b[data-to]');
    Array.prototype.forEach.call(els, function (el) {
      var to = parseFloat(el.getAttribute('data-to')), kind = el.getAttribute('data-kind');
      if (reduce) { el.textContent = fmt(to, kind); return; }
      var t0 = null, dur = 1500;
      el.textContent = fmt(0, kind);
      (function step(ts) {
        if (!dlg.open) return;
        if (t0 === null) t0 = ts;
        var u = Math.min(1, (ts - t0) / dur), e = 1 - Math.pow(1 - u, 4);
        el.textContent = fmt(to * e, kind);
        if (u < 1) requestAnimationFrame(step); else el.textContent = fmt(to, kind);
      })(performance.now() + 0);
      window.setTimeout(function () { /* guard: always end on the exact figure */ if (dlg.open) el.textContent = fmt(to, kind); }, dur + 900);
    });
  }

  function fill(m, index) {
    document.getElementById('d-role').textContent = m.role;
    document.getElementById('tm-title').textContent = m.name;
    big.src = m.photo; big.alt = 'Portrait of ' + m.name;
    document.getElementById('idx2').textContent = 'Member ' + pad(index + 1) + ' of the Vearl team';
    var k = 2;
    document.getElementById('d-about').innerHTML = m.about.map(function (p) { return '<p data-i="' + (k++) + '">' + esc(p) + '</p>'; }).join('');
    document.getElementById('d-facts').innerHTML = m.facts.map(function (f) { var a = '<dt data-i="' + k + '">' + esc(f[0]) + '</dt><dd data-i="' + k + '">' + esc(f[1]) + '</dd>'; k += 1; return a; }).join('');
    document.getElementById('d-tags').innerHTML = m.tags.map(function (t) { return '<li data-i="' + (k++) + '">' + esc(t) + '</li>'; }).join('');
    var st = document.getElementById('d-stats');
    st.innerHTML = (m.stats || []).map(function (s) {
      return '<div class="tm-stat" data-i="' + (k++) + '"><b data-to="' + s.to + '" data-kind="' + s.kind + '">' + fmt(s.to, s.kind) + '</b><span>' + esc(s.label) + '</span></div>';
    }).join('') + (m.statsNote ? '<p class="tm-stat-note" data-i="' + (k++) + '">' + esc(m.statsNote) + '</p>' : '');
    document.getElementById('d-links').innerHTML = (m.links || []).map(function (l) {
      return '<a class="tm-link" data-i="' + (k++) + '" href="' + esc(l.href) + '" target="_blank" rel="noopener noreferrer">' + esc(l.label) + '<i aria-hidden="true"></i></a>';
    }).join('');
    // the delays of the staggered reveal: set through the CSSOM (an inline style attribute would be refused by the site's content security policy)
    Array.prototype.forEach.call(document.getElementById('sheet').querySelectorAll('[data-i]'), function (e) { e.style.setProperty('--i', e.getAttribute('data-i')); });
  }

  function open(card, push) {
    var id = card.getAttribute('data-id'), idx = TEAM.findIndex(function (m) { return m.id === id; });
    if (idx < 0 || current) return;
    current = card; closing = false;
    fill(TEAM[idx], idx);
    document.documentElement.classList.add('tm-lock');
    dlg.classList.remove('out'); dlg.classList.remove('in');
    dlg.showModal();
    var src = card.querySelector('img').getBoundingClientRect(), dst = big.getBoundingClientRect();
    if (!reduce && dst.width && src.width) {                                   // the photo flies from the card to its place in the sheet
      card.classList.add('is-open');
      big.animate([
        { transformOrigin: '0 0', transform: 'translate(' + (src.left - dst.left) + 'px,' + (src.top - dst.top) + 'px) scale(' + (src.width / dst.width) + ',' + (src.height / dst.height) + ')' },
        { transformOrigin: '0 0', transform: 'none' }
      ], { duration: 640, easing: 'cubic-bezier(.2,.9,.2,1)' });
    }
    void dlg.offsetWidth;
    dlg.classList.add('in');
    window.setTimeout(countUp, 650);
    if (push) { try { history.replaceState(null, '', '#' + id); } catch (e) { /* ignore */ } }
    document.getElementById('close').focus({ preventScroll: true });
  }

  function finish() {
    dlg.close(); dlg.classList.remove('in', 'out');
    document.documentElement.classList.remove('tm-lock');
    if (current) { current.classList.remove('is-open'); current.focus({ preventScroll: true }); }
    current = null; closing = false;
    try { history.replaceState(null, '', location.pathname + location.search); } catch (e) { /* ignore */ }
  }

  function close() {
    if (!current || closing) return;
    closing = true; dlg.classList.remove('in'); dlg.classList.add('out');
    if (reduce) { finish(); return; }
    var src = current.querySelector('img').getBoundingClientRect(), dst = big.getBoundingClientRect();
    var a = big.animate([
      { transformOrigin: '0 0', transform: 'none' },
      { transformOrigin: '0 0', transform: 'translate(' + (src.left - dst.left) + 'px,' + (src.top - dst.top) + 'px) scale(' + (src.width / dst.width) + ',' + (src.height / dst.height) + ')' }
    ], { duration: 360, easing: 'cubic-bezier(.5,0,.2,1)', fill: 'forwards' });
    a.onfinish = function () { a.cancel(); finish(); };
  }

  grid.addEventListener('click', function (e) { var c = e.target.closest('.tm-card'); if (c) open(c, true); });
  document.getElementById('close').addEventListener('click', close);
  dlg.addEventListener('cancel', function (e) { e.preventDefault(); close(); });
  dlg.addEventListener('click', function (e) { if (e.target === dlg) close(); });

  var h = location.hash.replace('#', '');
  if (h) { var c = grid.querySelector('.tm-card[data-id="' + h.replace(/[^a-z0-9-]/gi, '') + '"]'); if (c) window.setTimeout(function () { open(c, false); }, 250); }
}());
