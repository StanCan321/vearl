/* Vearl whitepaper: navigation, table of contents with current section, back-to-top. No dependencies. */
(() => {
  'use strict';
  const $ = (s) => document.querySelector(s);
  /* contact address, assembled here so that it is not in the page source for address harvesters */
  document.querySelectorAll('.mail').forEach((s) => {
    const addr = s.dataset.u + '@' + s.dataset.d, a = document.createElement('a');
    a.href = 'mailto:' + addr; a.textContent = s.dataset.label || addr; a.className = 'mail-link';
    s.replaceWith(a);
  });

  const nav = $('#nav'), burger = $('#burger'), menu = $('#menu'), toc = $('#toc'), top = $('#to-top');

  const onScroll = () => {
    nav.classList.toggle('solid', window.scrollY > 24);
    if (top) top.classList.toggle('show', window.scrollY > 900);
  };
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

  // the contents list is always open on wide screens, collapsed on phones and tablets
  const wide = window.matchMedia('(min-width: 961px)');
  const syncToc = () => { if (toc) toc.open = wide.matches; };
  syncToc();
  if (wide.addEventListener) wide.addEventListener('change', () => { syncToc(); if (wide.matches) setMenu(false); });
  if (toc) {
    toc.querySelector('summary').addEventListener('click', (e) => { if (wide.matches) e.preventDefault(); });
    toc.querySelectorAll('a').forEach((a) => a.addEventListener('click', () => { if (!wide.matches) toc.open = false; }));
  }

  // current section in the contents list and in the top navigation
  const tocLinks = toc ? [...toc.querySelectorAll('a')] : [];
  const navLinks = [...menu.querySelectorAll('a[href^="#"]')];
  const all = tocLinks.concat(navLinks);
  const byId = new Map();
  all.forEach((a) => { const id = a.getAttribute('href').slice(1); (byId.get(id) || byId.set(id, []).get(id)).push(a); });
  if ('IntersectionObserver' in window) {
    const spy = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        all.forEach((a) => a.classList.remove('active'));
        (byId.get(e.target.id) || []).forEach((a) => a.classList.add('active'));
        // a top-level navigation link stays lit for every section of its group
        const group = e.target.dataset.group;
        if (group) (byId.get(group) || []).forEach((a) => a.classList.add('active'));
        const cur = tocLinks.find((a) => a.classList.contains('active'));
        if (cur && wide.matches && toc.scrollHeight > toc.clientHeight) {
          const r = cur.getBoundingClientRect(), t = toc.getBoundingClientRect();
          if (r.top < t.top + 40 || r.bottom > t.bottom - 40) toc.scrollTop += r.top - t.top - t.height / 2;
        }
      });
    }, { rootMargin: '-30% 0px -60% 0px' });
    document.querySelectorAll('.paper section[id]').forEach((s) => spy.observe(s));
  }

  // long tables turn into stacked rows on phones: every cell after the first carries its column title
  document.querySelectorAll('.tbl:not(.keep)').forEach((t) => {
    const heads = [...t.querySelectorAll('thead th')].map((th) => th.textContent.trim());
    t.querySelectorAll('tbody tr').forEach((tr) => [...tr.children].forEach((td, i) => { if (i > 0 && heads[i]) td.dataset.label = heads[i]; }));
    t.classList.add('stack');
  });

  if (top) top.addEventListener('click', (e) => e.preventDefault() || window.scrollTo({ top: 0, behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' }));

  // deep links: land on the section at once
  if (location.hash.length > 1) {
    const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (target) {
      const html = document.documentElement, prev = html.style.scrollBehavior;
      html.style.scrollBehavior = 'auto';
      target.scrollIntoView();
      html.style.scrollBehavior = prev;
    }
  }
})();
