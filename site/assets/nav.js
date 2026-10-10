/* Mobile menu button for the pages that have no script of their own (participate, leaderboard). */
(function () {
  'use strict';
  var nav = document.getElementById('nav'), burger = document.getElementById('burger'), menu = document.getElementById('menu');
  if (!nav || !burger) return;
  burger.addEventListener('click', function () {
    var open = nav.classList.toggle('open');
    burger.setAttribute('aria-expanded', String(open));
  });
  if (menu) menu.addEventListener('click', function (e) {
    if (e.target.closest('a')) { nav.classList.remove('open'); burger.setAttribute('aria-expanded', 'false'); }
  });
}());
