const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const videos = [...document.querySelectorAll('video[data-src]')];
function loadVideo(video) {
  if (!video.getAttribute('src')) { video.src = video.dataset.src; video.load(); }
}
const observer = new IntersectionObserver(entries => {
  entries.forEach(({target: video, isIntersecting}) => {
    if (isIntersecting) {
      loadVideo(video);
      if (video.hasAttribute('data-autoplay') && !reducedMotion.matches && !document.hidden) video.play().catch(() => {});
    } else video.pause();
  });
}, { threshold: .25 });
videos.forEach(video => observer.observe(video));
document.addEventListener('visibilitychange', () => {
  if (document.hidden) document.querySelectorAll('video').forEach(video => video.pause());
});
reducedMotion.addEventListener('change', event => {
  if (event.matches) videos.forEach(video => video.pause());
});
const filmDialog = document.getElementById('film-dialog');
const film = document.getElementById('film-video');
document.querySelectorAll('[data-film]').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('video').forEach(video => video.pause());
  document.getElementById('film-title').textContent = button.dataset.title;
  film.src = button.dataset.film;
  film.load();
  filmDialog.showModal();
  film.play().catch(() => {});
}));
filmDialog.addEventListener('close', () => {
  film.pause(); film.removeAttribute('src'); film.load();
});
const figureDialog = document.getElementById('figure-dialog');
document.querySelectorAll('[data-image]').forEach(button => button.addEventListener('click', () => {
  const image = figureDialog.querySelector('img');
  image.src = button.dataset.image;
  image.alt = button.querySelector('img').alt;
  figureDialog.showModal();
}));
document.querySelectorAll('dialog').forEach(dialog => {
  dialog.querySelector('[data-close]').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  }});
});
document.getElementById('copy').addEventListener('click', async () => {
  const status = document.getElementById('copy-status');
  try { await navigator.clipboard.writeText(document.getElementById('commands').textContent); status.textContent = 'Copied'; }
  catch { status.textContent = 'Select the commands above to copy.'; }
});
