const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const sequenceVideo = document.querySelector('.full-sequence video');
document.querySelector('.full-sequence').addEventListener('toggle', event => {
  if (event.target.open && !sequenceVideo.getAttribute('src')) {
    sequenceVideo.src = sequenceVideo.dataset.src;
    sequenceVideo.load();
  } else if (!event.target.open) sequenceVideo.pause();
});
const explorerTabs = [...document.querySelectorAll('[data-explorer-view]')];
const caseDialog = document.getElementById('case-dialog');
const caseVideo = document.getElementById('case-video');
const atlasViewport = document.getElementById('atlas-viewport');
const atlasInner = document.getElementById('atlas-inner');
let catalog;
let caseFilter = 'all';
let currentCaseId;
let currentPreview;
let selectedAsset = 0;
let zoom = 1;
let panX = 0;
let panY = 0;
let drag;
let detailedAtlasRequested = false;
let detailedAtlasReady = false;
let assetButtons = [];
const byAsset = new Map();

function stopPreview() {
  if (!currentPreview) return;
  currentPreview.pause();
  currentPreview.closest('.case-card').classList.remove('is-playing');
  currentPreview = undefined;
}

function selectedCases() {
  return catalog.cases.filter(item => caseFilter === 'all' || item.task === caseFilter);
}

function showExplorer(view) {
  explorerTabs.forEach(tab => {
    const active = tab.dataset.explorerView === view;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  document.getElementById('case-panel').hidden = view !== 'cases';
  document.getElementById('atlas-panel').hidden = view !== 'atlas';
  stopPreview();
  if (view === 'atlas' && catalog) {
    requestDetailedAtlas();
    drawAtlas();
  }
}
explorerTabs.forEach((tab, index) => {
  tab.addEventListener('click', () => showExplorer(tab.dataset.explorerView));
  tab.addEventListener('keydown', event => {
    const target = {ArrowRight: explorerTabs[(index + 1) % explorerTabs.length], ArrowLeft: explorerTabs[(index + explorerTabs.length - 1) % explorerTabs.length], Home: explorerTabs[0], End: explorerTabs[explorerTabs.length - 1]}[event.key];
    if (target) { event.preventDefault(); showExplorer(target.dataset.explorerView); target.focus(); }
  });
});

function renderCases() {
  stopPreview();
  const grid = document.getElementById('case-grid');
  grid.replaceChildren();
  selectedCases().forEach(item => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'case-card';
    button.setAttribute('aria-label', `Watch ${item.task} trajectory for ${item.asset_id}`);
    const media = document.createElement('span');
    media.className = 'case-card-media';
    const image = document.createElement('img');
    image.src = item.poster;
    image.alt = '';
    image.width = 640;
    image.height = 480;
    image.loading = 'lazy';
    const preview = document.createElement('video');
    preview.muted = true;
    preview.loop = true;
    preview.playsInline = true;
    preview.preload = 'none';
    preview.setAttribute('aria-hidden', 'true');
    const play = document.createElement('span');
    play.className = 'case-play';
    play.textContent = '↗';
    play.setAttribute('aria-hidden', 'true');
    media.append(image, preview, play);
    const caption = document.createElement('span');
    caption.className = 'case-card-caption';
    const title = document.createElement('strong');
    title.textContent = item.task === 'flatten' ? 'Flattening' : 'Folding';
    const duration = document.createElement('span');
    duration.textContent = `${item.duration.toFixed(1)} s`;
    const identity = document.createElement('small');
    identity.textContent = item.asset_id;
    caption.append(title, duration, identity);
    button.append(media, caption);
    button.addEventListener('pointerenter', event => {
      if (event.pointerType !== 'mouse' || reducedMotion.matches) return;
      stopPreview();
      currentPreview = preview;
      if (!preview.getAttribute('src')) preview.src = item.preview;
      preview.play().then(() => {
        if (currentPreview === preview) button.classList.add('is-playing');
      }).catch(() => {});
    });
    button.addEventListener('pointerleave', () => { if (currentPreview === preview) stopPreview(); });
    button.addEventListener('click', () => openCase(item.id));
    grid.append(button);
  });
  document.getElementById('catalog-status').textContent = `${selectedCases().length} selected examples · Each uses a different garment · Full replays at 1× speed`;
  document.querySelectorAll('[data-case-filter]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.caseFilter === caseFilter)));
}
document.querySelectorAll('[data-case-filter]').forEach(button => {
  button.addEventListener('click', () => {
    if (!catalog) return;
    caseFilter = button.dataset.caseFilter;
    renderCases();
  });
});

function openCase(id) {
  if (!catalog) return;
  const item = catalog.cases.find(entry => entry.id === id);
  if (!item) return;
  if (!selectedCases().some(entry => entry.id === id)) { caseFilter = 'all'; renderCases(); }
  currentCaseId = id;
  stopPreview();
  const visible = selectedCases();
  document.getElementById('case-index').textContent = `Example ${visible.findIndex(entry => entry.id === id) + 1} / ${visible.length}`;
  document.getElementById('case-title').textContent = `${item.task === 'flatten' ? 'Flattening' : 'Folding'} · ${item.asset_id}`;
  document.getElementById('case-metadata').textContent = `${item.duration.toFixed(1)} s · ${item.fps} fps · Recorded RGB replay · 1×`;
  caseVideo.pause();
  caseVideo.poster = item.poster;
  caseVideo.src = item.video;
  caseVideo.load();
  if (!caseDialog.open) caseDialog.showModal();
  if (!reducedMotion.matches) caseVideo.play().catch(() => {});
}
function nextCase(direction) {
  if (!catalog) return;
  const visible = selectedCases();
  const index = visible.findIndex(item => item.id === currentCaseId);
  openCase(visible[(index + direction + visible.length) % visible.length].id);
}
document.getElementById('case-close').addEventListener('click', () => caseDialog.close());
document.getElementById('case-prev').addEventListener('click', () => nextCase(-1));
document.getElementById('case-next').addEventListener('click', () => nextCase(1));
caseDialog.addEventListener('close', () => { caseVideo.pause(); caseVideo.removeAttribute('src'); caseVideo.load(); });
caseDialog.addEventListener('click', event => { if (event.target === caseDialog) caseDialog.close(); });
caseDialog.addEventListener('keydown', event => {
  if (event.target === caseVideo) return;
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); nextCase(event.key === 'ArrowLeft' ? -1 : 1); }
});
document.getElementById('case-show-asset').addEventListener('click', () => {
  const item = catalog.cases.find(entry => entry.id === currentCaseId);
  const index = catalog.atlas.assets.findIndex(asset => asset.asset_id === item.asset_id);
  caseDialog.close();
  showExplorer('atlas');
  selectAsset(index);
  focusAsset(index, 3);
  assetButtons[index].focus({preventScroll: true});
  atlasViewport.scrollIntoView({behavior: reducedMotion.matches ? 'instant' : 'smooth', block: 'center'});
});

function assetPosition(index) {
  const atlas = catalog.atlas;
  return {x: atlas.margin_x + (index % atlas.columns) * atlas.cell_width, y: atlas.margin_y + Math.floor(index / atlas.columns) * atlas.cell_height};
}
function updateAssetCrop() {
  const atlas = catalog.atlas;
  const {x, y} = assetPosition(selectedAsset);
  const crop = document.getElementById('asset-crop');
  crop.style.backgroundImage = `url("${detailedAtlasReady ? atlas.image : atlas.preview}")`;
  crop.style.backgroundSize = `${atlas.width / atlas.cell_width * 100}% ${atlas.height / atlas.cell_height * 100}%`;
  crop.style.backgroundPosition = `${x / (atlas.width - atlas.cell_width) * 100}% ${y / (atlas.height - atlas.cell_height) * 100}%`;
  crop.setAttribute('aria-label', catalog.atlas.assets[selectedAsset].asset_id);
}
function selectAsset(index) {
  if (!catalog || index < 0 || index >= catalog.atlas.assets.length) return;
  if (assetButtons[selectedAsset]) { assetButtons[selectedAsset].setAttribute('aria-pressed', 'false'); assetButtons[selectedAsset].tabIndex = -1; }
  selectedAsset = index;
  const asset = catalog.atlas.assets[index];
  assetButtons[index].setAttribute('aria-pressed', 'true');
  assetButtons[index].tabIndex = 0;
  document.getElementById('atlas-selection').textContent = `Garment ${index + 1} / 500`;
  document.getElementById('atlas-asset-id').textContent = asset.asset_id;
  document.getElementById('atlas-material').textContent = asset.material.replaceAll('_', ' ');
  document.getElementById('atlas-size').textContent = asset.size_m.map(value => `${value.toFixed(2)} m`).join(' × ');
  const item = byAsset.get(asset.asset_id);
  document.getElementById('atlas-watch').hidden = !item;
  document.getElementById('atlas-availability').textContent = item ? `A selected ${item.task} trajectory is available for this garment.` : 'No trajectory is selected for this garment in this 24-example browser.';
  updateAssetCrop();
}
function drawAtlas() {
  const width = atlasViewport.clientWidth;
  const height = atlasViewport.clientHeight;
  if (!width || !height) return;
  panX = Math.min(0, Math.max(width * (1 - zoom), panX));
  panY = Math.min(0, Math.max(height * (1 - zoom), panY));
  atlasInner.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
  atlasViewport.classList.toggle('is-zoomed', zoom > 1);
  document.getElementById('atlas-zoom-label').textContent = `${Number(zoom.toFixed(1))}×`;
  document.getElementById('atlas-zoom-out').disabled = zoom <= 1;
  document.getElementById('atlas-zoom-in').disabled = zoom >= 5;
}
function changeZoom(next) {
  const width = atlasViewport.clientWidth;
  const height = atlasViewport.clientHeight;
  const old = zoom;
  zoom = Math.min(5, Math.max(1, next));
  panX = width / 2 - (width / 2 - panX) * zoom / old;
  panY = height / 2 - (height / 2 - panY) * zoom / old;
  drawAtlas();
}
function focusAsset(index, nextZoom = zoom) {
  const {x, y} = assetPosition(index);
  const atlas = catalog.atlas;
  zoom = Math.min(5, Math.max(1, nextZoom));
  panX = atlasViewport.clientWidth / 2 - (x + atlas.cell_width / 2) / atlas.width * atlasViewport.clientWidth * zoom;
  panY = atlasViewport.clientHeight / 2 - (y + atlas.cell_height / 2) / atlas.height * atlasViewport.clientHeight * zoom;
  drawAtlas();
}
function requestDetailedAtlas() {
  if (detailedAtlasRequested) return;
  detailedAtlasRequested = true;
  const status = document.getElementById('atlas-status');
  status.textContent = 'Loading the detailed atlas… You can already select garments in the overview.';
  const image = new Image();
  image.onload = () => {
    detailedAtlasReady = true;
    document.getElementById('atlas-image').src = catalog.atlas.image;
    updateAssetCrop();
    status.textContent = 'Detailed atlas loaded · Select a garment · Zoom for detail · Drag to move when zoomed';
  };
  image.onerror = () => { detailedAtlasRequested = false; status.textContent = 'Overview available. The detailed atlas could not load; select the atlas tab to retry.'; };
  image.src = catalog.atlas.image;
}
document.getElementById('atlas-zoom-in').addEventListener('click', () => { if (catalog) changeZoom(zoom + 0.5); });
document.getElementById('atlas-zoom-out').addEventListener('click', () => { if (catalog) changeZoom(zoom - 0.5); });
document.getElementById('atlas-reset').addEventListener('click', () => { zoom = 1; panX = panY = 0; drawAtlas(); });
document.getElementById('atlas-next-example').addEventListener('click', () => {
  if (!catalog) return;
  const matches = catalog.atlas.assets.map((asset, index) => byAsset.has(asset.asset_id) ? index : -1).filter(index => index >= 0);
  const next = matches.find(index => index > selectedAsset) ?? matches[0];
  selectAsset(next);
  focusAsset(next, 3);
});
document.getElementById('atlas-watch').addEventListener('click', () => {
  const item = byAsset.get(catalog.atlas.assets[selectedAsset].asset_id);
  if (item) openCase(item.id);
});

atlasViewport.addEventListener('pointerdown', event => {
  if (!catalog || event.button !== 0) return;
  drag = {x: event.clientX, y: event.clientY, panX, panY, moved: false};
  atlasViewport.setPointerCapture(event.pointerId);
});
atlasViewport.addEventListener('pointermove', event => {
  if (!drag) return;
  const dx = event.clientX - drag.x;
  const dy = event.clientY - drag.y;
  if (Math.hypot(dx, dy) > 5) drag.moved = true;
  if (zoom > 1) { panX = drag.panX + dx; panY = drag.panY + dy; drawAtlas(); }
});
atlasViewport.addEventListener('pointerup', event => {
  if (!drag) return;
  if (!drag.moved) {
    const rect = atlasViewport.getBoundingClientRect();
    const atlas = catalog.atlas;
    const x = (event.clientX - rect.left - panX) / zoom / rect.width * atlas.width;
    const y = (event.clientY - rect.top - panY) / zoom / rect.height * atlas.height;
    const col = Math.floor((x - atlas.margin_x) / atlas.cell_width);
    const row = Math.floor((y - atlas.margin_y) / atlas.cell_height);
    if (col >= 0 && col < atlas.columns && row >= 0 && row < atlas.rows) selectAsset(row * atlas.columns + col);
  }
  drag = undefined;
});
atlasViewport.addEventListener('pointercancel', () => { drag = undefined; });
new ResizeObserver(drawAtlas).observe(atlasViewport);

fetch('assets/catalog/catalog.json').then(response => {
  if (!response.ok) throw new Error('Catalog could not load');
  return response.json();
}).then(data => {
  catalog = data;
  data.cases.forEach(item => byAsset.set(item.asset_id, item));
  renderCases();
  const hits = document.getElementById('atlas-hits');
  data.atlas.assets.forEach((asset, index) => {
    const {x, y} = assetPosition(index);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `atlas-hit${byAsset.has(asset.asset_id) ? ' has-replay' : ''}`;
    button.style.cssText = `left:${x / data.atlas.width * 100}%;top:${y / data.atlas.height * 100}%;width:${data.atlas.cell_width / data.atlas.width * 100}%;height:${data.atlas.cell_height / data.atlas.height * 100}%`;
    button.setAttribute('aria-label', `Garment ${index + 1}: ${asset.asset_id}${byAsset.has(asset.asset_id) ? ', replay available' : ''}`);
    button.setAttribute('aria-pressed', 'false');
    button.tabIndex = -1;
    button.addEventListener('click', event => { if (event.detail === 0) selectAsset(index); });
    button.addEventListener('keydown', event => {
      const offset = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -data.atlas.columns, ArrowDown: data.atlas.columns}[event.key];
      if (offset === undefined) return;
      event.preventDefault();
      const next = Math.min(data.atlas.assets.length - 1, Math.max(0, selectedAsset + offset));
      selectAsset(next);
      focusAsset(next);
      assetButtons[next].focus({preventScroll: true});
    });
    assetButtons.push(button);
    hits.append(button);
  });
  selectAsset(data.atlas.assets.findIndex(asset => byAsset.has(asset.asset_id)));
  if (!document.getElementById('atlas-panel').hidden) requestDetailedAtlas();
}).catch(() => {
  document.getElementById('catalog-status').textContent = 'The catalog could not load. Refresh the page to retry.';
  document.getElementById('atlas-status').textContent = 'The catalog could not load. Refresh the page to retry.';
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden) { stopPreview(); caseVideo.pause(); }
});
reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) stopPreview(); });
