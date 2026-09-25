// Building-asset symbols of the GridExpand map layer: one small "pill" per building with a badge
// per asset (PV, battery, heat pump, EV). Each badge has its own glyph, so identity never rests on
// colour alone. Pills are drawn on a canvas on demand (MapLibre's styleimagemissing event).

export const ASSETS = [
  { id: 'pv', letter: 'p', label: 'Rooftop PV', color: { light: '#d08c00', dark: '#e0a200' } },
  { id: 'battery', letter: 'b', label: 'Battery', color: { light: '#4a3aa7', dark: '#9085e9' } },
  { id: 'heat_pump', letter: 'h', label: 'Heat pump', color: { light: '#d63b3a', dark: '#e66767' } },
  { id: 'ev', letter: 'e', label: 'Electric vehicle', color: { light: '#0b7f3a', dark: '#2fae5f' } },
];
export const ICON_PREFIX = 'gx-asset-';
const RATIO = 3;       // device pixels per CSS pixel: stays sharp when the map scales the icon up
const BADGE = 15;      // badge diameter (CSS px)
const GAP = 2;
const PAD = 2.5;
const POINTER = 4;     // height of the pointer below the pill
const MARGIN = 3;      // room for the shadow

function mode() { return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'; }
function cssVar(name, fallback) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback; }
export function assetColor(asset) { return asset.color[mode()]; }

// White glyph centred at (x, y) inside a badge of radius r and colour `fill`.
function glyph(ctx, id, x, y, r, fill) {
  ctx.save();
  ctx.strokeStyle = '#fff';
  ctx.fillStyle = '#fff';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = r * 0.17;
  if (id === 'pv') {                     // sun
    ctx.beginPath(); ctx.arc(x, y, r * 0.3, 0, Math.PI * 2); ctx.fill();
    for (let i = 0; i < 8; i++) {
      const a = (i * Math.PI) / 4;
      ctx.beginPath();
      ctx.moveTo(x + Math.cos(a) * r * 0.46, y + Math.sin(a) * r * 0.46);
      ctx.lineTo(x + Math.cos(a) * r * 0.66, y + Math.sin(a) * r * 0.66);
      ctx.stroke();
    }
  } else if (id === 'battery') {         // battery with a charge bolt
    const w = r * 1.2, h = r * 0.72;
    ctx.beginPath(); ctx.roundRect(x - w / 2 - r * 0.06, y - h / 2, w, h, r * 0.12); ctx.stroke();
    ctx.beginPath(); ctx.roundRect(x + w / 2 - r * 0.02, y - h * 0.22, r * 0.14, h * 0.44, r * 0.05); ctx.fill();
    ctx.beginPath();
    ctx.moveTo(x + r * 0.02, y - h * 0.34); ctx.lineTo(x - r * 0.22, y + r * 0.03); ctx.lineTo(x - r * 0.02, y + r * 0.03);
    ctx.lineTo(x - r * 0.14, y + h * 0.34); ctx.lineTo(x + r * 0.16, y - r * 0.06); ctx.lineTo(x - r * 0.03, y - r * 0.06);
    ctx.closePath(); ctx.fill();
  } else if (id === 'heat_pump') {       // fan: three blades around a hub
    for (let i = 0; i < 3; i++) {
      const a = (i * 2 * Math.PI) / 3 - Math.PI / 2;
      ctx.beginPath();
      ctx.ellipse(x + Math.cos(a) * r * 0.33, y + Math.sin(a) * r * 0.33, r * 0.33, r * 0.15, a + 0.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.beginPath(); ctx.arc(x, y, r * 0.12, 0, Math.PI * 2); ctx.fill();
  } else if (id === 'ev') {              // car
    const y0 = y + r * 0.02;
    ctx.beginPath();
    ctx.moveTo(x - r * 0.62, y0 + r * 0.22);
    ctx.lineTo(x - r * 0.62, y0 - r * 0.02);
    ctx.lineTo(x - r * 0.4, y0 - r * 0.08);
    ctx.lineTo(x - r * 0.24, y0 - r * 0.36);
    ctx.lineTo(x + r * 0.26, y0 - r * 0.36);
    ctx.lineTo(x + r * 0.44, y0 - r * 0.08);
    ctx.lineTo(x + r * 0.62, y0 - r * 0.02);
    ctx.lineTo(x + r * 0.62, y0 + r * 0.22);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = fill;
    for (const dx of [-0.33, 0.33]) { ctx.beginPath(); ctx.arc(x + dx * r, y0 + r * 0.25, r * 0.17, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = '#fff';
    for (const dx of [-0.33, 0.33]) { ctx.beginPath(); ctx.arc(x + dx * r, y0 + r * 0.25, r * 0.09, 0, Math.PI * 2); ctx.fill(); }
  }
  ctx.restore();
}

function drawBadge(ctx, asset, x, y, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = assetColor(asset);
  ctx.fill();
  glyph(ctx, asset.id, x, y, r, assetColor(asset));
}

// The assets of an icon id suffix ('pbhe' -> PV, battery, heat pump, EV) in badge order.
export function assetsOf(letters) { return ASSETS.filter((a) => letters.includes(a.letter)); }

// ImageData-like object for map.addImage (pixelRatio: RATIO): pill with pointer and badges.
export function drawPill(letters) {
  const assets = assetsOf(letters);
  const n = Math.max(assets.length, 1);
  const w = PAD * 2 + n * BADGE + (n - 1) * GAP;
  const h = PAD * 2 + BADGE;
  const W = w + MARGIN * 2, H = h + POINTER + MARGIN * 2;
  const canvas = document.createElement('canvas');
  canvas.width = Math.ceil(W * RATIO);
  canvas.height = Math.ceil(H * RATIO);
  const ctx = canvas.getContext('2d');
  ctx.scale(RATIO, RATIO);
  const surface = cssVar('--surface', '#ffffff');
  const border = mode() === 'dark' ? 'rgba(255,255,255,0.18)' : 'rgba(18,22,32,0.16)';
  const x0 = MARGIN, y0 = MARGIN, cx = MARGIN + w / 2;
  ctx.save();
  ctx.shadowColor = mode() === 'dark' ? 'rgba(0,0,0,0.55)' : 'rgba(18,22,32,0.28)';
  ctx.shadowBlur = 3;
  ctx.shadowOffsetY = 1;
  ctx.beginPath();
  ctx.roundRect(x0, y0, w, h, h / 2);
  ctx.moveTo(cx - POINTER, y0 + h - 0.5);
  ctx.lineTo(cx, y0 + h + POINTER);
  ctx.lineTo(cx + POINTER, y0 + h - 0.5);
  ctx.fillStyle = surface;
  ctx.fill();
  ctx.restore();
  ctx.beginPath();
  ctx.roundRect(x0 + 0.5, y0 + 0.5, w - 1, h - 1, (h - 1) / 2);
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  ctx.stroke();
  assets.forEach((asset, i) => drawBadge(ctx, asset, x0 + PAD + BADGE / 2 + i * (BADGE + GAP), y0 + PAD + BADGE / 2, BADGE / 2));
  return { width: canvas.width, height: canvas.height, data: ctx.getImageData(0, 0, canvas.width, canvas.height).data };
}

// Data URL of one badge (legend and hover card).
export function badgeUrl(asset, size = 16) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size * 2;
  const ctx = canvas.getContext('2d');
  ctx.scale(2, 2);
  drawBadge(ctx, asset, size / 2, size / 2, size / 2);
  return canvas.toDataURL();
}

export const PIXEL_RATIO = RATIO;

// MapLibre expressions for the enabled assets (on: {pv: true, ...}).
export function iconExpression(on) {
  const parts = ASSETS.filter((a) => on[a.id]).map((a) => ['case', ['to-boolean', ['get', `has_${a.id}`]], a.letter, '']);
  return ['concat', ICON_PREFIX, ...parts, ''];
}
export function assetFilter(on) {
  const parts = ASSETS.filter((a) => on[a.id]).map((a) => ['to-boolean', ['get', `has_${a.id}`]]);
  return parts.length ? ['any', ...parts] : ['==', 1, 0];
}
export function assetCount(props) { return ASSETS.filter((a) => props[`has_${a.id}`]).length; }
