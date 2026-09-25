// Expansion result layer on pylovo's MapLibre map: cables coloured by the required action,
// transformers by their critical loading, building assets (PV, battery, heat pump, EV) as small
// symbols, with a legend card that filters the assets and shows the hovered feature.
import { createApp, markRaw, watch } from 'vue';
import { euro } from './lib.js';
import { ASSETS, ICON_PREFIX, PIXEL_RATIO, assetCount, assetFilter, badgeUrl, drawPill, iconExpression } from './assets.js';

const SOURCES = { lines: 'gridexpand-lines', trafos: 'gridexpand-trafos', assets: 'gridexpand-assets' };
const LAYERS = ['gridexpand-lines-casing', 'gridexpand-lines', 'gridexpand-trafos', 'gridexpand-assets-dot', 'gridexpand-assets'];
const GRID_LAYERS = LAYERS.slice(0, 3);
const ASSET_LAYERS = LAYERS.slice(3);
const PICK = ['gridexpand-assets', 'gridexpand-assets-dot', 'gridexpand-trafos', 'gridexpand-lines'];
const EMPTY = { type: 'FeatureCollection', features: [] };
const SOURCE_OF = { 'gridexpand-trafos': SOURCES.trafos, 'gridexpand-lines': SOURCES.lines,
  'gridexpand-assets': SOURCES.assets, 'gridexpand-assets-dot': SOURCES.assets };
const KIND_OF = { [SOURCES.trafos]: 'transformer', [SOURCES.lines]: 'cable', [SOURCES.assets]: 'building' };

export function installMapLayer(ctx) {
  const { host, state } = ctx;
  const { palette, format, iconSvg } = host.ui;
  let map = null;
  let overlay = null;
  let hovered = null;

  function colors() {
    const dark = palette.mode() === 'dark';
    return { none: palette.ink(dark ? 'neutral' : 'muted'), add1: palette.STATUS.serious, add2: palette.STATUS.critical,
      casing: dark ? '#0b0d12' : '#ffffff', dot: palette.ink('secondary') };
  }
  function loadingColor() {
    return ['interpolate', ['linear'], ['coalesce', ['get', 'loading_percent'], 0], ...palette.LOADING_STOPS.flat()];
  }
  function removeLayers() {
    if (!map?.style) return;
    for (const id of [...LAYERS].reverse()) if (map.getLayer(id)) map.removeLayer(id);
    for (const id of Object.values(SOURCES)) if (map.getSource(id)) map.removeSource(id);
  }
  // Pills are generated when a layer first asks for them; a theme switch drops them (new colours).
  function onImageMissing(e) {
    if (!e.id?.startsWith(ICON_PREFIX) || map.hasImage(e.id)) return;
    map.addImage(e.id, drawPill(e.id.slice(ICON_PREFIX.length)), { pixelRatio: PIXEL_RATIO });
  }
  function dropImages() {
    if (!map?.style) return;
    for (const id of map.listImages()) if (id.startsWith(ICON_PREFIX)) map.removeImage(id);
  }
  function applyAssetFilter() {
    if (!map?.getLayer('gridexpand-assets')) return;
    const on = state.layer.assetsOn;
    for (const id of ASSET_LAYERS) map.setFilter(id, assetFilter(on));
    map.setLayoutProperty('gridexpand-assets', 'icon-image', iconExpression(on));
  }
  function applyVisibility() {
    if (!map?.getLayer(GRID_LAYERS[0])) return;
    for (const id of GRID_LAYERS) map.setLayoutProperty(id, 'visibility', state.layer.gridOn ? 'visible' : 'none');
  }
  function addLayers() {
    if (!map || !map.isStyleLoaded()) return;
    removeLayers();
    const data = state.layer.data;
    if (!state.layer.on || !data) return;
    const c = colors();
    map.addSource(SOURCES.lines, { type: 'geojson', data: data.lines || EMPTY });
    map.addSource(SOURCES.trafos, { type: 'geojson', data: data.transformers || EMPTY });
    map.addSource(SOURCES.assets, { type: 'geojson', data: state.layer.assets || EMPTY });
    const action = (none, one, many) => ['match', ['get', 'action'], 'add_1', one, 'add_2plus', many, none];
    const hover = (normal, wide) => ['case', ['boolean', ['feature-state', 'hover'], false], wide, normal];
    // Zoom must stay the input of the top-level interpolate; per-feature factors go into the outputs.
    map.addLayer({ id: LAYERS[0], type: 'line', source: SOURCES.lines, filter: ['!=', ['get', 'action'], 'none'],
      layout: { 'line-cap': 'round', 'line-join': 'round' },
      paint: { 'line-color': c.casing, 'line-opacity': 0.85,
        'line-width': ['interpolate', ['linear'], ['zoom'], 13, hover(4, 6), 18, hover(10, 14)] } });
    map.addLayer({ id: LAYERS[1], type: 'line', source: SOURCES.lines, layout: { 'line-cap': 'round', 'line-join': 'round' },
      paint: { 'line-color': action(c.none, c.add1, c.add2), 'line-opacity': action(0.7, 1, 1),
        'line-width': ['interpolate', ['linear'], ['zoom'], 13, hover(action(1, 2.5, 3), 4.5), 18, hover(action(2.5, 6, 7.5), 10)] } });
    map.addLayer({ id: LAYERS[2], type: 'circle', source: SOURCES.trafos,
      paint: { 'circle-color': loadingColor(),
        'circle-radius': ['interpolate', ['linear'], ['zoom'], 12, hover(6, 8), 18, hover(12, 15)],
        'circle-stroke-color': ['case', ['boolean', ['get', 'requires_expansion'], false], c.add2, c.casing],
        'circle-stroke-width': ['case', ['boolean', ['get', 'requires_expansion'], false], 3, 1.5] } });
    // Every building with assets keeps a dot; its pill shows where there is room (collision
    // detection; buildings with more assets win).
    map.addLayer({ id: LAYERS[3], type: 'circle', source: SOURCES.assets, minzoom: 13,
      paint: { 'circle-color': c.dot, 'circle-stroke-color': c.casing, 'circle-stroke-width': 1,
        'circle-radius': ['interpolate', ['linear'], ['zoom'], 13, hover(1.8, 3), 18, hover(3, 4.5)] } });
    map.addLayer({ id: LAYERS[4], type: 'symbol', source: SOURCES.assets, minzoom: 14.5,
      layout: { 'icon-image': iconExpression(state.layer.assetsOn), 'icon-anchor': 'bottom', 'icon-offset': [0, -2],
        'icon-size': ['interpolate', ['linear'], ['zoom'], 14.5, 0.62, 16.5, 0.85, 18.5, 1.05],
        'icon-padding': 1, 'symbol-sort-key': ['-', 0, ['get', 'n_assets']] },
      paint: { 'icon-opacity': hover(0.96, 1) } });
    applyAssetFilter();
    applyVisibility();
  }
  function setHover(feature) {
    const source = feature ? SOURCE_OF[feature.layer.id] : null;
    const id = feature ? `${source}:${feature.id}` : null;
    if (hovered?.key === id) return;
    if (hovered && map.getSource(hovered.source)) map.setFeatureState({ source: hovered.source, id: hovered.id }, { hover: false });
    hovered = null;
    state.layer.hover = null;
    if (!feature) return;
    hovered = { key: id, source, id: feature.id };
    map.setFeatureState({ source, id: feature.id }, { hover: true });
    state.layer.hover = { kind: KIND_OF[source], props: { ...feature.properties } };
  }
  function onMove(e) {
    if (!state.layer.on || !map.getLayer(PICK[2])) return;
    const box = [[e.point.x - 4, e.point.y - 4], [e.point.x + 4, e.point.y + 4]];
    const hits = map.queryRenderedFeatures(box, { layers: PICK.filter((id) => map.getLayer(id)) });
    // Buildings first, then transformers; of overlapping cables the one with the largest reinforcement.
    const severity = (f) => ({ add_2plus: 0, add_1: 1 }[f.properties.action] ?? 2);
    hits.sort((a, b) => PICK.indexOf(a.layer.id) - PICK.indexOf(b.layer.id) || severity(a) - severity(b));
    setHover(hits[0] || null);
  }

  const Legend = {
    data() { return { collapsed: false }; },
    computed: {
      layer() { return state.layer; },
      analysis() { return state.layer.data?.label || state.layer.key; },
      counts() {
        const f = state.layer.data?.lines?.features || [];
        return { none: f.filter((x) => x.properties.action === 'none').length, add1: f.filter((x) => x.properties.action === 'add_1').length,
          add2: f.filter((x) => x.properties.action === 'add_2plus').length, trafos: (state.layer.data?.transformers?.features || []).length };
      },
      c() { void host.store.theme; return colors(); },
      ramp() { void host.store.theme; return palette.gradientCss(palette.LOADING_STOPS); },
      assetRows() {
        void host.store.theme;
        const totals = state.layer.assets?.totals || {};
        return ASSETS.map((a) => ({ ...a, url: badgeUrl(a), total: totals[a.id] || null, on: state.layer.assetsOn[a.id] }));
      },
      assetsInfo() {
        const a = state.layer.assets;
        if (!a) return null;
        if (a.error) return { error: a.error };
        if (a.stage === 'pre') return { text: 'Status quo: no added assets. Select a post case to see them.' };
        if (!a.features.length) return { text: 'No building assets were stored for these grids (runs before migration 0005 have none).' };
        return null;
      },
      rows() {
        const h = state.layer.hover;
        if (!h) return [];
        const p = h.props;
        const pct = (v) => (v === null || v === undefined ? '–' : `${format.num(v, 1)} %`);
        if (h.kind === 'transformer') {
          return [['Grid', `${p.kcid}/${p.bcid}`], ['Rated power', `${format.num(p.rated_kva)} kVA`], ['Critical loading', pct(p.loading_percent)],
            ['Required', `${format.num(p.required_kva)} kVA (+${format.num(p.additional_kva)})`], ['Measure', String(p.cost_basis || '–').replace(/_/g, ' ')],
            ['Cost', euro(p.cost_eur)]];
        }
        if (h.kind === 'building') return [];
        return [['Grid', `${p.kcid}/${p.bcid}`], ['Cable', `${p.std_type || '–'} · ${format.meters((p.length_km || 0) * 1000)}`],
          ['Critical loading', pct(p.loading_percent)], ['Parallel cables', `${p.required_parallel} required (+${p.additional_parallel})`],
          ...(p.reinforcement ? [['Reinforcement', p.reinforcement]] : []), ['Cost', euro(p.cost_eur)]];
      },
      building() {
        const h = state.layer.hover;
        if (h?.kind !== 'building') return null;
        const p = h.props;
        const n = (v, d = 1) => format.num(v, d);
        const asset = (id) => ASSETS.find((a) => a.id === id);
        const lines = [];
        if (p.has_pv) lines.push({ asset: asset('pv'), label: 'Rooftop PV', value: `${n(p.pv_kw)} kWp` });
        if (p.has_battery) lines.push({ asset: asset('battery'), label: 'Battery', value: `${n(p.battery_kwh)} kWh · ${n(p.battery_kw)} kW` });
        if (p.has_heat_pump) {
          const extra = [p.has_heating_rod ? `+ ${n(p.heating_rod_kw)} kW heating rod` : null,
            p.has_heat_storage ? `${n(p.heat_storage_kwh)} kWh buffer` : null].filter(Boolean).join(' · ');
          lines.push({ asset: asset('heat_pump'), label: 'Heat pump', value: `${n(p.heat_pump_kw)} kW el`, sub: extra });
        }
        if (p.has_ev) {
          lines.push({ asset: asset('ev'), label: p.ev_count === 1 ? 'Electric vehicle' : `${p.ev_count} electric vehicles`,
            value: `${n(p.ev_kwh, 0)} kWh`, sub: p.charger_kw ? `home charging ${n(p.charger_kw, 0)} kW` : '' });
        }
        const facts = [p.use, p.households ? `${p.households} household${p.households === 1 ? '' : 's'}` : null,
          p.floor_area ? `${format.num(p.floor_area, 0)} m²` : null].filter(Boolean).join(' · ');
        return { title: p.address || p.objectid, facts, grid: `${p.kcid}/${p.bcid} · bus ${p.bus}`, lines,
          shared: p.shared_bus ? `Assets of bus ${p.bus}, shared by ${p.shared_bus} buildings` : null,
          urls: Object.fromEntries(ASSETS.map((a) => [a.id, badgeUrl(a, 14)])) };
      },
    },
    methods: {
      icon: (name) => iconSvg(name, 13), hide() { state.layer.on = false; },
      num: (v, d = 0) => format.num(v, d),
      toggle(id) { state.layer.assetsOn = { ...state.layer.assetsOn, [id]: !state.layer.assetsOn[id] }; },
      totalText(r) {
        const t = r.total;
        if (!t) return '0';
        if (r.id === 'pv') return `${t.buildings} · ${format.num(t.power_kw, 0)} kWp`;
        if (r.id === 'battery') return `${t.buildings} · ${format.num(t.energy_kwh, 0)} kWh`;
        if (r.id === 'heat_pump') return `${t.buildings} · ${format.num(t.power_kw, 0)} kW`;
        return `${t.units} in ${t.buildings}`;
      },
    },
    template: `
      <div v-if="layer.on" class="map-overlay floating gx-legend">
        <div class="row between" style="gap: 6px">
          <div class="legend-title" style="margin: 0">GridExpand · {{ analysis }}</div>
          <span class="row" style="gap: 2px">
            <button class="btn ghost icon sm" :title="collapsed ? 'Expand' : 'Collapse'" @click="collapsed = !collapsed" v-html="icon(collapsed ? 'chevronRight' : 'chevronDown')"></button>
            <button class="btn ghost icon sm" title="Hide the GridExpand layer" @click="hide" v-html="icon('x')"></button>
          </span>
        </div>
        <div v-if="layer.loading" class="hint">Loading…</div>
        <div v-else-if="layer.error" class="hint" style="color: var(--critical-text)">{{ layer.error }}</div>
        <template v-else-if="!collapsed">
          <label class="gx-check"><input type="checkbox" v-model="layer.gridOn"> Cables and transformers</label>
          <template v-if="layer.gridOn">
            <div class="legend-row"><span class="sw line" :style="{background: c.none}"></span>No reinforcement<span class="count">{{ counts.none }}</span></div>
            <div class="legend-row"><span class="sw line" :style="{background: c.add1, height: '4px'}"></span>+1 parallel cable<span class="count">{{ counts.add1 }}</span></div>
            <div class="legend-row"><span class="sw line" :style="{background: c.add2, height: '5px'}"></span>+2 or more cables<span class="count">{{ counts.add2 }}</span></div>
            <div class="legend-title" style="margin-top: 8px">Transformer loading (critical hour)</div>
            <div class="legend-ramp" :style="{background: ramp}"></div>
            <div class="legend-scale"><span>0 %</span><span>50</span><span>75</span><span>100</span><span>130 %</span></div>
            <div class="legend-row" style="margin-top: 4px"><span class="sw round" :style="{background: 'transparent', boxShadow: 'inset 0 0 0 2.5px ' + c.add2}"></span>needs a larger transformer</div>
          </template>
          <div class="legend-title" style="margin-top: 9px">Building assets</div>
          <div v-if="assetsInfo" class="hint" :style="assetsInfo.error ? 'color: var(--critical-text)' : ''">{{ assetsInfo.error || assetsInfo.text }}</div>
          <template v-else>
            <button v-for="r in assetRows" :key="r.id" class="gx-asset-row" :class="{off: !r.on}" @click="toggle(r.id)" :aria-pressed="r.on" :title="(r.on ? 'Hide ' : 'Show ') + r.label">
              <img :src="r.url" width="16" height="16" alt=""><span>{{ r.label }}</span><span class="count">{{ totalText(r) }}</span>
            </button>
          </template>
          <div class="gx-hover">
            <template v-if="building">
              <div class="strong small">{{ building.title }}</div>
              <div class="small muted" style="margin-bottom: 4px">{{ building.facts }}<template v-if="building.facts"> · </template>grid {{ building.grid }}</div>
              <div v-for="l in building.lines" :key="l.label" class="gx-asset-line">
                <img :src="building.urls[l.asset.id]" width="14" height="14" alt="">
                <div class="grow"><div class="tt-row"><span>{{ l.label }}</span><span>{{ l.value }}</span></div>
                  <div v-if="l.sub" class="small muted">{{ l.sub }}</div></div>
              </div>
              <div v-if="building.shared" class="small muted" style="margin-top: 3px">{{ building.shared }}</div>
            </template>
            <template v-else-if="rows.length">
              <div class="strong small" style="margin-bottom: 2px">{{ layer.hover.kind === 'transformer' ? 'Transformer' : 'Cable' }}</div>
              <div v-for="r in rows" :key="r[0]" class="tt-row"><span>{{ r[0] }}</span><span>{{ r[1] }}</span></div>
            </template>
            <div v-else class="hint">Hover a building symbol, cable or transformer for details.</div>
          </div>
        </template>
      </div>`,
  };

  function mountOverlay() {
    overlay?.app.unmount();
    overlay?.el.remove();
    const wrap = map.getContainer().parentElement || map.getContainer();
    const el = document.createElement('div');
    wrap.appendChild(el);
    const app = createApp(Legend);
    app.mount(el);
    overlay = { app, el };
  }

  async function loadData() {
    const r = ctx.region();
    const key = state.layer.key;
    if (!state.layer.on || !key) { addLayers(); return; }
    state.layer.loading = true;
    const params = { plz: r.plz, pylovo_version_id: r.version };
    const path = `results/analyses/${encodeURIComponent(key)}`;
    try {
      const [data, assets] = await Promise.all([
        host.api(`${path}/geojson`, { params }),
        host.api(`${path}/assets`, { params }).catch((err) => ({ ...EMPTY, error: err.message })),
      ]);
      if (state.layer.key !== key) return;
      for (const f of assets.features || []) f.properties.n_assets = assetCount(f.properties);
      state.layer.data = markRaw({ ...data, label: state.layer.label || key });
      state.layer.assets = markRaw(assets);
      state.layer.error = null;
    } catch (err) {
      state.layer.data = null;
      state.layer.assets = null;
      state.layer.error = err.message;
    } finally {
      state.layer.loading = false;
    }
    addLayers();
  }

  host.onMapReady((newMap) => {
    map = newMap;
    hovered = null;
    map.on('style.load', addLayers);  // basemap / theme switches replace the style
    map.on('styleimagemissing', onImageMissing);
    map.on('mousemove', onMove);
    map.on('mouseout', () => setHover(null));
    mountOverlay();
    addLayers();
  });
  watch(() => [state.layer.on, state.layer.key, ctx.region().plz, ctx.region().version, state.resultsTick], loadData);
  watch(() => state.layer.assetsOn, applyAssetFilter);
  watch(() => state.layer.gridOn, applyVisibility);
  watch(() => host.store.theme, () => setTimeout(() => { dropImages(); addLayers(); }, 50));
}
