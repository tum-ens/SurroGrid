// Expansion result layer on pylovo's MapLibre map: cables coloured by the required action,
// transformers by their critical loading, with a legend card that shows the hovered feature.
import { createApp, markRaw, watch } from 'vue';
import { euro } from './lib.js';

const SOURCES = { lines: 'gridexpand-lines', trafos: 'gridexpand-trafos' };
const LAYERS = ['gridexpand-lines-casing', 'gridexpand-lines', 'gridexpand-trafos'];
const PICK = ['gridexpand-trafos', 'gridexpand-lines'];
const EMPTY = { type: 'FeatureCollection', features: [] };

export function installMapLayer(ctx) {
  const { host, state } = ctx;
  const { palette, format, iconSvg } = host.ui;
  let map = null;
  let overlay = null;
  let hovered = null;

  function colors() {
    const dark = palette.mode() === 'dark';
    return { none: palette.ink(dark ? 'neutral' : 'muted'), add1: palette.STATUS.serious, add2: palette.STATUS.critical,
      casing: dark ? '#0b0d12' : '#ffffff' };
  }
  function loadingColor() {
    return ['interpolate', ['linear'], ['coalesce', ['get', 'loading_percent'], 0], ...palette.LOADING_STOPS.flat()];
  }
  function removeLayers() {
    if (!map?.style) return;
    for (const id of [...LAYERS].reverse()) if (map.getLayer(id)) map.removeLayer(id);
    for (const id of Object.values(SOURCES)) if (map.getSource(id)) map.removeSource(id);
  }
  function addLayers() {
    if (!map || !map.isStyleLoaded()) return;
    removeLayers();
    const data = state.layer.data;
    if (!state.layer.on || !data) return;
    const c = colors();
    map.addSource(SOURCES.lines, { type: 'geojson', data: data.lines || EMPTY });
    map.addSource(SOURCES.trafos, { type: 'geojson', data: data.transformers || EMPTY });
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
  }
  function setHover(feature) {
    const id = feature ? `${feature.layer.id}:${feature.id}` : null;
    if (hovered?.key === id) return;
    if (hovered && map.getSource(hovered.source)) map.setFeatureState({ source: hovered.source, id: hovered.id }, { hover: false });
    hovered = null;
    state.layer.hover = null;
    if (!feature) return;
    const source = feature.layer.id === 'gridexpand-trafos' ? SOURCES.trafos : SOURCES.lines;
    hovered = { key: id, source, id: feature.id };
    map.setFeatureState({ source, id: feature.id }, { hover: true });
    state.layer.hover = { kind: source === SOURCES.trafos ? 'transformer' : 'cable', props: { ...feature.properties } };
  }
  function onMove(e) {
    if (!state.layer.on || !map.getLayer(PICK[0])) return;
    const box = [[e.point.x - 4, e.point.y - 4], [e.point.x + 4, e.point.y + 4]];
    const hits = map.queryRenderedFeatures(box, { layers: PICK.filter((id) => map.getLayer(id)) });
    // Transformers first; of overlapping cables the one with the largest required reinforcement.
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
        return [['Grid', `${p.kcid}/${p.bcid}`], ['Cable', `${p.std_type || '–'} · ${format.meters((p.length_km || 0) * 1000)}`],
          ['Critical loading', pct(p.loading_percent)], ['Parallel cables', `${p.required_parallel} required (+${p.additional_parallel})`],
          ...(p.reinforcement ? [['Reinforcement', p.reinforcement]] : []), ['Cost', euro(p.cost_eur)]];
      },
    },
    methods: { icon: (name) => iconSvg(name, 13), hide() { state.layer.on = false; } },
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
          <div class="legend-row"><span class="sw line" :style="{background: c.none}"></span>No reinforcement<span class="count">{{ counts.none }}</span></div>
          <div class="legend-row"><span class="sw line" :style="{background: c.add1, height: '4px'}"></span>+1 parallel cable<span class="count">{{ counts.add1 }}</span></div>
          <div class="legend-row"><span class="sw line" :style="{background: c.add2, height: '5px'}"></span>+2 or more cables<span class="count">{{ counts.add2 }}</span></div>
          <div class="legend-title" style="margin-top: 8px">Transformer loading (critical hour)</div>
          <div class="legend-ramp" :style="{background: ramp}"></div>
          <div class="legend-scale"><span>0 %</span><span>50</span><span>75</span><span>100</span><span>130 %</span></div>
          <div class="legend-row" style="margin-top: 4px"><span class="sw round" :style="{background: 'transparent', boxShadow: 'inset 0 0 0 2.5px ' + c.add2}"></span>needs a larger transformer</div>
          <div class="gx-hover">
            <template v-if="rows.length">
              <div class="strong small" style="margin-bottom: 2px">{{ layer.hover.kind === 'transformer' ? 'Transformer' : 'Cable' }}</div>
              <div v-for="r in rows" :key="r[0]" class="tt-row"><span>{{ r[0] }}</span><span>{{ r[1] }}</span></div>
            </template>
            <div v-else class="hint">Hover a cable or transformer for details.</div>
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
    try {
      const data = await host.api(`results/analyses/${encodeURIComponent(key)}/geojson`, { params: { plz: r.plz, pylovo_version_id: r.version } });
      if (state.layer.key !== key) return;
      state.layer.data = markRaw({ ...data, label: state.layer.label || key });
      state.layer.error = null;
    } catch (err) {
      state.layer.data = null;
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
    map.on('mousemove', onMove);
    map.on('mouseout', () => setHover(null));
    mountOverlay();
    addLayers();
  });
  watch(() => [state.layer.on, state.layer.key, ctx.region().plz, ctx.region().version, state.resultsTick], loadData);
  watch(() => host.store.theme, () => setTimeout(addLayers, 50));
}
