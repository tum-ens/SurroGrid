// Panel "Expansion results": analyses of pylovo's selected region, KPIs, cost and transformer
// loading per grid and model case (ECharts), per-grid table and the map layer toggle.
import { markRaw } from 'vue';
import { CASES, CASE_LABEL, TIMEFRAME_LABEL, analysisLabel, euro } from '../lib.js';

const gridLabel = (row) => `${row.kcid}/${row.bcid}`;
const gridKey = (row) => `${row.plz}/${row.kcid}/${row.bcid}`;

export function createResultsPanel(ctx) {
  const { host, state } = ctx;
  const { format, charts, palette } = host.ui;
  return {
    name: 'GridExpandResults',
    components: { EChart: charts.EChart },
    data() {
      return { analyses: [], loading: false, error: null, groupKey: null, selectedKey: null, gridRows: {}, powerflow: [], pfLoading: false,
        assets: null };
    },
    computed: {
      s() { return state; },
      region() { return ctx.region(); },
      regionKey() { return `${this.region.plz}/${this.region.version}`; },
      groups() {
        const out = new Map();
        for (const a of this.analyses) {
          const key = `${a.scenario}|${a.timeframe_mode}`;
          if (!out.has(key)) out.set(key, { key, scenario: a.scenario, scenario_key: a.scenario_key, timeframe: a.timeframe_mode, items: [] });
          out.get(key).items.push(a);
        }
        return [...out.values()];
      },
      group() { return this.groups.find((g) => g.key === this.groupKey) || this.groups[0] || null; },
      // One entry per case: the status quo (pre case) and the post stage of every post case.
      comparable() {
        const items = this.group?.items || [];
        const pre = items.find((a) => a.stage === 'pre' && a.model_case === 'pre') || items.find((a) => a.stage === 'pre');
        const posts = items.filter((a) => a.stage !== 'pre');
        return [...(pre ? [pre] : []), ...posts];
      },
      selected() { return this.analyses.find((a) => a.analysis_key === this.selectedKey) || null; },
      rows() { return this.gridRows[this.selectedKey] || []; },
      p99() {
        // Max over the grids of the P99 transformer loading: status quo and post stage per case.
        const pf = this.powerflow;
        const pre = pf.filter((r) => r.stage === 'pre' && (r.model_case === 'pre' || !pf.some((x) => x.model_case === 'pre')));
        const out = [{ label: 'Status quo', value: Math.max(...pre.map((r) => r.trafo_loading_p99_time_percent ?? -Infinity)) }];
        for (const c of ['post-hems-heuristic', 'post-hems-optimized', 'post-inflex-heuristic']) {
          const rows = pf.filter((r) => r.stage === 'post' && r.model_case === c);
          if (rows.length) out.push({ label: CASES.find((x) => x.id === c)?.short || c, case: c, value: Math.max(...rows.map((r) => r.trafo_loading_p99_time_percent ?? -Infinity)) });
        }
        return out.filter((x) => Number.isFinite(x.value));
      },
      optCost() {
        void state.status; void host.store.theme;
        const list = this.comparable.filter((a) => this.gridRows[a.analysis_key]);
        const grids = this.gridAxis(list.flatMap((a) => this.gridRows[a.analysis_key]));
        const series = list.map((a, i) => {
          const byGrid = new Map(this.gridRows[a.analysis_key].map((r) => [gridKey(r), r]));
          return charts.bar(analysisLabel(a), grids.map((g) => { const r = byGrid.get(g.key); return r ? Math.round(r.total_cost_eur) : null; }), palette.series(i));
        });
        return charts.baseOption({
          tooltip: { ...charts.baseOption().tooltip, valueFormatter: (v) => euro(v) },
          xAxis: charts.categoryAxis(grids.map((g) => g.label)),
          yAxis: charts.valueAxis('€', { axisLabel: { color: palette.ink('muted'), fontSize: 10.5, formatter: (v) => euro(v) } }),
          series,
        });
      },
      optLoading() {
        void host.store.theme;
        const pf = this.powerflow;
        const grids = this.gridAxis(pf);
        const cases = [{ label: 'Status quo', pick: (r) => r.stage === 'pre' && r.model_case === 'pre' }];
        for (const c of ['post-hems-heuristic', 'post-hems-optimized']) {
          if (pf.some((r) => r.model_case === c && r.stage === 'post')) cases.push({ label: CASE_LABEL[c], pick: (r) => r.stage === 'post' && r.model_case === c });
        }
        const series = cases.map((c, i) => {
          const byGrid = new Map(pf.filter(c.pick).map((r) => [gridKey(r), r]));
          const values = grids.map((g) => { const r = byGrid.get(g.key); return r?.trafo_loading_p99_time_percent != null ? +r.trafo_loading_p99_time_percent.toFixed(1) : null; });
          return charts.bar(c.label, values, palette.series(i), i === 0 ? {
            markLine: { silent: true, symbol: 'none', lineStyle: { color: palette.STATUS.critical, width: 1 }, label: { formatter: 'rated power', color: palette.ink('muted'), fontSize: 10, position: 'insideEndTop' }, data: [{ yAxis: 100 }] },
          } : {});
        });
        return charts.baseOption({
          tooltip: { ...charts.baseOption().tooltip, valueFormatter: (v) => (v === null || v === undefined ? '–' : `${format.num(v, 1)} %`) },
          xAxis: charts.categoryAxis(grids.map((g) => g.label)),
          yAxis: charts.valueAxis('%', { max: (v) => Math.max(110, Math.ceil(v.max / 20) * 20) }),
          series,
        });
      },
      assetKpi() {
        const a = this.assets;
        if (!a || a.stage !== 'post' || !a.buildings) return null;
        const t = a.totals || {};
        const parts = [];
        if (t.pv) parts.push(`PV ${t.pv.power_kw >= 1000 ? format.num(t.pv.power_kw / 1000, 2) + ' MWp' : format.num(t.pv.power_kw, 0) + ' kWp'}`);
        if (t.battery) parts.push(`${format.num(t.battery.energy_kwh, 0)} kWh batteries`);
        if (t.heat_pump) parts.push(`${format.num(t.heat_pump.buildings)} heat pumps`);
        if (t.ev) parts.push(`${format.num(t.ev.units)} EVs`);
        return { value: `${format.num(a.buildings)} buildings`, sub: parts.join(' · ') };
      },
      layerOn: {
        get() { return state.layer.on; },
        set(v) { state.layer.on = v; state.layer.key = this.selectedKey; },
      },
    },
    watch: {
      regionKey: { immediate: true, handler() { this.load(); } },
      's.resultsTick'() { this.load(); },
      's.focusAnalysis'(key) { if (key && this.analyses.some((a) => a.analysis_key === key)) this.select(key); else if (key) this.load(); },
      groupKey() { this.loadGroup(); },
      selectedKey(key) {
        this.loadAssets(key);
        state.layer.key = key;
        const a = this.analyses.find((x) => x.analysis_key === key);
        state.layer.label = a ? analysisLabel(a) : key;
      },
    },
    mounted() { ctx.ensureLoaded(); },
    methods: {
      euro, analysisLabel, num: format.num, km: format.km, relative: format.relative,
      caseLabel(c) { return CASE_LABEL[c] || c || '–'; },
      timeframeLabel(t) { return TIMEFRAME_LABEL[t] || t || '–'; },
      gridAxis(rows) {
        const seen = new Map();
        for (const r of rows) if (!seen.has(gridKey(r))) seen.set(gridKey(r), { key: gridKey(r), label: gridLabel(r), plz: r.plz, kcid: r.kcid, bcid: r.bcid });
        return [...seen.values()].sort((a, b) => a.plz - b.plz || a.kcid - b.kcid || a.bcid - b.bcid);
      },
      async load() {
        const r = this.region;
        if (!r.plz) { this.analyses = []; return; }
        this.loading = true;
        try {
          this.analyses = await host.api('results/analyses', { params: { plz: r.plz, pylovo_version_id: r.version } });
          this.error = null;
          const focus = state.focusAnalysis && this.analyses.find((a) => a.analysis_key === state.focusAnalysis);
          const current = this.analyses.find((a) => a.analysis_key === this.selectedKey);
          const pick = focus || current || this.analyses.find((a) => a.stage === 'post') || this.analyses[0];
          this.groupKey = pick ? `${pick.scenario}|${pick.timeframe_mode}` : null;
          this.selectedKey = pick?.analysis_key || null;
          state.focusAnalysis = null;
          await this.loadGroup();
        } catch (err) {
          this.error = err.message;
          this.analyses = [];
        } finally {
          this.loading = false;
        }
      },
      async loadGroup() {
        const g = this.group;
        const r = this.region;
        if (!g) { this.gridRows = {}; this.powerflow = []; return; }
        if (!g.items.some((a) => a.analysis_key === this.selectedKey)) this.selectedKey = (g.items.find((a) => a.stage === 'post') || g.items[0]).analysis_key;
        const params = { plz: r.plz, pylovo_version_id: r.version };
        this.pfLoading = true;
        try {
          const [rows, pf] = await Promise.all([
            Promise.all(g.items.map((a) => host.api(`results/analyses/${encodeURIComponent(a.analysis_key)}/grids`, { params }).then((x) => [a.analysis_key, x]))),
            host.api('results/powerflow', { params: { ...params, scenario_key: g.scenario_key } }),
          ]);
          this.gridRows = markRaw(Object.fromEntries(rows));
          this.powerflow = markRaw(pf.filter((x) => x.mode === 'summary' || x.mode === null));
        } catch (err) {
          host.ui.errorToast('GridExpand results', err);
        } finally {
          this.pfLoading = false;
        }
      },
      async loadAssets(key) {
        this.assets = null;
        const a = this.analyses.find((x) => x.analysis_key === key);
        if (!a || a.stage !== 'post') return;
        const r = this.region;
        try {
          const data = await host.api(`results/analyses/${encodeURIComponent(key)}/assets`, { params: { plz: r.plz, pylovo_version_id: r.version, features: false } });
          if (this.selectedKey === key) this.assets = markRaw(data);
        } catch (err) { /* the map legend shows asset errors */ }
      },
      select(key) {
        const a = this.analyses.find((x) => x.analysis_key === key);
        if (!a) return;
        const group = `${a.scenario}|${a.timeframe_mode}`;
        if (group !== this.groupKey) this.groupKey = group;
        this.selectedKey = key;
      },
      zoom() { if (state.layer.data?.bounds) host.actions.flyTo(state.layer.data.bounds, 18); },
      inspect(row) { if (row.pylovo_grid_result_id) host.actions.selectGrid(row.pylovo_grid_result_id); },
    },
    template: `
    <div class="panel gx-panel">
      <div class="panel-toolbar nowrap">
        <div class="panel-title"><Icon name="gauge" :size="15"/>Expansion results</div>
        <select v-if="groups.length > 1" class="select sm" v-model="groupKey" aria-label="Scenario and timeframe">
          <option v-for="g in groups" :key="g.key" :value="g.key">{{ g.scenario }} · {{ timeframeLabel(g.timeframe) }}</option>
        </select>
        <span class="grow"></span>
        <Toggle v-model="layerOn" label="Map layer" :disabled="!selected"/>
        <button class="btn ghost icon sm" title="Zoom to the analysed grids" :disabled="!s.layer.data" @click="zoom"><Icon name="fit" :size="14"/></button>
        <button class="btn ghost icon sm" title="Reload" @click="load"><Icon name="refresh" :size="14"/></button>
      </div>
      <div class="panel-body stack">
        <EmptyState v-if="!region.plz" icon="pin" title="No region selected" text="Select a PLZ in pylovo's Regions or Results step; its GridExpand analyses appear here."/>
        <EmptyState v-else-if="loading && !analyses.length" loading title="Loading analyses…"/>
        <EmptyState v-else-if="error" error title="Results could not be loaded" :text="error"/>
        <EmptyState v-else-if="!analyses.length" icon="chart" title="No expansion analyses yet" :text="'Run the pipeline for PLZ ' + region.plz + ' (pylovo v' + region.version + ') in the GridExpand runs panel.'"/>
        <template v-else>
          <div class="small muted">PLZ {{ region.plz }} · pylovo v{{ region.version }} · {{ group.scenario }} · {{ timeframeLabel(group.timeframe) }}</div>
          <div class="kpis" v-if="selected">
            <Kpi label="Expansion cost" icon="bolt" :value="euro(selected.total_cost_eur)" :sub="'cables ' + euro(selected.line_cost_eur) + ' · transformers ' + euro(selected.transformer_cost_eur)"/>
            <Kpi label="Cables to reinforce" icon="cable" :value="num(selected.lines_to_reinforce)" :sub="km(selected.km_to_reinforce) + ' · +' + num(selected.additional_cables) + ' parallel'"/>
            <Kpi label="Transformers to reinforce" icon="transformer" :value="num(selected.transformers_to_reinforce) + ' / ' + num(selected.transformers_total)" :sub="'+' + num(selected.additional_transformer_kva) + ' kVA'"/>
            <Kpi v-if="assetKpi" label="Building assets" icon="building" :value="assetKpi.value" :sub="assetKpi.sub" title="Buildings with PV, battery, heat pump or EV in this case (installed capacities of Step 3)"/>
            <Kpi label="P99 transformer loading" icon="gauge" :value="p99.length ? num(p99[0].value, 0) + ' %' : '–'" :title="'Maximum over the grids of the 99th percentile of the hourly transformer loading'">
              <span>status quo<template v-for="x in p99.slice(1)" :key="x.label"> · <strong>{{ num(x.value, 0) }} %</strong> {{ x.label }}</template></span>
            </Kpi>
          </div>
          <div class="table-wrap gx-table" style="max-height: 190px">
            <table class="table">
              <thead><tr><th>Analysis</th><th>Stage</th><th class="num">Grids</th><th class="num">Cost</th><th class="num">Cables</th><th class="num">Trafos</th><th class="num">Max trafo</th><th>Created</th></tr></thead>
              <tbody>
                <tr v-for="a in group.items" :key="a.analysis_key" class="clickable" :class="{selected: a.analysis_key === selectedKey}" @click="select(a.analysis_key)" :title="a.analysis_key + ' · ' + a.run_name">
                  <td>{{ analysisLabel(a) }}</td><td>{{ a.stage }}</td><td class="num">{{ a.grids }}</td><td class="num">{{ euro(a.total_cost_eur) }}</td>
                  <td class="num">{{ a.lines_to_reinforce }}</td><td class="num">{{ a.transformers_to_reinforce }}</td>
                  <td class="num">{{ a.max_transformer_loading_percent !== null ? num(a.max_transformer_loading_percent, 0) + ' %' : '–' }}</td><td class="muted">{{ relative(a.created_at) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="charts gx-charts">
            <div class="chart-card"><h4>Expansion cost per grid</h4><div class="card-sub">cables + transformers, per case (post stage)</div>
              <EChart :option="optCost" :height="210"/></div>
            <div class="chart-card"><h4>P99 transformer loading per grid</h4><div class="card-sub">status quo vs. post stage of each case</div>
              <EChart :option="optLoading" :height="210"/></div>
          </div>
          <div v-if="rows.length" class="table-wrap gx-table">
            <table class="table">
              <thead><tr><th>Grid</th><th class="num">Cost</th><th class="num">Cables</th><th class="num">Length</th><th class="num">Trafo loading</th><th class="num">Rated → required</th><th>Transformer measure</th></tr></thead>
              <tbody>
                <tr v-for="r in rows" :key="r.grid_case_id" class="clickable" @click="inspect(r)" title="Open the grid in pylovo's inspector">
                  <td>{{ r.kcid }}/{{ r.bcid }}</td><td class="num">{{ euro(r.total_cost_eur) }}</td><td class="num">{{ r.lines_to_reinforce }}</td><td class="num">{{ km(r.km_to_reinforce) }}</td>
                  <td class="num" :style="{color: r.transformer_loading_percent > 100 ? 'var(--critical-text)' : null}">{{ r.transformer_loading_percent !== null ? num(r.transformer_loading_percent, 0) + ' %' : '–' }}</td>
                  <td class="num">{{ num(r.transformer_rated_power_kva) }} → {{ num(r.required_transformer_kva) }} kVA</td>
                  <td class="muted small">{{ (r.transformer_cost_basis || '').replace(/_/g, ' ') }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </template>
      </div>
    </div>`,
  };
}
