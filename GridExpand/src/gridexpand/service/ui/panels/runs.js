// Panel "GridExpand runs": run the full pipeline for the grid selected in pylovo (case studies),
// for several grids of the region, or prepare a whole-PLZ run for a terminal (tmux); pipeline
// jobs with progress per case and grid, and the live log (SSE).
import { CASES, TIMEFRAMES, TIMEFRAME_LABEL } from '../lib.js';

const ROW = 18;
const FILTERS = [{ value: 'all', label: 'All' }, { value: 'main', label: 'Pipeline' }, { value: 'warn', label: 'Warnings' }];
const MODES = [{ value: 'grid', label: 'This grid' }, { value: 'grids', label: 'Several grids' }, { value: 'terminal', label: 'Whole PLZ' }];
const ALL_CASES = CASES.map((c) => c.id);

export function createRunsPanel(ctx) {
  const { host, state } = ctx;
  const { format } = host.ui;
  return {
    name: 'GridExpandRuns',
    data() {
      return {
        tab: 'run', mode: 'grid', CASES, TIMEFRAMES, FILTERS, MODES,
        grids: null, gridsError: null, gridsLoading: false, selected: [], allGrids: null,
        scenario: null, scenarioTouched: false, cases: [...ALL_CASES], casesTouched: false,
        timeframe: 'max_base_electricity_demand_week', minBuildings: 5, starting: false,
        terminal: null, terminalRuns: [], preparing: false,
        filter: 'all', follow: true, scrollTop: 0, height: 300, detail: null, detailAt: 0,
      };
    },
    computed: {
      s() { return state; },
      region() { return ctx.region(); },
      regionKey() { const r = this.region; return `${r.plz}/${r.version}/${this.minBuildings}`; },
      db() { return state.status?.database || null; },
      dbDot() { if (state.statusError) return 'bad'; if (!this.db) return ''; return this.db.connected ? (this.db.schema?.surrogrid_ready ? 'good' : 'warn') : 'bad'; },
      dbLabel() { if (state.statusError) return 'service offline'; return this.db ? `${this.db.database} @ ${this.db.host}:${this.db.port}` : 'connecting…'; },
      dbTitle() { return state.statusError || this.db?.error || 'Database of the GridExpand service (its .env)'; },
      solvers() { return state.status?.solvers || null; },
      postOk() { return !this.solvers || this.solvers.post_cases_supported; },
      scenarioInfo() { return state.scenarios.find((x) => x.name === this.scenario) || null; },
      adoption() {
        const a = this.scenarioInfo?.adoption;
        if (!a) return '';
        const share = (x) => (x.building_share === null || x.building_share === undefined ? x.mode.replace(/_/g, ' ') : `${Math.round(x.building_share * 100)} %`);
        return `heat ${share(a.heat)} · EV ${share(a.mobility)} · PV ${share(a.pv_battery)}`;
      },
      candidates() { return this.grids?.candidates || []; },
      agsText() { return this.grids?.ags ? String(this.grids.ags).padStart(8, '0') : '…'; },
      sortedSelection() { return [...this.selected].sort((a, b) => a - b); },
      contiguous() { const v = this.sortedSelection; return !v.length || v[v.length - 1] - v[0] + 1 === v.length; },
      pylovoCandidate() { return this.candidates.find((c) => c.grid_result_id === this.region.gridId) || null; },
      // The grid of pylovo's inspector (any size), with its building count and existing results.
      pylovoGrid() {
        const g = host.store.grid.detail?.grid;
        if (!g || !this.region.gridId) return null;
        const c = (this.allGrids?.candidates || []).find((x) => x.grid_result_id === g.grid_result_id);
        return { ...g, n_buildings: c?.n_buildings ?? null, results: c?.results || [], known: !!c };
      },
      casesOk() { return this.cases.length > 0 && (this.postOk || this.cases.every((c) => c === 'pre')); },
      formOk() { return !!this.region.plz && !!this.region.version && !!this.scenarioInfo?.valid && this.casesOk; },
      canStart() {
        if (this.starting || !this.formOk) return false;
        if (this.mode === 'grid') return !!this.pylovoGrid;
        return this.selected.length > 0 && this.contiguous;
      },
      plzGrids() { return (this.allGrids?.candidates || []).length; },
      job() { return state.jobs.find((j) => j.id === state.activeJobId) || null; },
      activeCount() { return state.jobs.filter((j) => j.status === 'running' || j.status === 'queued').length; },
      logLines() {
        void state.logTick;
        const all = ctx.lines.get(state.activeJobId) || [];
        if (this.filter === 'all') return all;
        if (this.filter === 'warn') return all.filter((l) => l.level === 'warning' || l.level === 'error');
        return all.filter((l) => l.level !== 'debug' && l.level !== 'trace');
      },
      total() { void state.logTick; return this.logLines.length; },
      first() { return Math.max(0, Math.floor(this.scrollTop / ROW) - 20); },
      visible() { void state.logTick; return this.logLines.slice(this.first, this.first + Math.ceil(this.height / ROW) + 40); },
      analysisKeys() {
        return Object.values(this.detail?.summaries || {}).flatMap((sum) => (sum?.materialized_expansion || []).map((m) => m.analysis_key));
      },
    },
    watch: {
      regionKey: { immediate: true, handler() { this.loadGrids(); } },
      's.resultsTick'() { this.loadGrids(); this.loadDetail(true); },
      's.activeJobId'() { this.follow = true; this.detail = null; this.loadDetail(true); this.$nextTick(this.toBottom); },
      's.scenarios'() { this.pickScenario(); },
      's.focusScenario'(name) { if (name) this.focusScenario(name); },
      solvers(v) { if (v && !v.post_cases_supported && !this.casesTouched) this.cases = ['pre']; },
      mode(m) { if (m === 'terminal') this.loadTerminalRuns(); },
      total() { if (this.follow) this.$nextTick(this.toBottom); },
      'job.phase'() { this.loadDetail(); },
      'job.status'() { this.loadDetail(true); },
      tab(t) { if (t === 'jobs') this.$nextTick(() => { this.measure(); this.toBottom(); }); },
    },
    mounted() {
      ctx.ensureLoaded();
      this.pickScenario();
      this.ro = new ResizeObserver(() => this.measure());
      this.$watch(() => this.$refs.body, (el) => { if (el) this.ro.observe(el); }, { immediate: true });
    },
    beforeUnmount() { this.ro?.disconnect(); },
    methods: {
      num: format.num, duration: format.duration, relative: format.relative,
      measure() { this.height = this.$refs.body?.clientHeight || 300; },
      reload() { ctx.loadStatus(); ctx.loadScenarios(); ctx.loadJobs(); this.loadGrids(); if (this.mode === 'terminal') this.loadTerminalRuns(); },
      focusScenario(name) {
        if (!state.scenarios.some((x) => x.name === name)) return;
        this.scenario = name;
        this.scenarioTouched = true;
        state.focusScenario = null;
      },
      editScenario() {
        state.editScenario = this.scenario;
        host.ui.focusPanel('gridexpand.scenarios');
      },
      pickScenario() {
        if (this.scenarioTouched && state.scenarios.some((x) => x.name === this.scenario && x.valid)) return;
        const keys = new Set(this.candidates.flatMap((c) => c.results.map((r) => r.scenario)));
        const usable = state.scenarios.filter((x) => x.valid && !x.template);
        const used = usable.find((x) => keys.has(x.scenario_key));
        this.scenario = (used || usable[0] || state.scenarios.find((x) => x.valid) || {}).name || null;
      },
      async loadGrids() {
        const r = this.region;
        if (!r.plz || !r.version) { this.grids = null; this.allGrids = null; this.selected = []; return; }
        const key = this.regionKey;
        this.gridsLoading = true;
        try {
          const params = { plz: r.plz, pylovo_version_id: r.version };
          const [data, all] = await Promise.all([
            host.api('grids', { params: { ...params, min_buildings: this.minBuildings } }),
            this.minBuildings === 1 ? null : host.api('grids', { params: { ...params, min_buildings: 1 } }),
          ]);
          if (key !== this.regionKey) return;
          this.allGrids = all || data;
          const keep = this.grids && this.grids.plz === data.plz && this.grids.pylovo_version_id === data.pylovo_version_id;
          this.grids = data;
          this.gridsError = null;
          const ids = data.candidates.map((c) => c.candidate_index);
          this.selected = keep ? this.selected.filter((i) => ids.includes(i)) : ids;
          this.pickScenario();
        } catch (err) {
          this.grids = null;
          this.gridsError = err.message;
        } finally {
          this.gridsLoading = false;
        }
      },
      caseBadges(candidate) {
        const key = this.scenarioInfo?.scenario_key;
        const rows = candidate.results.filter((r) => !key || r.scenario === key);
        return [...new Set(rows.map((r) => CASES.find((c) => c.id === r.model_case)?.short || r.model_case).filter(Boolean))];
      },
      toggle(index) {
        this.selected = this.selected.includes(index) ? this.selected.filter((i) => i !== index) : [...this.selected, index];
      },
      selectAll() { this.selected = this.candidates.map((c) => c.candidate_index); },
      selectPylovoGrid() { if (this.pylovoCandidate) this.selected = [this.pylovoCandidate.candidate_index]; },
      setCase(id, on) {
        this.casesTouched = true;
        this.cases = on ? ALL_CASES.filter((c) => c === id || this.cases.includes(c)) : this.cases.filter((c) => c !== id);
      },
      requestBody(scope) {
        const r = this.region;
        const body = { plz: r.plz, pylovo_version_id: r.version, scenario: this.scenario, model_cases: this.cases,
          timeframe_mode: this.timeframe, min_buildings: this.minBuildings };
        if (this.grids?.ags) body.ags = this.grids.ags;
        if (scope === 'grid') body.grid_result_id = this.pylovoGrid.grid_result_id;
        else if (scope === 'grids' && this.selected.length !== this.candidates.length) body.candidate_indexes = this.sortedSelection;
        return body;
      },
      async start() {
        const body = this.requestBody(this.mode);
        this.starting = true;
        try {
          const job = await ctx.startPipeline(body);
          if (job) this.tab = 'jobs';
        } finally {
          this.starting = false;
        }
      },
      async prepareTerminal() {
        this.preparing = true;
        try {
          this.terminal = await host.api('jobs/terminal', { method: 'POST', body: this.requestBody('plz') });
          this.loadTerminalRuns();
        } catch (err) {
          host.ui.errorToast('Could not prepare the terminal run', err);
        } finally {
          this.preparing = false;
        }
      },
      async loadTerminalRuns() {
        try { this.terminalRuns = await host.api('jobs/terminal'); } catch (err) { this.terminalRuns = []; }
      },
      async copy(text) {
        try { await navigator.clipboard.writeText(text); host.ui.toast('good', 'Copied', text.length > 70 ? text.slice(0, 70) + '…' : text, 2500); }
        catch (err) { host.ui.toast('warn', 'Copy failed', 'Select the text and copy it by hand.'); }
      },
      download(name, text) {
        const url = URL.createObjectURL(new Blob([text], { type: 'text/yaml' }));
        const a = Object.assign(document.createElement('a'), { href: url, download: name });
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      },
      terminalBadge(status) { return { done: 'good', succeeded: 'good', failed: 'bad', running: 'accent', interrupted: 'warn', cancelled: 'warn' }[status] || ''; },
      resultBadges(results) {
        const key = this.scenarioInfo?.scenario_key;
        return [...new Set(results.filter((r) => !key || r.scenario === key).map((r) => CASES.find((c) => c.id === r.model_case)?.short || r.model_case).filter(Boolean))];
      },
      pick(id) { ctx.selectJob(id); },
      async cancel() {
        const job = this.job;
        if (!job) return;
        if (job.status === 'running') {
          const ok = await host.ui.confirmDialog({
            title: 'Cancel this GridExpand job?', danger: true, confirmText: 'Cancel job',
            body: 'The runner and every step it started are stopped (SIGTERM to the process group). Finished grids keep their results; '
              + 'a grid stopped during its power flow keeps its earlier results (Step 4 swaps a new run in only when it is complete).',
          });
          if (!ok) return;
        }
        ctx.cancelJob(job.id);
      },
      async loadDetail(force = false) {
        const id = state.activeJobId;
        if (!id || (!force && Date.now() - this.detailAt < 1500)) return;
        this.detailAt = Date.now();
        try { const d = await host.api(`jobs/${id}`); if (state.activeJobId === id) this.detail = d; } catch (err) { /* ignore */ }
      },
      stepGrids(step) { return this.detail?.grids?.[step.name] || []; },
      showResults() {
        const keys = this.analysisKeys;
        state.focusAnalysis = keys.length ? keys[keys.length - 1] : null;
        host.ui.focusPanel('gridexpand.results');
      },
      onScroll(e) {
        const el = e.target;
        this.scrollTop = el.scrollTop;
        this.follow = el.scrollHeight - el.scrollTop - el.clientHeight < ROW * 2;
      },
      toBottom() {
        const el = this.$refs.body;
        if (!el) return;
        el.scrollTop = el.scrollHeight;
        this.scrollTop = el.scrollTop;
      },
      icon(status) { return { running: 'clock', queued: 'clock', succeeded: 'check', failed: 'alert', cancelled: 'stop' }[status] || 'info'; },
      stepBadge(status) { return { succeeded: 'good', failed: 'bad', running: 'accent', cancelled: 'warn', skipped: '', pending: '' }[status] || ''; },
      logHref() { return host.url(`api/jobs/${this.job.id}/log?format=text`); },
      fileHref(step, row) { return host.url(`api/jobs/${this.job.id}/files/${step.name}/logs/${encodeURIComponent(row.log)}`); },
      timeframeLabel(id) { return TIMEFRAME_LABEL[id] || id; },
    },
    template: `
    <div class="panel gx-panel">
      <div class="panel-toolbar nowrap">
        <div class="panel-title"><Icon name="play" :size="15"/>GridExpand runs</div>
        <span class="grow"></span>
        <span class="chip" :title="dbTitle"><span class="dot" :class="dbDot"></span><span class="label mono" style="font-size: 11.5px">{{ dbLabel }}</span></span>
        <button class="btn ghost icon sm" title="Reload" @click="reload"><Icon name="refresh" :size="14"/></button>
      </div>
      <div class="tabs">
        <button :class="{on: tab === 'run'}" @click="tab = 'run'"><Icon name="sliders" :size="13"/>New run</button>
        <button :class="{on: tab === 'jobs'}" @click="tab = 'jobs'"><Icon name="terminal" :size="13"/>Jobs
          <span v-if="activeCount" class="badge accent">{{ activeCount }}</span></button>
      </div>

      <div v-if="tab === 'run'" class="panel-body stack">
        <div v-if="s.statusError" class="callout bad"><Icon name="alert" :size="15"/><div>The GridExpand service is not reachable: {{ s.statusError }}</div></div>
        <div class="card sunken gx-region">
          <div class="grow">
            <div class="section-title" style="margin: 0 0 2px">Region from pylovo</div>
            <div v-if="region.plz" class="strong">PLZ {{ region.plz }} <span class="muted" style="font-weight: 500">· AGS {{ agsText }} · pylovo v{{ region.version || '–' }}</span></div>
            <div v-else class="muted small">Select a PLZ in pylovo's Regions or Results step.</div>
          </div>
          <div class="stat" style="min-width: 76px"><div class="s-label">grids</div><div class="s-value">{{ gridsLoading ? '…' : plzGrids }}</div></div>
        </div>

        <template v-if="region.plz">
          <Seg v-model="mode" :options="MODES"/>

          <template v-if="mode === 'grid'">
            <div v-if="pylovoGrid" class="card gx-grid-card">
              <div class="gx-grid-icon"><Icon name="grid" :size="18"/></div>
              <div class="grow">
                <div class="strong">Grid {{ pylovoGrid.kcid }}/{{ pylovoGrid.bcid }} <span class="muted" style="font-weight: 500">· PLZ {{ pylovoGrid.plz }}</span></div>
                <div class="small muted">{{ pylovoGrid.n_buildings !== null ? num(pylovoGrid.n_buildings) + ' buildings · ' : '' }}{{ pylovoGrid.size_label || (pylovoGrid.kva ? pylovoGrid.kva + ' kVA' : '') }}</div>
                <div v-if="resultBadges(pylovoGrid.results).length" class="gx-grid-chips"><span class="small muted">results:</span>
                  <span v-for="b in resultBadges(pylovoGrid.results)" :key="b" class="badge good">{{ b }}</span></div>
              </div>
            </div>
            <div v-else class="callout info"><Icon name="info" :size="15"/><div>Select a grid on pylovo's map (click one of its cables or its station) or in the Results list: this panel runs the full GridExpand pipeline for exactly that grid.</div></div>
          </template>

          <template v-else-if="mode === 'grids'">
            <div class="row between" style="margin-bottom: -4px">
              <div class="section-title">Candidate grids</div>
              <div class="row" style="gap: 4px">
                <button class="btn ghost sm" @click="selectAll" :disabled="!candidates.length">All</button>
                <button class="btn ghost sm" @click="selectPylovoGrid" :disabled="!pylovoCandidate" title="Only the grid selected in pylovo's grid inspector">Inspector grid</button>
              </div>
            </div>
            <div v-if="gridsError" class="callout bad"><Icon name="alert" :size="15"/><div>{{ gridsError }}</div></div>
            <div v-else class="table-wrap gx-table">
              <table class="table">
                <thead><tr><th style="width: 26px"></th><th class="num">#</th><th>Grid</th><th class="num">Buildings</th><th>Results ({{ scenarioInfo ? scenarioInfo.id : 'all scenarios' }})</th></tr></thead>
                <tbody>
                  <tr v-for="c in candidates" :key="c.candidate_index" class="clickable" :class="{selected: selected.includes(c.candidate_index)}" @click="toggle(c.candidate_index)">
                    <td><input type="checkbox" :checked="selected.includes(c.candidate_index)" @click.stop="toggle(c.candidate_index)" :aria-label="'Grid ' + c.kcid + '/' + c.bcid"></td>
                    <td class="num mono">{{ c.candidate_index }}</td>
                    <td>{{ c.kcid }}/{{ c.bcid }} <span v-if="c.grid_result_id === region.gridId" class="badge accent" title="Selected in pylovo">inspector</span></td>
                    <td class="num">{{ num(c.n_buildings) }}</td>
                    <td><span v-for="b in caseBadges(c)" :key="b" class="badge good" style="margin-right: 3px">{{ b }}</span><span v-if="!caseBadges(c).length" class="muted small">–</span></td>
                  </tr>
                  <tr v-if="!candidates.length && !gridsLoading"><td colspan="5" class="muted" style="text-align: center; padding: 14px">No grid of this PLZ and version has at least {{ minBuildings }} buildings.</td></tr>
                </tbody>
              </table>
            </div>
            <div v-if="!contiguous" class="hint" style="color: var(--warn-text)">Select consecutive grids: the runner runs one range of candidate numbers.</div>
          </template>

          <div v-else class="callout info"><Icon name="terminal" :size="15"/><div>All {{ plzGrids }} grids of PLZ {{ region.plz }} take long (minutes per grid and case for a week, much longer for a full year). Prepare the run here and start it in a terminal with <span class="mono">tmux</span>: it survives a closed browser and a restarted service, writes to the same database, and its results appear in Expansion results. Its progress is listed below.</div></div>

          <div class="form-grid gx-form">
            <div class="form-field" style="grid-column: 1 / -1">
              <label class="ff-label" for="gx-scenario">Scenario</label>
              <div class="row" style="gap: 6px">
                <select id="gx-scenario" class="select grow" v-model="scenario" @change="scenarioTouched = true">
                  <option v-for="x in s.scenarios" :key="x.name" :value="x.name" :disabled="!x.valid">{{ x.name }}{{ x.user ? ' (yours)' : '' }}{{ x.template ? ' (template)' : '' }}{{ x.valid ? '' : ' (invalid)' }}</option>
                </select>
                <button class="btn sm" @click="editScenario" :disabled="!scenario" title="Adjust the scenario in the scenario editor and save it as your own"><Icon name="sliders" :size="13"/>Edit…</button>
              </div>
              <div v-if="scenarioInfo && scenarioInfo.valid" class="ff-help">{{ scenarioInfo.id }} · {{ scenarioInfo.milestone_year }} · heat {{ scenarioInfo.heat_source }} · {{ adoption }} · <span class="mono" :title="scenarioInfo.configuration_hash">{{ scenarioInfo.configuration_hash.slice(0, 12) }}</span></div>
              <div v-else-if="scenarioInfo" class="ff-help" style="color: var(--critical-text)">{{ scenarioInfo.error }}</div>
            </div>
            <div class="form-field">
              <label class="ff-label" for="gx-timeframe">Timeframe</label>
              <select id="gx-timeframe" class="select" v-model="timeframe"><option v-for="t in TIMEFRAMES" :key="t.id" :value="t.id">{{ t.label }}</option></select>
            </div>
            <div v-if="mode === 'grids'" class="form-field">
              <label class="ff-label" for="gx-minb">Min. buildings per grid</label>
              <input id="gx-minb" class="input" type="number" min="1" v-model.number="minBuildings" @change="minBuildings = Math.max(1, Math.round(minBuildings || 1))">
            </div>
          </div>
          <div class="form-field">
            <div class="ff-label">Model cases</div>
            <div class="row wrap" style="gap: 14px">
              <Toggle v-for="c in CASES" :key="c.id" :model-value="cases.includes(c.id)" @update:modelValue="(v) => setCase(c.id, v)" :disabled="c.post && !postOk" :label="c.label"/>
            </div>
            <div v-if="!postOk" class="ff-help" style="color: var(--warn-text)">Post cases need the Step 3 solver: {{ solvers.post_cases_reason }}</div>
          </div>

          <template v-if="mode === 'terminal'">
            <div class="row between">
              <div class="small muted">{{ plzGrids }} grid(s) × {{ cases.length }} case(s) · {{ timeframeLabel(timeframe) }}</div>
              <button class="btn primary" :disabled="preparing || !formOk" @click="prepareTerminal"><Spinner v-if="preparing" :size="13"/><Icon v-else name="terminal" :size="13"/>Prepare terminal run</button>
            </div>
            <div v-if="terminal" class="card gx-terminal">
              <div class="row between"><div class="strong small">{{ terminal.title }}</div>
                <span class="row" style="gap: 4px">
                  <button class="btn ghost sm" @click="download(terminal.run_id + '.yaml', terminal.portable_run_yaml)" title="Run YAML for another GridExpand checkout (config/runs/)"><Icon name="download" :size="13"/>Run YAML</button>
                </span></div>
              <div v-for="c in terminal.commands" :key="c.label" class="gx-cmd">
                <div class="small muted">{{ c.label }}</div>
                <div class="gx-cmd-line"><code class="mono">{{ c.command }}</code><button class="btn ghost icon sm" title="Copy" @click="copy(c.command)"><Icon name="copy" :size="13"/></button></div>
              </div>
              <div class="hint">Run directory <span class="mono">{{ terminal.run_dir }}</span></div>
            </div>
            <div v-if="terminalRuns.length" class="stack tight">
              <div class="section-title" style="margin-bottom: 0">Terminal runs</div>
              <div v-for="t in terminalRuns" :key="t.run_id" class="gx-trun">
                <span class="badge" :class="terminalBadge(t.status)">{{ t.status }}</span>
                <span class="grow ellipsis small" :title="t.run_id">{{ t.title }}</span>
                <span v-if="t.jobs && t.jobs.total" class="small muted num">{{ t.jobs.done || 0 }}/{{ t.jobs.total }} jobs<template v-if="t.jobs.failed"> · <span style="color: var(--critical-text)">{{ t.jobs.failed }} failed</span></template></span>
                <span class="small muted">{{ relative(t.updated_at || t.prepared_at) }}</span>
              </div>
            </div>
          </template>
          <template v-else>
            <div class="row between">
              <div class="small muted">{{ mode === 'grid' ? (pylovoGrid ? 'grid ' + pylovoGrid.kcid + '/' + pylovoGrid.bcid : 'no grid selected') : selected.length + ' grid(s)' }} × {{ cases.length }} case(s) · {{ timeframeLabel(timeframe) }}</div>
              <button class="btn primary" :disabled="!canStart" @click="start"><Spinner v-if="starting" :size="13"/><Icon v-else name="play" :size="13"/>{{ mode === 'grid' ? 'Run full pipeline' : 'Start run' }}</button>
            </div>
            <div class="hint">Runs <span class="mono">gridexpand run</span> once per case: Step 2 demand allocation, Step 3 optimisation (post cases), Step 4 power flow and the expansion analysis; results go to the service's database{{ mode === 'grid' ? ' (the electrification of this grid alone decides which buildings get heat pumps, EVs and PV)' : '' }}.</div>
          </template>
        </template>
      </div>

      <div v-else class="gx-jobs">
        <div class="gx-job-list">
          <div v-if="!s.jobs.length" class="empty" style="padding: 18px"><div class="empty-icon"><Icon name="terminal" :size="20"/></div><h3>No GridExpand jobs yet</h3><p>Start a run in the New run tab; its live log appears here.</p></div>
          <button v-for="j in s.jobs" :key="j.id" class="job-item" :class="{on: j.id === s.activeJobId}" @click="pick(j.id)">
            <span class="status-ico" :class="j.status"><Spinner v-if="j.status === 'running'" :size="13"/><Icon v-else :name="icon(j.status)" :size="14"/></span>
            <span class="j-title" :title="j.title">{{ j.title }}</span>
            <span class="j-meta">
              <span>{{ j.status === 'running' ? (j.phase || 'running') : j.status === 'queued' ? 'queued #' + j.queue_position : j.status }}</span>
              <span v-if="j.duration_s !== null">{{ duration(j.duration_s) }}</span>
              <span v-if="j.counts.error" style="color: var(--critical-text)">{{ j.counts.error }} err</span>
              <span style="margin-left: auto">{{ relative(j.created_at) }}</span>
            </span>
            <div v-if="j.status === 'running'" class="progress" :class="{indeterminate: j.progress === null}"><span :style="{width: (j.progress || 0) * 100 + '%'}"></span></div>
          </button>
        </div>
        <div v-if="job" class="console">
          <div class="console-bar">
            <StatusBadge :status="job.status"/>
            <span v-if="job.status === 'running' && job.progress !== null" class="small muted num">{{ Math.round(job.progress * 100) }} %</span>
            <span v-if="job.status === 'queued'" class="small muted">position {{ job.queue_position }}</span>
            <span class="grow"></span>
            <Seg v-model="filter" :options="FILTERS"/>
            <a class="btn ghost icon sm" :href="logHref()" title="Download log"><Icon name="download" :size="14"/></a>
            <button v-if="job.status === 'succeeded' || job.status === 'failed'" class="btn sm soft" @click="showResults"><Icon name="chart" :size="13"/>Results</button>
            <button v-if="job.status === 'running' || job.status === 'queued'" class="btn sm danger-ghost" @click="cancel"><Icon name="stop" :size="13"/>Cancel</button>
          </div>
          <div class="gx-steps">
            <div v-for="step in job.steps" :key="step.name" class="gx-step">
              <div class="row" style="gap: 8px">
                <span class="badge" :class="stepBadge(step.status)">{{ step.status }}</span>
                <span class="strong small">{{ step.name }}</span>
                <span class="small muted num" v-if="step.grids_total !== null">{{ step.grids_done + step.grids_failed }}/{{ step.grids_total }} grids<template v-if="step.grids_failed"> · <span style="color: var(--critical-text)">{{ step.grids_failed }} failed</span></template></span>
                <span class="grow"></span>
                <span class="small muted">{{ step.status === 'running' ? (step.stage || '') : step.duration_s !== null ? duration(step.duration_s) : '' }}</span>
              </div>
              <div v-if="step.status === 'running'" class="progress" style="height: 4px; margin-top: 5px" :class="{indeterminate: !step.grids_total}"><span :style="{width: (step.grids_total ? (step.grids_done + step.grids_failed) / step.grids_total * 100 : 0) + '%'}"></span></div>
              <div v-if="stepGrids(step).length" class="gx-grid-chips">
                <a v-for="g in stepGrids(step)" :key="g.candidate_index" class="badge" :class="g.status === 'done' ? 'good' : g.status === 'failed' ? 'bad' : 'accent'"
                   :href="g.log ? fileHref(step, g) : null" target="_blank" :title="'Grid ' + g.kcid + '/' + g.bcid + ': ' + g.status + (g.stage ? ' · ' + g.stage : '') + (g.seconds ? ' · ' + g.seconds + ' s' : '') + (g.log ? ' (click: log)' : '')">#{{ g.candidate_index }} {{ g.kcid }}/{{ g.bcid }}</a>
              </div>
            </div>
          </div>
          <div class="small mono muted gx-command" :title="job.command">$ {{ job.command }}</div>
          <div ref="body" class="console-body" @scroll="onScroll" tabindex="0" aria-label="Job log">
            <div class="log-lines" :style="{height: total * 18 + 'px', position: 'relative'}">
              <div :style="{position: 'absolute', top: first * 18 + 'px', left: 0, minWidth: '100%'}">
                <div v-for="l in visible" :key="l.seq" class="log-line" :class="l.level"><span class="t"></span><span class="m">{{ l.text }}</span></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>`,
  };
}
