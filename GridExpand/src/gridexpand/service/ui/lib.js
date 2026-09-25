// Shared state of the GridExpand panels: region from pylovo's selection, service status,
// scenarios, jobs (followed with Server-Sent Events) and the result map layer.
import { reactive } from 'vue';

export const CASES = [
  { id: 'pre', label: 'Status quo', short: 'pre', post: false },
  { id: 'post-hems-heuristic', label: 'HEMS, heuristic sizing', short: 'heuristic', post: true },
  { id: 'post-hems-optimized', label: 'HEMS, optimised sizing', short: 'optimised', post: true },
];
export const CASE_LABEL = Object.fromEntries(CASES.map((c) => [c.id, c.label]));
export const TIMEFRAMES = [
  { id: 'max_base_electricity_demand_week', label: 'Peak-demand week (168 h)' },
  { id: 'min_temperature_week', label: 'Coldest week (168 h)' },
  { id: 'max_solar_radiation_week', label: 'Sunniest week (168 h)' },
  { id: 'full_year', label: 'Full year (8760 h)' },
];
export const TIMEFRAME_LABEL = Object.fromEntries(TIMEFRAMES.map((t) => [t.id, t.label]));

export function euro(value) {
  if (value === null || value === undefined) return '–';
  const abs = Math.abs(value);
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)} M€`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(abs >= 1e5 ? 0 : 1)} k€`;
  return `${Math.round(value)} €`;
}

export function createContext(host) {
  const { store } = host;
  const state = reactive({
    status: null, statusError: null, statusLoading: false,
    scenarios: [], scenariosError: null,
    jobs: [], activeJobId: null, logTick: 0,
    resultsTick: 0,           // bumped when a job finished: result panels reload
    focusAnalysis: null,      // analysis key the results panel should select
    focusScenario: null,      // scenario file the runs panel should select (set after saving a scenario)
    editScenario: null,       // scenario file the scenario editor should open (set by the runs panel)
    layer: { on: false, key: null, label: null, data: null, loading: false, error: null, hover: null,
      assets: null, gridOn: true, assetsOn: { pv: true, battery: true, heat_pump: true, ev: true } },
  });
  const lines = new Map();    // job id -> log lines (kept outside Vue reactivity)
  const streams = new Map();

  function region() {
    const grid = store.grid.detail?.grid;
    const plz = store.results.plz || store.region || grid?.plz || (store.basket.length === 1 ? store.basket[0].plz : null);
    const versions = store.versions || [];
    const version = store.results.version || grid?.version_id || (versions.length ? versions[versions.length - 1].version_id : null);
    return { plz: plz ? Number(plz) : null, version: version ? String(version) : null, gridId: grid?.grid_result_id ?? store.grid.id ?? null };
  }

  async function loadStatus() {
    state.statusLoading = true;
    try { state.status = await host.api('status'); state.statusError = null; } catch (err) { state.statusError = err.message; } finally { state.statusLoading = false; }
  }
  async function loadScenarios() {
    try { state.scenarios = await host.api('scenarios'); state.scenariosError = null; } catch (err) { state.scenariosError = err.message; }
  }

  function upsert(summary) {
    const i = state.jobs.findIndex((j) => j.id === summary.id);
    if (i >= 0) state.jobs.splice(i, 1, { ...state.jobs[i], ...summary });
    else state.jobs.unshift(summary);
  }
  async function loadJobs() {
    try {
      state.jobs = await host.api('jobs');
      for (const job of state.jobs) if (job.status === 'running' || job.status === 'queued') follow(job.id);
      if (!state.activeJobId && state.jobs.length) selectJob(state.jobs[0].id);
    } catch (err) { /* the status chip shows service problems */ }
  }
  async function selectJob(id) {
    state.activeJobId = id;
    if (!lines.has(id)) {
      try {
        const res = await host.api(`jobs/${id}/log`, { params: { after: 0 } });
        if (!streams.has(id)) lines.set(id, res.lines);
        upsert(res.job);
        state.logTick++;
      } catch (err) { /* ignore */ }
    }
  }
  function follow(id) {
    if (streams.has(id)) return;
    const list = lines.get(id) || [];
    lines.set(id, list);
    const after = list.length ? list[list.length - 1].seq + 1 : 0;
    const source = new EventSource(host.url(`api/jobs/${id}/events?after=${after}`));
    streams.set(id, source);
    let pending = false;
    source.addEventListener('log', (ev) => {
      const target = lines.get(id);
      for (const line of JSON.parse(ev.data)) if (!target.length || line.seq > target[target.length - 1].seq) target.push(line);
      if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; state.logTick++; }); }
    });
    source.addEventListener('status', (ev) => upsert(JSON.parse(ev.data)));
    source.addEventListener('end', (ev) => {
      source.close();
      streams.delete(id);
      const job = JSON.parse(ev.data);
      upsert(job);
      state.logTick++;
      finished(job);
    });
    source.onerror = () => { if (source.readyState === EventSource.CLOSED) streams.delete(id); };
  }
  function finished(job) {
    const { toast } = host.ui;
    if (job.status === 'succeeded') toast('good', 'GridExpand job finished', `${job.title} · ${Math.round(job.duration_s)} s`);
    else if (job.status === 'failed') toast('bad', 'GridExpand job failed', `${job.title}: ${job.error || 'see the log'}`, 12000);
    else toast('warn', 'GridExpand job cancelled', job.title);
    state.resultsTick++;
    loadStatus();
  }
  async function startPipeline(body) {
    try {
      const job = await host.api('jobs/pipeline', { method: 'POST', body });
      upsert(job);
      lines.set(job.id, []);
      state.activeJobId = job.id;
      follow(job.id);
      host.ui.toast('info', 'GridExpand job queued', job.title, 3500);
      return job;
    } catch (err) {
      host.ui.errorToast('Could not start the GridExpand job', err);
      return null;
    }
  }
  async function cancelJob(id) {
    try { upsert(await host.api(`jobs/${id}/cancel`, { method: 'POST' })); } catch (err) { host.ui.errorToast('Could not cancel', err); }
  }

  let loaded = false;
  function ensureLoaded() {
    if (loaded) return;
    loaded = true;
    loadStatus();
    loadScenarios();
    loadJobs();
    setInterval(() => { if (!document.hidden) loadStatus(); }, 30000);
  }

  return { host, store, state, lines, region, ensureLoaded, loadStatus, loadScenarios, loadJobs, selectJob, follow, startPipeline, cancelJob };
}

// Display name of an analysis: the status quo, or the post stage of a model case.
export function analysisLabel(a) {
  if (!a) return '';
  if (a.stage === 'pre') return a.model_case && a.model_case !== 'pre' ? `Status quo (reference in the ${CASES.find((c) => c.id === a.model_case)?.short || a.model_case} run)` : 'Status quo';
  return CASE_LABEL[a.model_case] || a.model_case || a.stage;
}
