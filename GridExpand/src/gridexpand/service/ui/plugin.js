// GridExpand plugin for pylovo-ui (host API 1): three panels and a map layer.
// Loaded by pylovo-ui's static/js/plugins.js from <base>ui/manifest.json -> entry.
import { createContext } from './lib.js';
import { createRunsPanel } from './panels/runs.js';
import { createResultsPanel } from './panels/results.js';
import { createScenariosPanel } from './panels/scenarios.js';
import { installMapLayer } from './maplayer.js';

const CSS = `
.gx-panel .panel-body.stack > * { flex-shrink: 0; }
.gx-panel .gx-region { display: flex; align-items: center; gap: 12px; padding: 10px 12px; }
.gx-panel .gx-table { border: 1px solid var(--border); border-radius: 10px; max-height: 230px; }
.gx-panel .gx-table input[type=checkbox] { accent-color: var(--accent); margin: 0; }
.gx-panel .gx-form { grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); }
.gx-panel .gx-charts { grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); }
.gx-panel .gx-jobs { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.gx-panel .gx-job-list { flex: 0 0 auto; max-height: 32%; overflow: auto; background: var(--surface-2); border-bottom: 1px solid var(--border); }
.gx-panel .gx-jobs .console { flex: 1; }
.gx-panel .gx-steps { padding: 6px 12px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 6px; max-height: 40%; overflow: auto; }
.gx-panel .gx-grid-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }
.gx-panel .gx-grid-chips a { text-decoration: none; }
.gx-panel .gx-command { padding: 4px 12px; border-bottom: 1px solid var(--border); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.gx-panel .log-line { grid-template-columns: 0 auto; gap: 0; }
.gx-scen .panel-toolbar.nowrap .select { max-width: 260px; }
.gx-scen .gx-summary { flex: none; display: flex; align-items: center; gap: 10px; padding: 7px 12px; border-bottom: 1px solid var(--border); background: var(--surface-2); }
.gx-scen .gx-keys { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 2px; }
.gx-scen .gx-changes { flex: none; max-height: 150px; overflow: auto; padding: 4px 12px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 2px; }
.gx-scen .gx-changes.boxed { border: 1px solid var(--border); border-radius: 9px; max-height: 190px; }
.gx-scen .gx-change { display: flex; justify-content: space-between; gap: 12px; font-size: 12px; }
.gx-scen .gx-issues { flex: none; padding: 8px 12px; border-bottom: 1px solid var(--border); max-height: 28%; overflow: auto; }
.gx-scen .form-field .gx-reset { width: 20px; height: 20px; margin: -3px 0; }
.gx-scen .form-field.gx-bad .input, .gx-scen .form-field.gx-bad .select { border-color: var(--critical); }
.gx-scen .gx-dialog-grid { grid-template-columns: 1fr 1fr; }
.gx-scen .gx-keyline { padding: 8px 12px; display: flex; flex-direction: column; gap: 3px; }
.gx-legend { top: 58px; left: 10px; padding: 8px 10px 10px; width: 250px; font-size: 12px; max-height: calc(100% - 150px); overflow: auto; }
.gx-legend .gx-hover { margin-top: 8px; padding-top: 7px; border-top: 1px solid var(--border); min-height: 34px; }
.gx-legend .tt-row { display: flex; justify-content: space-between; gap: 10px; color: var(--text-2); }
.gx-legend .tt-row span:last-child { color: var(--text); font-variant-numeric: tabular-nums; text-align: right; }
`;

function addStyles() {
  if (document.getElementById('gridexpand-plugin-css')) return;
  const style = document.createElement('style');
  style.id = 'gridexpand-plugin-css';
  style.textContent = CSS;
  document.head.appendChild(style);
}

export function register(host) {
  addStyles();
  const ctx = createContext(host);
  host.registerPanel('runs', { title: 'GridExpand runs', icon: 'play', component: createRunsPanel(ctx) });
  host.registerPanel('results', { title: 'Expansion results', icon: 'gauge', component: createResultsPanel(ctx) });
  host.registerPanel('scenarios', { title: 'Scenario editor', icon: 'sliders', component: createScenariosPanel(ctx) });
  installMapLayer(ctx);
  window.gridexpandPlugin = ctx;  // for debugging in the browser console
}
