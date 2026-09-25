// GridExpand plugin for pylovo-ui (host API 1): two panels and a map layer.
// Loaded by pylovo-ui's static/js/plugins.js from <base>ui/manifest.json -> entry.
import { createContext } from './lib.js';
import { createRunsPanel } from './panels/runs.js';
import { createResultsPanel } from './panels/results.js';
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
.gx-panel .gx-grid-card { display: flex; align-items: flex-start; gap: 11px; padding: 11px 12px; border-color: color-mix(in srgb, var(--accent) 35%, var(--border)); }
.gx-panel .gx-grid-icon { width: 34px; height: 34px; border-radius: 9px; display: grid; place-items: center; flex: none; color: var(--accent); background: color-mix(in srgb, var(--accent) 12%, transparent); }
.gx-panel .gx-terminal { padding: 10px 12px; display: flex; flex-direction: column; gap: 7px; }
.gx-panel .gx-cmd-line { display: flex; align-items: center; gap: 6px; background: var(--surface-2); border: 1px solid var(--border); border-radius: 7px; padding: 3px 4px 3px 8px; }
.gx-panel .gx-cmd-line code { flex: 1; font-size: 11.5px; overflow-x: auto; white-space: nowrap; }
.gx-panel .gx-trun { display: flex; align-items: center; gap: 8px; padding: 5px 2px; border-bottom: 1px solid var(--border); }
.gx-legend { top: 58px; left: 10px; padding: 8px 10px 10px; width: 250px; font-size: 12px; max-height: calc(100% - 150px); overflow: auto; }
.gx-legend .gx-hover { margin-top: 8px; padding-top: 7px; border-top: 1px solid var(--border); min-height: 34px; }
.gx-legend .tt-row { display: flex; justify-content: space-between; gap: 10px; color: var(--text-2); }
.gx-legend .tt-row span:last-child { color: var(--text); font-variant-numeric: tabular-nums; text-align: right; }
.gx-legend .gx-check { display: flex; align-items: center; gap: 6px; margin: 6px 0 4px; font-weight: 600; font-size: 11.5px; color: var(--text-2); cursor: pointer; }
.gx-legend .gx-check input { accent-color: var(--accent); margin: 0; }
.gx-legend .gx-asset-row { display: flex; align-items: center; gap: 7px; width: 100%; padding: 3px 4px; margin: 0 -4px; border: 0; background: none; border-radius: 6px; color: var(--text); font: inherit; cursor: pointer; text-align: left; }
.gx-legend .gx-asset-row:hover { background: var(--surface-2); }
.gx-legend .gx-asset-row.off { opacity: 0.42; }
.gx-legend .gx-asset-row.off img { filter: grayscale(1); }
.gx-legend .gx-asset-row .count { margin-left: auto; color: var(--text-2); font-variant-numeric: tabular-nums; }
.gx-legend .gx-asset-line { display: flex; align-items: flex-start; gap: 6px; margin-top: 3px; }
.gx-legend .gx-asset-line img { margin-top: 1px; flex: none; }
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
  installMapLayer(ctx);
  window.gridexpandPlugin = ctx;  // for debugging in the browser console
}
