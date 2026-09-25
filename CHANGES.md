# CHANGES on `gridexpand-fable` (GridExpand review, 2026-09-25)

This file summarises one autonomous review session of **GridExpand** (GridForecast was not touched). It is written
for the maintainer who reviews and merges the branch. Delete it (or fold it into `CHANGELOG.md`) before merging.

- Base: `feature/update-pipeline-opus` (`87ed13f`). Branch `gridexpand-fable`: 121 commits (109 non-merge), nothing
  pushed, `main`/`develop` untouched. Every area was developed on its own branch and merged with `--no-ff`
  (`fable/restructure`, `fable/impl-{db,alloc,optpf,orch,post,service,docs}`, `fable/pf-staging`), so one area can
  be reverted with `git revert -m 1 <merge>`.
- `AGENTS.md` asks to propose options and wait before significant work; you asked me to decide everything except
  very important questions, so I decided and list every decision here (section 2).
- **Your databases were not touched** (no reads, no writes). All testing ran in the throwaway sandbox container
  `pylovo-fable-sandbox` (127.0.0.1:55439) on an OpenStreetMap demo region.
- Size: GridExpand Python outside tests 63,921 → 55,863 lines; without the new web service 53,443 (−16.4 %) — while
  adding a run model, migrations and a CLI. Tests: 0 → 535 (4,976 lines; plus an end-to-end regression harness).
  `uv run pytest -q` and `uv run ruff check src tests scripts` pass.

---

## 0. Read first

### 0.1 Results are unchanged — proven

GridExpand had no tests, so the first step was an end-to-end regression harness (`GridExpand/tests/regression/`):
the ordinary synthetic pipeline (Steps 2–4 + expansion) on 4 sandbox grids × 3 model runs (`pre`,
`post-hems-heuristic`, `post-hems-optimized`, one-week timeframe, raw + summary output), snapshotting all 26
`surrogrid` tables (2.3 M rows) and comparing them row by row on natural keys. Two runs of the same code are
**IDENTICAL**. Every merge was checked against it:

| Step | vs previous reference | differences |
|---|---|---|
| bug fixes on the old layout (`3acde89`…`74361b7`) | — | deliberate: heat seeding (see 0.2) |
| one package, Step 3 on Python 3.12 / numpy 2 | IDENTICAL | none |
| DB layer, Step 2, Step 3/4 (wave A) | identical values | only metadata: `schema_migration` (new), unused `baseline_static` row removed, `expansion_line_result.critical_ts` now filled (was always NULL), solver provenance keys in the run assumptions |
| orchestration, analysis (wave B) + follow-ups | identical values | expansion analysis keys now carry AGS, scenario and model case, so the heuristic analyses are **no longer overwritten** by the optimized ones (3 → 5 analyses); `expansion_analysis_run.scenario_id` filled. Mapping the keys back gives IDENTICAL |
| service, docs, final fixes (tip of the branch) | IDENTICAL | none |

Also proven: every urbs LP file is byte-identical after the urbs cleanup; `gridexpand run <run.yaml>` produces the
same database as the command-line runner; a run cancelled in Step 3 and resumed equals an uninterrupted run; a
migrated legacy schema equals a fresh one. The week harness is 27 % faster end to end (679 s → 494 s).

**Not run end to end** (and changed only mechanically, covered by unit tests): the paired SWF / aligned SWF+ÜZW
pipelines (confidential data, and pylovo v1 was dropped), full-year runs, TSAM, the InfDB `ro_heat` path, h5 mode.

### 0.2 One deliberate result change: heat is now reproducible

TEASER occupancy and hot-water draws used Python's global `random`, which nothing seeded: heat demand changed between
runs **and between the heuristic and optimized case of the same grid** (up to 63 kWh/a space heat, 1.1 % DHW per
building), contrary to the `profile_seed` contract. It is now seeded per building (`4faeb56`). New runs therefore
differ once from old runs; the paired heat libraries you already prepared stay valid as files.

### 0.3 How to adopt the branch

1. Merge (or check out) `gridexpand-fable`. `feature/update-pipeline-opus` has no commits since the branch point,
   so this is a fast-forward. Directories changed completely (section 5); git does not move your untracked files
   (results, prepared paired datasets, run logs, the large statistics files).
2. `cd GridExpand && uv sync` (one environment now; add `--extra service` for the web service).
3. `uv run python scripts/migrate_local_layout.py` shows the moves (dry run), `--execute` moves them into `work/` and
   `data/` — it never overwrites or deletes, and leaves git-tracked files to git. Dry run on your checkout
   (`AI/SurroGrid/GridExpand/fable-review/migrate_dryrun.txt`): 5,985 moves (≈33 GB, renames on the same disk),
   **0 conflicts**; the only files it does not cover are the five old per-step `.venv` folders (3.2 GB),
   `__pycache__` and `.ruff_cache` — delete them afterwards.
4. `.env`: unchanged keys; optionally `GUROBI_HOME` / `GRB_LICENSE_FILE` (defaults: the old paths when they exist).
5. **Database:** new code refuses a pre-migration schema. Run `uv run gridexpand db migrate --plan`, then `--apply`
   (index cleanup, constraints; no data rewrite). Because the pylovo schema was dropped, production has no pylovo
   foreign key and no views: after pylovo v1 is regenerated, run `gridexpand db relink-pylovo --plan/--apply`. It
   matches each `grid_case` to the new grid by (version, PLZ, kcid, bcid), checks the building set, lists
   mismatches, re-adds the FK and recreates the views. Take a backup first (`docs/database.md` has the runbook).
6. Launch runs with `uv run gridexpand run config/runs/<file>.yaml` (all three pipelines), follow them with
   `gridexpand status <run dir>`.

---

## 1. Overview

| Area | Merge | Main changes |
|---|---|---|
| Bug fixes before refactoring | `3acde89`, `1c8174f`, `c97dd33`, `8be64f9`, `4faeb56`, `74361b7` | fresh-DB schema, week-mode runs, synthetic assignment hash, grids without PV, heat seeding, scenario row |
| Regression harness | `e59afa9` | `GridExpand/tests/regression/` |
| One package | `fable/restructure` (ff) | `src/gridexpand/`, one environment, `gridexpand` CLI, `paths.py`, `work/`, `data/`, `config/`, `docs/` |
| Database | `d74a3de` | migrations, `gridexpand db …`, COPY writers (13.5×), indexes, RESTRICT FK, relink, compression |
| Step 2 | `d93b6e7` | `run_allocation`, one weather module, shared helpers, −1,850 lines, 4 bugs |
| Step 3/4 | `a98a86e` | urbs 8,738 → 2,789 lines (LP identical), configurable solver + provenance, one-pass power flow |
| Service + Docker | `5ec62c5`, `199a291` | `gridexpand serve`, plugin panels, Docker image; staging runs hidden, container fix |
| Step 4 atomic rerun | `41bf94e` | staging run swapped in at the end |
| Dependencies | `f77f47c` | 11 unused packages dropped |
| Orchestration | `cc149c0` | `gridexpand run` for all pipelines, run directories with `state.json`, command builders, model-case table |
| Analysis + sampling | `d4cf6da` | expansion SQL in files, one transaction, replace guard, Step 1 coordinates fixed, notebook archive |
| Keys + refresh | `2a8d9a9` | scenario-qualified analysis keys, one QGIS refresh per batch |
| Final fixes | `8a87ae7`, `c49e9f7`, `efd80c0` | INFLEX refusal test, linear layout migration, `ruff check` clean with the project config |
| Documentation | `2f6eecd` | see section 10 |
| Final fixes | `958a139`, `7a0b5e0` | `--run-name` required for `gridexpand expansion`, runnable heatmaps module, HiGHS in the standalone compose |

Outside this repository: pylovo branch `fable/ui-plugins-dev-review` (plugin loader on `feature/dev-review`), and the
new local repository `../GridPlanner` (sections 3 and 9).

---

## 2. Decisions for you

Decided by me and reversible; please confirm or override.

1. **Synthetic mobility still uses the legacy v1 pool** (`mobility_profile_pool_old`, clipped, not
   energy-conserving); only paired/aligned runs use the v2 session pool. Consequence: the synthetic INFLEX power flow
   cannot work (no `urbs_in/ev_sessions`), so `post-inflex-heuristic` is refused up front for synthetic runs and was
   removed from `forchheim_2045_synthetic.yaml` / `schweinfurt_2045_synthetic.yaml`. Migrate synthetic runs to v2?
2. **Heuristic dispatch is a degenerate LP** (flat tariff, zero feed-in): EV/HP/battery timing is one arbitrary optimal
   vertex; HiGHS picks another (import peak 526.5 vs 535.0 kW on one grid). Add a tie-breaker? And `MIPGap=0.05` for
   the optimized case (solves take < 1 s per cluster in a week): tighten?
3. **The Step 3 partition depends on `--step3-max-cpus`** (a machine flag): the same input on another machine can give
   different results. Make it depend on building count only?
4. **pylovo deletions:** the FK from `surrogrid.grid_case` to `pylovo.grid_result` was `ON DELETE CASCADE` (deleting a
   pylovo version silently deleted SurroGrid results). It is now `RESTRICT`. OK?
5. **Raw full-year power-flow output** for every grid would be ≈2.7 TB for 584 grids (estimate). `gridexpand db
   compress` (TimescaleDB licence) reduces it ≈8–10×; YAML synthetic runs now default to `powerflow_output: summary`.
   Which grids need raw output?
6. **h5 input mode** (Step 1 → Step 2 without DB) cannot run post cases (Step 1 exports no LoD2 roofs). Fix or retire?
   Related: Step 1 wrote **wrong coordinates** (assumed EPSG:3035, pylovo uses 25832 — a building at 47.97 N 11.77 E
   became 54.2 N −56.0 E), so weather of h5-mode grids came from the wrong place. Fixed; old sampled grids are affected.
7. **Electrification selection scope** differs by entry point: `gridexpand synthetic` selects AGS-wide, a YAML run
   selects within its region filter (AGS, PLZ or one grid). Keep?
8. **PV fallback** is unreachable for selected buildings (eligibility requires real LoD2 roofs). Retire it?
9. **DST:** synthetic profiles do not preserve annual energy across the autumn hour, paired electricity does. Align?
10. **Heat internal gains** use bus electricity including non-residential components of mixed buildings. Intended?
11. **SWF-only `paired_validation` pipeline** (v11 YAMLs) and the SWF-only audits: still needed next to `paired_aligned`?
12. **QGIS materialized views**: still used? (now refreshed once per batch instead of up to 12× per run)
13. **Cost assumptions** live in the DB baseline migration (seed row no longer reset on every call). Move to a YAML?
14. **Write-only audit tables** (`allocated_vehicle`, `electrification_assignment`, raw demand): keep writing them?
15. **Job state for the UI** is file-based (`state.json` in the run directory), not a DB table. OK?
16. **After the pylovo v1 restore:** are grids identical (PLZ/kcid/bcid and building sets)? `relink-pylovo` checks it
    and lists what does not match.
17. **Five run-bound notebooks** moved to `notebooks/archive/` (historical, not maintained); `materialize_powerflow_summary`
    was retired (diverged from Step 4, no caller). OK?
18. **`AGENTS.md`** still says "use `<package> --help`"; after the pylovo incident I suggest adding "read argparse code
    first; never run project CLIs against the real DB". Not edited (your file).
19. Possible real-data issue to check: SWF bus `geo` may be double-encoded (two helpers decode it differently); if so,
    real line geometries in the expansion views are NULL.
20. **Data licences before publishing:** the tracked inputs (`data/sampling/*`, `data/statistics/*`: Zensus 2022, BKG
    VG250, RegioStaR, PLZ shapes of unknown origin) have no verified licence or per-file provenance
    (`THIRD_PARTY_LICENSES` marks them "to be checked").
21. `CHANGELOG.md` has one entry (2025-12-15) and the new `CONTRIBUTING.md` asks for entries: keep it or drop it?
    Four research-note headers say "not recorded" for run id or pylovo version; fill them in if you know them.

---

## 3. Architecture recommendation: GridPlanner

**Recommendation:** keep pylovo and GridExpand as independent repositories that each publish a Docker image with
their CLI and a small job API, and add a **thin integration repository GridPlanner** that contains no application
code — only composition, proxy, docs, demo data and end-to-end tests. One UI shell (today pylovo-ui) loads the other
tools' panels as **plugins**.

Options considered:

| Option | For | Against |
|---|---|---|
| Monorepo / merge GridExpand into pylovo | atomic changes, one env | couples an open-source generator to a research pipeline; conflicting heavy dependencies (urbs/Gurobi/TEASER vs pgRouting/GDAL); different release cadence |
| pylovo-ui imports GridExpand as a library | no extra service | UI environment must satisfy both dependency sets; wrong dependency direction |
| iframes of two independent UIs | isolation | no shared map and selection, two UIs to learn |
| Workflow engine (Airflow/Prefect/Argo) | scheduling, retries | heavy infrastructure for a research tool; generic UI without grid maps |
| **Integration repo + images + job APIs + plugin panels** | independent development; each tool still usable alone (CLI, notebooks, HPC); one UX; versioned images | contracts (DB views, job API, plugin API) must be versioned |

Contracts: (1) **data plane** = the InfDB PostgreSQL, each tool owns its schema; cross-schema reads should go through
documented, versioned views (pylovo could publish its read contract plus a `schema_version`); (2) **control plane** =
one job-API shape per tool wrapping its own CLI (`POST /api/jobs…`, SSE logs, cancel); (3) **UI plugin contract**:
the shell fetches `<plugin>/ui/manifest.json`, imports the ES module and calls `register(host)` (panels, shared
selection, map, charts, toasts); (4) **images** built by each repository's CI and pinned by version in GridPlanner.
Start with pylovo-ui as the shell; extract a generic shell only when a third tool joins. Details:
`../GridPlanner/docs/ARCHITECTURE.md`.

What exists now (prototype, proven end to end in containers, screenshots in `GridPlanner/docs/img/`): `gridexpand
serve`, the pylovo-ui plugin loader, GridPlanner compose with a Caddy proxy (one origin: `/` pylovo-ui,
`/gridexpand/` GridExpand), a demo database recipe from OpenStreetMap. Workflow in one UI: pylovo generates and
analyses the grids of a PLZ → GridExpand panel: pick scenario, model cases and timeframe, start the run, follow the
live log → expansion results (costs, cables/transformers to reinforce, P99 loading, map layer). Containers use
HiGHS by default (no licence needed); Gurobi works through a WLS licence file but was not tested in a container.
Missing for production: CI-built images, a detached worker for multi-day runs, authentication beyond localhost.

![Expansion results in the pylovo UI](../GridPlanner/docs/img/gridplanner-3-expansion-results.png)

---

## 4. Bugs fixed

| Where | Bug | Effect |
|---|---|---|
| schema | DDL split on `;` inside a comment | a fresh database could never be initialised |
| Step 2 | `Path` in run assumptions | every week-mode DB run failed |
| synthetic runner | assignment hashed before CSV round trip | every synthetic post run rejected its own assignment |
| Step 2 | price table indexed from the PV table | grids without PV crashed urbs |
| Step 2 | unseeded TEASER occupancy/DHW | heat not reproducible, cases compared on different heat |
| DB | per-grid values in the shared scenario row | row depended on which run finished last |
| DB | schema marker never noticed dropped views/FK | views lost in the pylovo drop were never recreated |
| DB | scenario cleanup counts | under-reported; `--keep-demands` left real runs |
| DB | duplicate FKs / unique index | double checks and cascades |
| DB | `critical_ts` | always NULL |
| DB | DB URL not escaped | passwords with `@:/%` broke |
| Step 2 | vendored heat generator swallowed errors | a selected heat building could get 0 kW |
| Step 2 | PVGIS errors / no timeout | `TypeError`, hanging requests |
| Step 2 | occupancy fallback, zero-load partition | latent crash, silently dropped rows |
| Step 3 | non-optimal solves accepted | missing results became silent zero demand in Step 4 |
| Step 3 | `pdb.set_trace()` in an error handler; silent annuity 1; date-dependent labels; partial result files | hangs, wrong costs on errors, non-reproducible files |
| Step 4 | empty time chunks, `Pool()` sizing, single transformer, NaN demand | crashes with some CPU counts, over-subscription, silent zeros |
| Step 4 | rerun deleted old results first | a cancelled/crashed run left the grid without results |
| runner | expansion keys without case/AGS/scenario | optimized case overwrote heuristic analyses; regions/scenarios overwrote each other |
| runner | smoke runners, `run_aligned` subset order, paired resume by position, status without grid id | crashes, wrong resume |
| analysis | `aligned_expansion` NameError (since `87ed13f`), dry-run counts, missing case | aligned postprocessing could not start |
| analysis | non-atomic materialization, no `scenario_id`, `--replace` deleted foreign analyses | zero-cost ghost analyses, orphans |
| analysis | real-grid critical `t_index` always NULL; cutoff-plot crash | missing diagnostics |
| Docker | TEASER opens its shipped JSON inputs read-write | every heat generation failed for a non-root container user |
| Step 1 | EPSG:3035 assumed (pylovo uses 25832); lexicographic "latest" version; readout without occupants repair | wrong coordinates and weather in h5 mode |

Found and **not** fixed (decisions above): `pre` + weather-selected week modes cannot work (the status-quo branch
never fetches weather); synthetic INFLEX (decision 1).

---

## 5. Structure (`GridExpand/`)

```
pyproject.toml, uv.lock   one environment (Python 3.12); `gridexpand` console script; extras notebooks, service
src/gridexpand/
  cli.py        gridexpand run | status | grids | config | synthetic | allocate | optimize | powerflow |
                expansion | db | serve        (every --help works without a database)
  paths.py      the only module that knows directories (GRIDEXPAND_DATA_DIR / WORK_DIR / ENV_FILE)
  common/ db/ sampling/ allocation/ optimization/ powerflow/ analysis/ scenario/ paired/ service/
config/  data/  notebooks/{sampling,analysis,archive}/  docs/  scripts/  tests/  docker/
work/                     runtime artifacts (gitignored)
```

Removed: 5 per-step projects and lock files, conda environment files, 108 `sys.path` hacks, `os.chdir` hacks,
cwd-relative paths, three copies of `resource_report.py`, hard-coded Gurobi paths.

## 6. Database

(`src/gridexpand/db/`, `docs/database.md`)

- One versioned schema: `sql/0001_baseline.sql` … `0004`, re-runnable `views.sql`, `schema_migration`. Fresh
  databases initialise automatically; existing ones only via `gridexpand db migrate --plan/--apply`.
- `gridexpand db init-schema | migrate | compress | relink-pylovo | delete-scenario`.
- COPY writers: 751k rows 57.9 s → 4.3 s (13.5×) — for a full-year grid with 4 cases roughly 48 → 4 min of inserts.
- Hypertable per-bus/line/ts indexes dropped (≈18 % of raw storage, never used, one misled the planner); missing FK
  and lookup indexes added; building views computed per grid (verified identical).
- `list_grid_candidates()`: one copy of the candidate query, restricted to region + version (it aggregated the whole
  `buildings_result` table for every call).
- Step 4 reruns are atomic (staging run, swapped in one transaction; tested with SIGTERM mid-run).

## 7. Pipeline steps

- **Step 2:** `run_allocation(settings)` with stage and output-key tables (CLI unchanged); `Grid` 1,442 → 959 lines;
  one weather module for Steps 1 and 2 (timeouts, retries; unused soil-temperature download removed); one
  electrification inventory; cached profile tables; dead code (770-line module, config aliases, stale data, dead
  vendored parts) removed.
- **Step 3:** urbs 8,738 → 2,789 lines with byte-identical LP files; indexed commodity/storage balances (−27 % build
  time on a week, more for larger clusters); configurable solver `--solver` / `GRIDEXPAND_SOLVER` (`gurobi` default,
  `appsi_highs` licence-free); solver provenance in `urbs_out/solver_audit` and the run assumptions;
  `--scenario-config` required.
- **Step 4:** new engine (preallocated arrays, no per-timestep deepcopy, parallel summary); `--outputs raw,summary`
  in one pass (per case 22–24 s → 15–16 s, engine 5× faster); `ScenarioResultReader` and output sinks.
- **Step 5:** `grid_expansion` 1,124 → 480 lines, SQL in `analysis/expansion/sql/` (proven equivalent, 8× faster per
  analysis), one transaction, `--replace` guard, shared reinforcement heuristics with a SQL/Python parity test,
  ~3,100 lines of dead or duplicated loader/plot code removed; Step 1 uses the DB readers.

## 8. Orchestration and configuration

- `gridexpand run <run.yaml>` runs `pipeline: synthetic` (renamed from `scenario`; it used to run Step 2 only),
  `paired_validation` and `paired_aligned` with stages prepare → execute → postprocess. The run directory holds frozen
  input YAMLs, `identity.json`, an atomic `state.json`, `events.jsonl`, `plan.json`, `summary.json`; `--resume` is keyed
  by job, SIGTERM cancels all steps. `gridexpand status`, `gridexpand grids`, `gridexpand config check` (no DB).
- One command-builder module and one model-case table replace 29 hand-built command lists and 7 copies of the case
  rules; `run_candidate` 669 → ~90 lines; the two broken smoke runners, `run_scenario.py` and `run_aligned.py` removed.
- Scenario YAML values are unchanged; run YAMLs only renamed the pipeline (and dropped the impossible INFLEX case).
  `config/runs/sandbox_example.yaml` is the harness run as a YAML.

## 9. Integration prototype

See section 3. Files: `GridExpand/src/gridexpand/service/`, `GridExpand/docker/`, `GridExpand/docs/service.md`;
pylovo `fable/ui-plugins-dev-review` (+376/−6 lines on `feature/dev-review`, 118 frontend tests pass);
`../GridPlanner` (compose, proxy, `.env.example`, e2e tests, architecture, `demo/` OSM database recipe).

Final check with the built images, the proxy and a fresh sandbox database (HiGHS, one-week timeframe):

| Test | Result |
|---|---|
| `tests/e2e_smoke.sh --cases pre,post-hems-heuristic` (1 grid, 63 buildings) | all checks pass; `pre` 6.5 s, post case 54 s |
| Browser walk-through (`tests/e2e_ui.py`): grid in pylovo → GridExpand run of 2 grids (144 + 90 buildings), `pre` + `post-hems-heuristic` → results | job 183 s; 230 k€, 39 cables and 1 of 3 transformers to reinforce, P99 transformer loading 37 % → 253 %; map hover shows the reinforced cable |
| `tests/e2e_smoke.sh --dev --build` (images built from the checkouts) | all checks pass |

Not tested in containers: `post-hems-optimized`, Gurobi. GridPlanner has no remote; the images are local
(`gridexpand:fable-dev`, `pylovo-ui:fable-dev`).

## 10. Documentation

The documentation now describes the merged code (checked: 156 relative links, 0 broken; every documented
`gridexpand` flag exists in the `--help` texts).

| File | Content |
|---|---|
| `GridExpand/README.md` | what it is, install, large data assets, database setup, quick starts (run YAML, paired/aligned, service/Docker), command table, layout and `GRIDEXPAND_*` variables, adopting the new layout, testing |
| `docs/README.md` | index |
| `docs/configuration.md` | every scenario-YAML and run-YAML key (all three pipelines), what changes the hashes, grid selection, `gridexpand run` options, exit codes, run directory; `config/README.md` is a short checklist |
| `docs/method.md` | the scientific method (merged from `SCENARIO_METHOD.md`, the paired contract and the GHD evidence rules); values checked against the five scenario YAMLs; heat seeding |
| `docs/steps/1…5_*.md` | inputs, outputs, HDF keys, CLI options, conventions (Step 4: positive Q absorbs inductive reactive power; staging runs) |
| `docs/paired_validation.md`, `database.md`, `service.md`, `expansion_costs.md` | rewritten or checked against the code |
| `docs/research/<date>_<topic>.md` | 11 historical notes (audits, forensic notes, calibration runs) with a "historical record" header |
| repository `README.md`, `CONTRIBUTING.md`, `THIRD_PARTY_LICENSES`, `.gitignore` | GridExpand parts updated; CONTRIBUTING true again (GitHub, uv, pytest, ruff); upstream URLs of emobpy/urbs fixed |

Removed from the repository (agent handovers, per `AGENTS.md`): the two `*_AGENT_HANDOVER.md` files and
`VERIFICATION_PLAN.md`, now in `AI/SurroGrid/GridExpand/archived-docs/`.

---

## 11. Leftovers outside the repository

- Sandbox container `pylovo-fable-sandbox` (127.0.0.1:55439) with databases `surrogrid_fable_base` (harness template),
  `sg_*` (agent, e2e and reference runs); safe to drop the `sg_*` databases.
- The pylovo review worktree's `.env` points at `sg_impl_service_final` (uncommitted, only there).
- Local images `gridexpand:fable-dev`, `pylovo-ui:fable-dev` (no containers left running).
- Worktrees in the session scratchpad (`git worktree prune` cleans them after the scratchpad is gone); branches
  `fable/*` in SurroGrid and pylovo can be deleted after merging.
- Agent briefs, reviews and reports: `../AI/SurroGrid/GridExpand/fable-review/` (a copy of this file too).
- pylovo: branch `fable/ui-plugins-dev-review` (plugin loader) is not merged; it merges cleanly onto the current
  `feature/dev-review` (`c3020e8`). Your pylovo checkout (`local-grid-preparation-new`) was not touched.
