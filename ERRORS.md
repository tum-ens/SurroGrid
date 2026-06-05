# Error Log

No repeated failed approaches logged yet.


## 2026-05-29 - Step 3 Gurobi/Pyomo solve attempts

- What did not work: Using only `gurobi_cl` with `SolverFactory("gurobi")`; Pyomo accepted the executable in a toy case but did not load a valid solution. Running the full Step 3 smoke test in multiprocessing also hid the child failure behind `KeyError: 0`.
- What worked instead: Installing `gurobipy` made `SolverFactory("gurobi")` available and license access succeeded. Adding worker traceback propagation made multiprocessing failures clearer.
- Note for next time: If Step 3 exits with code 137 after model setup, check memory/resource limits or reduce the smoke problem size before looking for another Gurobi installation issue.


## 2026-05-29 - Step 4 pf_res_plotly integration attempts

- What didn't work: Calling `pf_res_plotly` with only `vm_pu` and `loading_percent` populated from saved outputs failed because this pandapower version also expects `res_bus.va_degree`, `res_line.i_from_ka`, `res_line.i_to_ka`, and `res_trafo` fields. A second run then failed due missing optional `plotly` dependency in the step-4 environment.
- What worked instead: Populate the missing result columns with available/neutral defaults before plotting and add `plotly` to `GridExpand/4.powerflow/pyproject.toml`, then `uv sync`.
- Note for next time: `pf_res_plotly` is convenience-heavy and assumes a fuller result table than strictly needed for color maps; verify required columns and optional plotting extras up front.


## 2026-05-29 - Step 4 line-loading colorbar not shown with create_line_trace

- What didn't work: Switching from `pf_res_plotly` to explicit `create_line_trace(..., show_colorbar=True, cbar_title=...)` did not render the line-loading colorbar in output HTML for this environment.
- What worked instead: Keep pandapower trace-based plotting, then append one minimal invisible Plotly marker trace with a dedicated colorbar titled `Line Loading [%]` when missing.
- Note for next time: Validate colorbar presence from generated figure JSON/HTML, not only function arguments, because trace-helper behavior can vary by pandapower/plotly combination.


## 2026-06-01 - SurroGrid schema cleanup DDL attempts

- What did not work: `apply_patch` continued to fail in the sandbox with `bwrap: loopback: Failed RTM_NEWADDR`, and the first scenario DDL used a semicolon inside a quoted description string, which broke the existing simple semicolon-based SQL statement splitter.
- What worked instead: Apply the file edits through the project `uv run` Python environment after approval, keep the DDL strings free of embedded semicolons, and validate with `SurroGridDatabase.ensure_schema()`.
- Note for next time: Either keep `surrogrid_schema.sql` free of semicolons inside quoted strings or replace the schema loader with a real SQL splitter before adding richer DDL blocks.


## 2026-06-02 - Timeslice revert file-edit attempts

- What did not work: `apply_patch` failed again with the sandbox loopback error. A direct `uv run python -c` edit then failed first under the same sandbox restriction and later due shell quoting/newline issues in embedded replacement strings.
- What worked instead: Run an elevated `uv run python` heredoc and use explicit text replacements with triple-quoted strings.
- Note for next time: For multi-file edits in this environment, prefer `apply_patch` first for policy compliance, but if the loopback error recurs, use an elevated `uv run python` heredoc with simple, readable replacements.

## 2026-06-03 - Markdown edits through shell-quoted Python

- What did not work: Running `uv run python -c` inside double quotes for Markdown-heavy README replacements let shell command substitution execute text inside backticks before Python received the script. Intermittent `apply_patch`/shell commands also failed with `bwrap: loopback: Failed RTM_NEWADDR`.
- What worked instead: Restore the README from `git show HEAD:<path>` inside a single-quoted Python command, then apply exact string replacements with triple-double-quoted Python strings so Markdown backticks remain literal.
- Note for next time: For Markdown-heavy scripted edits, use single-quoted shell Python or a temporary script file pattern instead of double-quoted `python -c` strings containing backticks.

## 2026-06-05 - Local edit fallback after sandbox loopback

- What didn't work: `apply_patch` and one `uv run python -c` edit hit the recurring `bwrap: loopback: Failed RTM_NEWADDR` sandbox failure while updating `GridExpand/database.py`. A first escalated `python -c` retry then failed because nested triple-quoted replacement text broke shell quoting.
- What worked instead: Use an escalated `uv run python` heredoc with explicit `Path.read_text()` / `write_text()` replacements, avoiding shell interpolation inside the replacement strings.
- Note for next time: For multi-line Python fallback edits after `apply_patch` loopback failures, use a heredoc directly and do not compress complex replacements into `python -c`.

## 2026-06-05 - Concurrent schema setup deadlocks in Munich runner

- What didn't work: Running Munich timeslice batches with multiple workers while every Step 2 and Step 4 process independently called `SurroGridDatabase.ensure_schema()`. Workers intermittently deadlocked in PostgreSQL during `pipeline_run` registration or `CREATE INDEX IF NOT EXISTS` schema setup.
- What worked instead: Add a fast schema-readiness check so already-current databases skip DDL during candidate startup, and keep a PostgreSQL transaction advisory lock for the rare case where schema/migration DDL is actually needed. Keep candidate concurrency at 3 unless a later run proves higher DB startup concurrency is safe.
- Note for next time: If concurrent AGS runs fail immediately with `psycopg2.errors.DeadlockDetected` near schema/index creation or run registration, check for unguarded DDL in per-candidate startup before tuning solver memory.

## 2026-06-05 - Open-Meteo transient failures in Munich timeslice

- What didn't work: Treating Step 2 Open-Meteo soil-temperature request failures as candidate/model failures. Candidate 48 first hit a read timeout, candidate 231 hit a long timeout, and candidate 255 hit `Network is unreachable` while requesting `archive-api.open-meteo.com`.
- What worked instead: Leave the failed candidates recorded by the AGS runner and rerun them with `--resume --rerun-failed`; all Open-Meteo failures passed on the failed-only rerun without code changes.
- Note for next time: If Step 2 fails in `get_open_meteo_soil_temperature`, inspect the candidate log before changing code. Prefer a failed-only rerun first; add request retry/caching only if the same candidate repeatedly fails across rerun windows.
