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
