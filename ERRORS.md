# Error Log

No repeated failed approaches logged yet.


## 2026-05-29 - Step 3 Gurobi/Pyomo solve attempts

- What did not work: Using only `gurobi_cl` with `SolverFactory("gurobi")`; Pyomo accepted the executable in a toy case but did not load a valid solution. Running the full Step 3 smoke test in multiprocessing also hid the child failure behind `KeyError: 0`.
- What worked instead: Installing `gurobipy` made `SolverFactory("gurobi")` available and license access succeeded. Adding worker traceback propagation made multiprocessing failures clearer.
- Note for next time: If Step 3 exits with code 137 after model setup, check memory/resource limits or reduce the smoke problem size before looking for another Gurobi installation issue.
