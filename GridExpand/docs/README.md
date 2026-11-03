# GridExpand documentation

Start with the [GridExpand README](../README.md) (install, data, database, quick starts, commands).

## Using GridExpand

| document | read it when you |
|---|---|
| [configuration.md](configuration.md) | write a scenario or run YAML; need a key, a default, the grid selection, the run directory or an environment variable |
| [paired_validation.md](paired_validation.md) | run the paired DSO A or the aligned DSO A/DSO B validation |
| [api.md](api.md) | run the HTTP API (for the GridPlanner UI) or the Docker image; its contract |
| [database.md](database.md) | set up or migrate the `surrogrid` schema, query results, run maintenance commands |

## Steps

| step | document |
|---|---|
| 1 | [steps/1_grid_sampling.md](steps/1_grid_sampling.md): pylovo export to HDF5, sampling notebooks |
| 2 | [steps/2_demand_allocation.md](steps/2_demand_allocation.md): time series, asset plans, urbs inputs, regional assignment, mobility pools |
| 3 | [steps/3_urbs.md](steps/3_urbs.md): urbs optimization, partition, solver, TSAM |
| 4 | [steps/4_powerflow.md](steps/4_powerflow.md): demand reconstruction, power flow, raw and summary outputs, sign conventions |
| 5 | [steps/5_postprocessing.md](steps/5_postprocessing.md): expansion materialization, notebooks, plots, audits |

## Method

| document | content |
|---|---|
| [method.md](method.md) | model cases, configuration ownership, time axis, reproducible realization, electrification, PV, battery, heat, mobility, TSAM, paired contract, acceptance checks |
| [expansion_costs.md](expansion_costs.md) | cable and transformer reinforcement rules and the cost assumptions with sources |

Other READMEs: [config/README.md](../config/README.md) (YAML checklist) and
[tests/regression/README.md](../tests/regression/README.md) (regression harness).
