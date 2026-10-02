# Step 3: building optimization (urbs or PyPSA)

Step 3 solves the building energy-system model of one grid: dispatch of all post cases and, for
`post-hems-optimized`, the investment in PV, battery, heat pump, auxiliary heater and buffer. It reads a Step 2
file and writes the results into a copy of it. Two optimizers solve the same model: `urbs` (Pyomo, adapted urbs,
the default) and `pypsa` (PyPSA/linopy, several times faster; see [PyPSA optimizer](#pypsa-optimizer)).

Entry point: `gridexpand optimize` = `src/gridexpand/optimization/run_urbs_cluster.py`. Model code:
`src/gridexpand/optimization/urbs/` (trimmed urbs, GPL-3.0, see its `LICENSE`); solver selection and provenance:
`optimization/solver.py`; input identity checks: `optimization/identity.py`. Compared with urbs-lvds (4 February
2025) this version has no grid optimization, 14a/bui-react, uhp, coordination, curtailment, microgrids, CO2
limits, intertemporal support timeframes, reactive power, or Excel/LP outputs.

## Inputs

`inputfile_id` is a path to a file, an exact file name in `work/optimization/input/`, or the unique prefix before
the first underscore of one file there (an ambiguous prefix is an error). The orchestrators copy the Step 2 result
there. The file must contain:

- `urbs_in/*` (all tables of a post profile, [Step 2](2_demand_allocation.md#outputs)); the sites are the columns
  of `urbs_in/demand`;
- `metadata/timeframe` with `scenario_key`, `scenario_hash`, `model_case`, the assignment hashes and
  `profile_seed`;
- for post cases, `raw_data/electrification_assignment`.

Before solving, Step 3 checks that the Step 2 scenario hash equals the hash of `--scenario-config`, that the
assignment matches its recorded hash and the scenario's adoption rules, and that the scenario key is canonical for
the scenario and timeframe; it never resamples the technology adoption. Full-year inputs must carry no
type-period weights.

## Command line

```bash
uv run gridexpand optimize <inputfile_id> --scenario-config config/scenarios/<scenario>.yaml --n_cpu 16
```

| option | default | meaning |
|---|---|---|
| `--scenario-config` | required | scenario YAML; its hash must match the Step 2 input |
| `--optimizer` | `$GRIDEXPAND_OPTIMIZER`, else `urbs` | `urbs` or `pypsa` |
| `--n_cpu` (alias `--partitions`) | 1 | urbs: number of building clusters; each is one model in its own process. **Changes the result**, it is not a CPU limit. pypsa: not used (one model per building) |
| `--cluster-concurrency` | urbs: `$URBS_CLUSTER_CONCURRENCY`, else all; pypsa: CPUs / 4, at most one per 2 GB of available memory | models solved at the same time |
| `--solver` | `$GRIDEXPAND_SOLVER`, else `gurobi` | `gurobi` or `appsi_highs` |
| `--tsam` | scenario YAML | enable TSAM regardless of `time_aggregation.enabled` |
| `--tsam-periods`, `--tsam-hours-per-period`, `--tsam-extreme-method` | scenario YAML | TSAM overrides |
| `--reduce-only` | off | run preprocessing and TSAM, write `urbs_out/reduced_data` and `urbs_out/tsam`, skip the solve (needs TSAM) |

**Partition.** The buildings (sites) are split into `n_cpu` contiguous, equally sized clusters in site order (the
first `len % n_cpu` clusters get one more building; empty clusters are dropped). The synthetic runner chooses
`n_cpu` per grid (`--step3-cpus`, `--step3-max-cpus`, `--step3-target-columns`, see
[configuration.md](../configuration.md#pipeline-synthetic)).

**Solver.** `gurobi` is Pyomo's LP-file interface to Gurobi with `Method=4` (deterministic concurrent),
`MIPFocus=2`, `MIPGap=0.05`, `Presolve=2`, `Threads=4` per cluster (so up to `4 × concurrency` threads);
it needs a Gurobi installation and licence (`GUROBI_HOME`, `GRB_LICENSE_FILE`, see the
[README](../../README.md#install)). `appsi_highs` needs no licence (`mip_rel_gap=0.05`); the heuristic cases are
degenerate LPs and the optimized case a MIP stopped at a 5 % gap, so another solver can return a different optimal
vertex or MIP solution: treat the solver as part of the scenario. With the PyPSA optimizer, a Gurobi solve that
does not end optimal is solved once more with `Presolve=0`: Gurobi's presolve can declare a feasible internal-heat
model infeasible or unbounded. The solver audit records the options of the final solve. A solve that does not
terminate optimally fails the run.

**TSAM.** With `time_aggregation.enabled` (or `--tsam`), typical periods are selected from `Tamb` and
`Irradiation` only. TSAM is refused when the input has EV sessions (paired inputs), and the paired and aligned
scenarios run the full chronological year.

## Outputs

`work/optimization/result/<scenario key>/<input stem>_<scenario key>.h5`. The input is copied to `<result>.partial`,
results are appended, and the file is renamed only on success (removed on failure). Keys added:

| key | content |
|---|---|
| `urbs_out/temporal_method` | `temporal_method` (`full_year_no_tsam` or `shared_weather_tsam`), operating hours, `annual_weight`, storage and EV boundary policies, scenario key and hash (Step 4 and the paired checks read it) |
| `urbs_out/tsam/*` | `kept_timesteps` (always); with TSAM also `clusterOrder`, `clusterCenterIndices`, `hoursPerPeriod`, `noTypicalPeriods`, accuracy indicators, ... |
| `urbs_out/reduced_data/*` | the (possibly reduced) input tables used for solving, incl. `global_prop` (run settings) |
| `urbs_out/MILP/*` | urbs: Pyomo sets, parameters, variables and expressions merged across clusters (e.g. `tau_pro`, `cap_pro`, `costs`); pypsa: the [result keys](#result-keys) |
| `urbs_out/solver_audit` | one row per cluster (urbs) or building (pypsa): solver, interface, version, options, status, termination, objective, best bound, gap; pypsa adds `optimizer` and `optimizer_version` |

Solver logs: `work/optimization/logs/<gurobi|appsi_highs|highs>/<input stem>_<scenario key>_<model>.log` (`highs`:
the PyPSA optimizer with HiGHS). The run settings are printed at start; Step 4 copies a summary of the solver audit
(including `optimization_optimizer`) into its run assumptions.

## HPC

```bash
SCENARIO_CONFIG=config/scenarios/<scenario>.yaml sbatch scripts/hpc/optimization/run_cluster_serialstd.sh <inputfile_id>
SCENARIO_CONFIG=config/scenarios/<scenario>.yaml bash scripts/hpc/start_batch_jobs.sh optimization 0 24
```

The templates (adapt the `#SBATCH` header) run `uv run --frozen gridexpand optimize <INDEX> --n_cpu
$SLURM_CPUS_PER_TASK`; logs go to `work/runs/slurm/`.

## Conventions

- One urbs site per bus (synthetic) or per scenario unit (paired); all energies per hourly step, costs in EUR.
- Heuristic assets are fixed (`inst-cap` = `cap-up`, zero investment cost); optimized assets start at zero with
  finite building-specific upper bounds ([method.md](../method.md)).
- The row order of `urbs_out/MILP/e_co_buy` and `e_co_sell` depends on `PYTHONHASHSEED` (values do not).

Inspect a result:

```python
import pandas as pd
path = "work/optimization/result/<scenario key>/<file>.h5"
with pd.HDFStore(path, mode="r") as store:
    print([key for key in store.keys() if key.startswith("/urbs_out")])
print(pd.read_hdf(path, "urbs_out/solver_audit"))
```

## PyPSA optimizer

`gridexpand optimize --optimizer pypsa` (or `execution.optimizer: pypsa` in a run YAML, or
`GRIDEXPAND_OPTIMIZER=pypsa`) solves the model with PyPSA 1.2 and linopy instead of Pyomo. Code:
`src/gridexpand/optimization/pypsa_model/` (`building_model.py` model of the buildings, `constraints.py` urbs
extras, `results.py` result keys, `solve.py` one model, `runfunctions.py` partition and result file). It reads
the same Step 2 file, runs the same identity checks and writes a result file with the same layout, so Steps 4 and
5 are unchanged.

Step 3 contains no power grid: every building is an independent energy system. PyPSA calls its model container a
`pypsa.Network`; here a PyPSA *bus* is one energy balance of one building (`<site>|electricity`,
`<site>|common_heat`, `<site>|space_heat`, `<site>|water_heat`, `<site>|mobility<i>`), and no component connects
two buildings. The diagram of one building is in the docstring of `building_model.py`.

**Scope.** What the study's scenarios use: full-year chronological inputs, heuristic (LP) and optimized (MILP)
cases, the legacy mobility buffer (synthetic inputs) and dedicated EV sessions (paired inputs). Refused with
`NotImplementedError`, never approximated: TSAM type periods and `--reduce-only`; processes with several inputs
or outputs or an input ratio other than 1; sizing on top of an existing capacity; fixed or variable operating
costs of processes and fixed costs of storages (0 in every scenario); storages without power, or sized without an
energy/power ratio; a heat storage and its heat pump of which only one is sized.

**Mapping.** Same feasible set and objective as urbs:

| urbs | PyPSA |
|---|---|
| commodity balance per site (`res_vertex`) | one `Bus` per (site, commodity), except SupIm/Buy/Sell commodities |
| demand | `Load` |
| grid import (Buy commodity, `import` process) | `Generator`, marginal cost = buy price(t) × commodity price |
| feed-in (`feed_in` process, Sell commodity) | `Generator` with `p_min_pu = -1`, `p_max_pu = 0`; marginal cost = sell price(t): revenue for p < 0 |
| rooftop PV (SupIm, `e_pro_in == cap · supim`) | `Generator` with `p_min_pu = p_max_pu = supim(t)` (must-take, no curtailment) |
| heat pump, heating rod, heat dummies, legacy charging stations | `Link`, efficiency = output ratio × `eff_factor(t)` (COP, availability); charging stations also `p_max_pu = eff_factor(t)` |
| charger with dedicated EV sessions (no output) | electricity sink `Generator`, `p_min_pu = -fraction(t)` (0 outside sessions) |
| storage | `StorageUnit` (`max_hours` = energy/power, cyclic, self-discharge, efficiencies) |
| expansion with annuity, `cap-up` | `p_nom_extendable`, `capital_cost`, `p_nom_max` |
| fixed investment cost if built (`pro_cap_expands`) | binary `Generator-build`/`Link-build`, `p_nom − inst ≤ cap_up · build` (linopy) |
| variable cost of storage input | objective term on `StorageUnit-p_store` (PyPSA charges dispatch only) |
| heat storage ≤ ratio × heat-pump capacity | linopy constraint |
| EV session energy | one vectorised linopy constraint (group sum over the session hours) |

The model goes to the solver as matrices (linopy `io_api="direct"`, no names, no LP file), with the solver options
of the urbs optimizer. A worker process opens one Gurobi environment (one licence session) for all its models.

**Partition: one model per building.** The buildings of a grid share no constraint, so the optimum does not
depend on the partition; one building per model was the fastest partition in every measurement, needs the least
memory and applies the MIP gap to each building. Grid 84180/2/14 (72 buildings, full year, Gurobi):

| buildings per model | 1 | 2 | 5 | 9 |
|---|---|---|---|---|
| heuristic (LP), wall time / memory of all workers | **32 s** / 6 GB | 35 s / 8 GB | 49 s / 11 GB | 63 s / 11 GB |
| optimized (MILP) | **109 s** / 10 GB | 131 s / 12 GB | 153 s / 11 GB | 219 s / 11 GB |

(8, 8, 5 and 3 workers; the urbs optimizer needs 284 s and 766 s with 16 clusters on the same machine, and runs out
of memory with more than 2 concurrent optimized clusters.)

**Workers.** RAM is about 1 GB for the parent process plus 0.6 GB (LP) to 1.0 GB (MILP) per worker. On the development
VM (32 vCPUs) the throughput stops improving at 8 workers: heuristic 161/85/49/34/32 s and optimized
215/133/111/121 s for 1/2/4/8/12 workers (Gurobi; the optimized case from 2 workers). The default (no
`--cluster-concurrency`, and `step3_cluster_concurrency: null` in run YAMLs) is one worker per 4 CPUs and per 2 GB
of available memory, 8 there. Runs that process several grids at once (`workers` > 1) should set
`step3_cluster_concurrency` so that `workers × step3_cluster_concurrency` stays near that value. An explicit
`step3_cluster_concurrency: 1` (as in the repository's run YAMLs, chosen for urbs) solves one building at a time. Workers free each model and return the memory to the
operating system after every solve; a worker then peaks at about 1 GB (LP) to 1.8 GB (MILP).

### Result keys

The PyPSA optimizer writes only the keys that are consumed or that cannot be derived, with the index names, order
and units of the urbs entities:

| key | index | used by |
|---|---|---|
| `tau_pro` | t, stf, sit, pro | Step 4 (import, feed-in, heat pump, PV), the runners |
| `cap_pro` | stf, sit, pro | Step 4 (INFLEX heat split, PV, installed assets), analyses |
| `cap_sto_c`, `cap_sto_p` | stf, sit, sto, com | Step 4 (installed assets per building), analyses |
| `e_sto_in`, `e_sto_out` | t, stf, sit, sto, com | storage dispatch (HEMS analyses) |
| `e_sto_con` | t (incl. 0 = last step, cyclic), stf, sit, sto, com | storage state of charge |
| `costs` | cost_type | cost breakdown (Invest, Fixed, Variable, Revenue, Purchase) with urbs' definitions |

The other urbs entities follow from these and `urbs_out/reduced_data`: `e_pro_in`/`e_pro_out` (`tau_pro` times the
ratios and `eff_factor`), `e_co_buy`/`e_co_sell` (`tau_pro` of `import`/`feed_in`), `*_new` (`cap_*` minus
`inst-cap`), `pro_cap_expands` (`cap_pro_new > 0`).

### Same results as urbs, and when hourly results differ

`tests/optimization/test_pypsa_equivalence.py` solves a scenario with a unique optimum
(`tests/optimization/deterministic_fixture.py`: three buildings with every asset, heuristic LP, optimized MILP
and EV sessions) with both optimizers; objective, cost breakdown, capacities and every hourly value of all result
keys are equal (within 1e-5), and the uniqueness of the optimum is certified on the PyPSA model. On eight real
buildings (full year) with the same tie-breaks, urbs (one model) and PyPSA (eight models) agree to 1e-14.

With the real scenarios the objective, all capacities, the use of every storage and the annual import of every
building are identical, but the hourly split of surplus PV can differ. Self-consumption is optimized: PV that a
storage can use profitably is stored the same way in every solution. What is left has no value at a feed-in
tariff of 0 EUR/kWh, and the model can feed it in or run the heating rod instead of the heat pump. Both cost
nothing, so the LP has many optimal solutions and every solver, interface or partition returns another one (urbs
does the same when the solver changes). With a constant import price, the hour in which stored energy is used is
free as well, so hourly imports differ while annual imports do not. On grid 84180/2/14 (heuristic case, both
optimizers with Gurobi) the annual feed-in is 147.3 vs 146.3 MWh.

**Charging stations (legacy mobility buffer).** Their `eff_factor` is the connected share of the hour. It scales
the output and, since 2026-09-28, also limits the input (`tau_pro <= dt * cap_pro * eff_factor`, urbs
`res_process_availability`, PyPSA `p_max_pu`). Before that a station could draw electricity with zero output while
the car was away, a free sink for surplus PV at a tariff of 0: on grid 84180/2/14 urbs lost 43.9 MWh a year this
way (objective unchanged). Chargers of dedicated EV sessions are limited by their sessions; a charger without
sessions is limited by its `eff_factor` in the same way.

A small positive feed-in tariff removes the choice for the surplus: with 0.001 EUR/kWh, urbs + Gurobi, PyPSA +
Gurobi and PyPSA + HiGHS give identical annual flows (feed-in 147.9 MWh; measured before the charger fix), but the
hours still differ (import in 6-12 % of the building-hours, peak import 959-995 kW), because with a flat import
price and a flat tariff it is free when a car or battery charges from the grid and which surplus hours are stored.
Unique hourly results need prices that differ from hour to hour or an explicit tie-break rule; the deterministic
fixture shows tie-breaks that make the optimum unique (prices and tariffs that differ in every hour, and storage
costs and self-discharge that differ per storage and vehicle).

### Dependencies and environment

- PyPSA 1.2.x (`pypsa>=1.2.4,<1.3`) and linopy 0.9.x. PyPSA 1.3 requires pandas 3, but pandapower (Step 4) declares
  `pandas~=2.3` up to its latest release (3.5.5, 2026-09-22) and pandapower 3.1 fails under pandas 3
  (copy-on-write makes `.values` read-only); move to PyPSA 1.3 together with a pandapower that supports pandas 3.
- highspy >= 1.15: linopy excludes highspy 1.14.0 ("wrong results due to broken presolve"; HiGHS issue 2957,
  an incorrect MIP optimum). This also applies to the urbs optimizer's `appsi_highs`.
- polars (a linopy dependency) warns on import that the CPU lacks `pclmulqdq` when the machine is a KVM guest whose
  CPU model hides that flag (the development VM, and containers on it). The host CPU executes the instruction
  (checked), so the warning is a false positive there, and the direct solver hand-off does not use polars. Set
  `POLARS_SKIP_CPU_CHECK=1` to silence it, or give the VM a CPU model that exposes the flag (e.g. host
  passthrough), which also lets other libraries use their faster CRC/GCM code paths. On a CPU that really lacks
  the instruction, install `polars[rtcompat]`.
