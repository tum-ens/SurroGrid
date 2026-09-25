# Step 3: urbs optimization

Step 3 solves the building energy-system model (Pyomo, adapted urbs) of one grid: dispatch of all post cases and,
for `post-hems-optimized`, the investment in PV, battery, heat pump, auxiliary heater and buffer. It reads a
Step 2 file and writes the results into a copy of it.

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
| `--n_cpu` (alias `--partitions`) | 1 | number of building clusters; each is one model in its own process. **Changes the result**, it is not a CPU limit |
| `--cluster-concurrency` | `$URBS_CLUSTER_CONCURRENCY`, else all | clusters solved at the same time |
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
vertex or MIP solution: treat the solver as part of the scenario. A solve that does not terminate optimally fails
the run.

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
| `urbs_out/MILP/*` | Pyomo sets, parameters, variables and expressions merged across clusters (e.g. `tau_pro`, `cap_pro`, `costs`) |
| `urbs_out/solver_audit` | one row per cluster: solver, interface, version, options, status, termination, objective, best bound, gap |

Solver logs: `work/optimization/logs/<gurobi|appsi_highs>/<input stem>_<scenario key>_<cluster>.log`. The run
settings are printed at start; Step 4 copies a summary of the solver audit into its run assumptions.

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
