# GridExpand documentation

Start with the [GridExpand README](../README.md) (install, data, database, quick starts, commands).

## Using GridExpand

| document | read it when you |
|---|---|
| [configuration.md](configuration.md) | write a scenario or run YAML; need a key, a default, the grid selection, the run directory or an environment variable |
| [paired_validation.md](paired_validation.md) | run the paired SWF or the aligned SWF/ÜZW validation |
| [service.md](service.md) | run the web service, the pylovo-ui plugin or the Docker image |
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

## Research notes (historical records, not maintained)

| note | topic |
|---|---|
| [2026-07-20_forchheim_paired_battery_tsam_run.md](research/2026-07-20_forchheim_paired_battery_tsam_run.md) | summary of the paired Forchheim battery/TSAM run (pylovo v3) |
| [2026-07-21_failed_real_swf_grids.md](research/2026-07-21_failed_real_swf_grids.md) | failed real SWF grids LV 38, 47, 113 |
| [2026-07-21_forchheim_ghd_calibration_v5.md](research/2026-07-21_forchheim_ghd_calibration_v5.md) | GHD and mixed-use evidence audit (pylovo v5) |
| [2026-07-21_synthetic_real_equipment_capacity.md](research/2026-07-21_synthetic_real_equipment_capacity.md) | transformer and cable capacities, synthetic vs real |
| [2026-07-24_feeder_structure_comparison.md](research/2026-07-24_feeder_structure_comparison.md) | feeder structure, synthetic vs real |
| [2026-07-24_forchheim_paired_v5_audit.md](research/2026-07-24_forchheim_paired_v5_audit.md) | paired scope audit, LV113 diagnostic, commands of that time (pylovo v5) |
| [2026-07-24_street_path_junction_refinement.md](research/2026-07-24_street_path_junction_refinement.md) | open pylovo generator issue (street-path junctions) |
| [2026-08-14_forchheim_heat_asset_pilot.md](research/2026-08-14_forchheim_heat_asset_pilot.md) | heat-asset pilot on one Forchheim grid |
| [2026-09-10_forchheim_2045_paired_v11_forensic_audit.md](research/2026-09-10_forchheim_2045_paired_v11_forensic_audit.md) | forensic audit of the paired v11 run |
| [2026-09-10_reactive_power_sign_error.md](research/2026-09-10_reactive_power_sign_error.md) | the original reactive-power sign error (fixed) |
| [2026-09-24_teaser_vs_infdb_ro_heat.md](research/2026-09-24_teaser_vs_infdb_ro_heat.md) | TEASER vs INFDB `ro_heat` space heat |

New research notes: `research/<YYYY-MM-DD>_<topic>.md`, starting with a header that names the run id, the pylovo
version, the date and "historical record". Agent plans and handovers do not belong in the repository (see
`AGENTS.md`).

Other READMEs: [config/README.md](../config/README.md) (YAML checklist),
[tests/regression/README.md](../tests/regression/README.md) (regression harness),
[notebooks/archive/README.md](../notebooks/archive/README.md) (archived notebooks).
