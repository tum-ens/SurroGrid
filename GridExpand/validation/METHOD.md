# Paired validation method

The scientific contract of the real-vs-synthetic validation ([README.md](README.md)). The scenario method itself
(model cases, sizing, heat, mobility, GHD, prices) is GridExpand's and is described in
[docs/method.md](../docs/method.md).

## Contract

The aligned paired pipeline compares the real grids of the DSOs in a pylovo alignment bundle (DSO A and DSO B) with
synthetic pylovo grids of the same buildings under the same post-electrification scenario. It holds the physical
demand and technology realization constant and changes only the electrical network and the mapping of scenario units
to buses. Paired validation may project a
compiled scenario onto real and synthetic buses; it may not redefine scenario assumptions.

| Layer | Shared between real and synthetic | Network-specific |
|---|---|---|
| Scope | the physical buildings of the alignment bundle's cohort, same minimum-building rule | grid partition |
| Base demand | building components of the open building data (households, GHD floor area), sampled profile realization | allocation bus |
| Sector assets | the deterministic electrification assignment of the scenario | allocation bus |
| Time series | profile seed, weather, mobility pool, heat demand, COP, temporal horizon | none |
| Optimization | identical scenario-unit inputs, formulation, technology assumptions, solver settings, temporal method | target-grid batch partition |
| Power flow | active-power time series and metric definitions | pandapower topology, impedance, equipment, bus mapping |

The comparison fails before optimization when the paired plans contain different physical buildings or different
household/GHD totals.

**Scenario unit.** A physical building can be associated with several real connection buses, so the stable
demand-allocation unit is `(source_lv_id, source_allocation_bus, building_objectid)`. Optimization stays at
scenario-unit resolution (`optimization_space=scenario_unit`); aggregating units to buses before urbs could
change flexibility. Both Step 4 paths read `raw_data/allocation_plan` and aggregate the unit profiles onto the
selected target buses immediately before the power flow; this conserves active and reactive demand. Each
target-grid batch recreates the same deterministic scenario-unit inputs; only the grouping and the final bus
projection differ.

**One HEMS per building.** The alignment bundle maps every canonical building to one real grid and one bus
(pylovo's alignment rejects buildings with conflicting connections), so every building is one HEMS.

**Reference grids.** DSO A grids are radialized station workbooks with one transformer each (cycle rank 0, no
unsupplied load buses); DSO B grids are area deliveries. The alignment bundle names the grid file of every real grid.

## Status-quo test

Before pylovo generation, `python -m validation status-quo-test` checks every real grid of a provider under its status-quo
base electricity (households and GHD of the model above, with the scenario's `ghd:` rules) and writes the exclusion
list: the alignment removes the flagged real grids and their buildings from both sides, so that the comparison is
not dominated by grids that already fail today.

*Demand.* The alignment bundle gives the real grid, bus and grid file per building; the building attributes come
read-only from InfDB `basedata.buildings`. The pylovo derivations the model needs before pylovo has run mirror pylovo
8afedcc: the non-residential use (Mixed buildings take the class of their function, Residential ones Commercial),
missing households by type (1 for SFH/TH, area/181 m² for MFH and untyped residential with at least 2 and 1, area/146
m² for AB with at least 5), component peaks (16.825 kW per household; 79, 29 and 29 W/m² for Commercial, Public and
Unknown) and MV-direct above 100 kW; missing occupants are households × 2.03. Components, sampling, annual energies,
seeds and the paired profile builder are those of the aligned allocation. `--dataset` takes the demand of a prepared
paired dataset instead.

*Power flow and flags.* Step 4's grid preparation and summary: LV busbar at 0.96 p.u. with up to two off-load tap
steps, Newton-Raphson with the Iwamoto fallback, non-converged hours recorded. A grid is flagged for
`transformer_overload` (> 100 % of the rating; skipped for grids without a rating), `cable_overload` (a cable above
100 % of its rated current), `undervoltage` (a load bus below 0.90 p.u. after the taps) or `nonconvergence`. The
evaluated hours are the 60 largest hours of the grid total plus the 24 largest household and GHD hours (`--all-hours`
evaluates the year). On the prepared joint datasets this reproduces the full-year `pre` stage of the
aligned run: the same transformer loadings (to 0.0 percentage points) and the same flagged grids. A grid without a
transformer rating is checked for cables, voltage and convergence only.

*Output* (per provider): `status_quo_<provider>_exclusions.csv` with the columns `provider, real_grid_id, reason`
(`real_grid_id` is the bundle's own id, DSO A `station_id` such as `LV_059` and DSO B `area_id`; `reason` joins the
flags with `;`), `status_quo_<provider>_grids.csv` (demand and power-flow metrics of every grid),
`status_quo_<provider>_ghd_audit.csv` and `status_quo_<provider>_metadata.json`.

## Shared profile libraries

Physical heat and COP series are generated once per profile assumption set and stored in a network-independent
regional HDF5 library keyed by `building_objectid`; readiness checks coverage against it and the urbs inputs of
both targets read the same profiles before projecting them to buses. Topology changes can reuse a library; changes
of refurbishment, weather or other heat assumptions need a new profile-set id (aligned runs derive it as
`<paired_dataset_id>_teaser_heat_<scenario hash 12>`). Diagnostic heat fallbacks
(`--allow-diagnostic-heat-fallback`) are not publication results. The PV profile library is built the same way
(see [Rooftop PV potential](../docs/method.md#rooftop-pv-potential)).

## Publication criteria

A publication run requires:

1. equal paired-plan buildings and demand totals;
2. no diagnostic heat-profile fallbacks;
3. identical scenario-unit inputs and optimization settings for both targets;
4. an identical temporal horizon and, with TSAM, identical representative periods for both targets (the runner
   records one canonical mapping and rejects results that do not reproduce it);
5. no silently skipped power-flow timesteps: non-convergence is reported (real grids with failed timesteps in 1 %
   of the hours or more are `incomplete` in the expansion analysis, not zero-cost; below that they are costed from the
   converged hours as a lower bound);
6. a separate meshed-versus-radial sensitivity for critical real grids where radialization materially changes
   stress.
