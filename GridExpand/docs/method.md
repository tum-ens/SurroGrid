# Scientific method

This is the record of the scientific choices shared by the synthetic pipeline and the paired
validation. Adjustable values live in the scenario YAMLs (`config/scenarios/`, key reference in
[configuration.md](configuration.md)); the reasoning and the cross-stage contracts live here. Values
quoted below are those of every repository scenario YAML unless a table says otherwise.

| Scenario YAML | used by | heat source | adoption (heat / mobility / PV+battery) | TSAM |
|---|---|---|---|---|
| `config/scenarios/default_2045.yaml` | default of every run (the paper's final method) | `internal` (1R1C, HEMS preheating 2 K) | deterministic 0.8 / 0.5 (by type) / 0.7 | off |
| `validation/config/scenarios/joint_2045_full_year.yaml` | the real-vs-synthetic validation (paper) | `teaser` (level 1) | deterministic 0.8 / 1.0 / 0.8 | off |

The test fixtures in `tests/fixtures/scenarios/` vary these settings (`ro_heat_2045_synthetic.yaml`,
`teaser_2045.yaml` with `teaser` level 0 and 0.75 shares, a copy of the validation scenario).

## Model cases

| Case | Asset sizing | Operation | Step 2 profiles |
|---|---|---|---|
| `pre` | none (reference assets) | reference electricity demand | `status_quo` |
| `post-inflex-heuristic` | shared heuristic asset plan | rule-based dispatch (INFLEX) | `all` |
| `post-hems-heuristic` | shared heuristic asset plan | optimized dispatch (HEMS) | `all` |
| `post-hems-optimized` | endogenous urbs sizing | optimized dispatch (HEMS) | `all` |

The table is `gridexpand.scenario.model_cases.MODEL_CASES`, the only place that encodes these rules.
Requested cases are grouped into shared solves (`execution_groups`): `pre` alone; the two heuristic
cases share one Step 2 materialization and one Step 3 solve of `post-hems-heuristic`;
`post-hems-optimized` has its own. Only the first group emits the `pre` power-flow stage.

The controlled comparison of flexibility strategies is `post-inflex-heuristic` versus
`post-hems-heuristic`: both dispatches use the same materialized heuristic asset plan, so their PV,
battery, heat-pump, auxiliary-heater and buffer capacities are identical and only operation differs.
`post-hems-optimized` versus a heuristic case compares assets and dispatch together.

The synthetic pipeline cannot run `post-inflex-heuristic`: INFLEX needs the EV session table
(`urbs_in/ev_sessions`), which only the paired pipeline writes; `gridexpand run`, `gridexpand
synthetic` and the service refuse the case up front.

PV, stationary-battery and residential heat sizing exist in heuristic and optimized modes. The heat
method compiles one central system per physical residential building; commercial heat-pump sizing is
outside the present method.

## Configuration ownership

The scenario YAML is the single source of truth for scientific assumptions: prices, asset sizing,
electrification adoption, mobility behaviour, every urbs process and storage parameter, and time
aggregation. The run YAML owns the input (region, pylovo topology version, prepared datasets), the
model cases, the profile-realization seed and the execution resources. The seed selects one Monte Carlo
realization; it does not change the distributions defined by the scenario. Python keeps only
implementation references (data locations, API endpoints). For paired datasets, the run's pylovo
version must match the version recorded when the dataset was prepared.

The scenario identity is `scenario_<scenario.id>_<first 12 characters of the scenario hash>`, where the
hash is the SHA-256 of the parsed YAML (comments and formatting do not count, every key and value does).
Step 2 records it, Step 3 refuses inputs of another scenario, and it names the result directories, the
database scenario rows and the synthetic expansion analyses.

## Time axis

Every time axis is the reference year 2009 in fixed UTC+1 (CET without daylight saving time);
`mobility.reference_year` must be 2009. Weather is the PVGIS SARAH3 typical meteorological year moved to
2009. Behaviour-driven generators (occupancy and internal gains, EV use) and civil-time load profiles are
shifted around the 2009 daylight-saving transitions (the skipped hour of 29 March, the repeated hour of 25
October) and mapped back to UTC+1 (`allocation/functions/dst.py`). Full-year runs have 8,760 hours; the
one-week timeframes (`min_temperature_week`, `max_solar_radiation_week`, `max_base_electricity_demand_week`)
select 168 hours and are operational stress screenings, not annual investment optima.

An aligned paired run can take one real calendar year instead of the TMY (`resources.weather_year`): PVGIS
`seriescalc` hourly data from the same sources (SARAH3 irradiance, ERA5 meteorology) on the horizontal plane,
with `dni = Gb / sin(sun height)`, 29 February dropped and the year moved to 2009. One year for all providers
keeps irradiance and temperature physically consistent and gives every region the same meteorological year;
a TMY selects each month per site and can pick an atypically mild or cold winter. The series has no humidity,
pressure or wind direction, which the paired pipeline does not use.

## Reproducible profile realization

Stochastic input realization is separated from asset sizing and dispatch. Every stochastic choice uses a
sub-seed `stable_seed(profile_seed, physical building id, component, ...)` derived from the run-level
`profile_seed` (default 481527) and the topology-independent physical building id. Independent sub-seeds
cover household occupancy, annual household electricity, non-residential use type, heat-system attributes,
TEASER occupancy and hot-water draws (seeded per building since 2026-09-25), each OpenDHW flat, vehicle
ownership, vehicle model and schedule, and mobility-profile selection. Bus ids and row order are
deliberately excluded, so heuristic and optimized cases of the same grid, and real and synthetic targets of
a paired run, share one realization, and results do not depend on the CPU partition.

Every Step 2 HDF records `profile_seed`, `profile_realization_id` (seed and physical building inventory,
not the model case) and a fingerprint `profile_hash_<component>` for each generated component (base
electricity and, where selected, space heat, hot water, heat-pump COP, mobility demand and availability) in
`metadata/timeframe` and in the Step 2 run assumptions. The two heuristic cases additionally require identical PV, battery and heat asset
plans; optimized capacities differ by design. No runner compares the fingerprints of different cases
automatically since the fixed-grid smoke runners were removed; compare them from `metadata/timeframe` when
a controlled comparison matters.

## Electrification assignment

`electrification.<heat|mobility|pv_battery>.adoption_mode: deterministic_share` ranks the eligible physical
buildings of each technology by a stable seed and selects the exact rounded `building_share`.

With `building_share_by_type` (keys `SFH`, `TH`, `MFH`, `AB`) the ranking and the rounded selection run within
each building type; `building_share` applies to the types not listed (e.g. untyped mixed-use buildings). The
manifest then carries a `selection_stratum` column, and Step 3 validates the share per stratum; manifests
without per-type shares keep their columns and hashes. The joint scenario uses this for home charging:
0.85 for one- and two-family and terraced houses, 0.5 for multi-family houses, from a battery-electric fleet of
about 93 % in 2045 ([dena-Leitstudie, 2021](https://www.ewi.uni-koeln.de/de/publikationen/dena-ls2/211005_ewi-zusammenfassung_dena-leitstudie-aufbruch-klimaneutralitaet))
and home-charging access of about 85 % and 50 %
([dena, 2020](https://www.dena.de/fileadmin/dena/Publikationen/PDFs/2020/dena-STUDIE_Privates_Ladeinfrastrukturpotenzial_in_Deutschland.pdf);
[MiD 2017](https://www.mobilitaet-in-deutschland.de/archive/pdf/MiD2017_Ergebnisbericht.pdf)).
`pv_battery.battery_share_of_selected` gives a battery to a seeded share of the selected PV buildings (own
ranking stream `battery`); 1.0 is the PV + battery bundle. The joint scenario selects PV on 70 % of the eligible
buildings and a battery on 80 % of them (about 80 % of new home PV systems are sold with a battery,
[BSW-Solar, 2026](https://www.solarwirtschaft.de/datawall/uploads/2025/10/BSW-Solar_Faktenblatt_Stromspeicher.pdf)),
and heat pumps on 80 % of the residential buildings. The study grids are treated as areas without district heating:
outside district heating, national climate-neutrality scenarios reach 69–93 % heat pumps in 2045 (own calculation,
heat pumps / (all heating systems − heat network), from Fig. 4.2 of the
[Ariadne report 2025](https://ariadneprojekt.de/media/2025/03/Ariadne-Report_Szenarien2025_Maerz2025_lowres.pdf)
and Tables 1-5 and 1-7 of the
[Langfristszenarien O45 building report](https://www.ifeu.de/fileadmin/uploads/Publikationen/Energie/Langfristszenarien/20240924_Bericht_Gebaeude_LFS3_O45_clean.pdf)).
Heat of the other residential buildings and of all non-residential buildings is not electrified, i.e. it is assumed
to come from non-electric sources such as biomass or renewable gas.

The assignment is one manifest per region, written before Step 2 and reused unchanged by Step 3 (which
checks its hash). For synthetic runs the region is the selected scope of the run (the AGS, one PLZ or one
grid, see [configuration.md](configuration.md#grid-selection)). A selected PV+battery row is an investment
candidate; its valid sizing may still be zero, including zero battery capacity. Only positive genuine LoD2
roof capacity makes a building eligible for `pv_battery`; the PV fallback never broadens the denominator.
Exclusion reasons are recorded per building (`raw_data/electrification_assignment`, database table
`electrification_assignment`), roof reasons first.

## Rooftop PV potential

All pipelines use CityDB LoD2 roof sections joined through the pylovo building object id; the required
CityDB properties are `Flaeche`, `Dachneigung` and `Dachorientierung`. Random roof sampling is not an
accepted data source. For roof section $r$ of building $i$ the available peak capacity is

$$
P_{\mathrm{PV,max},i,r}=A_{i,r}\,u_r\,\rho_{\mathrm{PV}},
\qquad
P_{\mathrm{PV,max},i}=\sum_r P_{\mathrm{PV,max},i,r},
$$

with the LoD2 roof-surface area $A_{i,r}$, the usable-area fraction $u_r$ and the module peak power per
roof area $\rho_{\mathrm{PV}}$. The scenarios use $u_r=0.27$ for flat and $u_r=0.58$ for sloped roofs
(`flat_roof_utilization`, `slanted_roof_utilization`), following
[Mainzer et al. (2014)](https://doi.org/10.1016/j.solener.2014.04.015), and
$\rho_{\mathrm{PV}}=0.202$ kWp/m² (`module_capacity_kw_per_m2`) from the module statistics of
[Kräling et al. (2022)](https://doi.org/10.4229/WCPEC-82022-3BO.14.1). Because $A_{i,r}$ is already the
inclined surface, no footprint or tilt correction is applied. The pvlib surface tilt is
$90^\circ-\text{Dachneigung}$; flat sections use azimuth 0°, and sections with an invalid tilt or
orientation are excluded and audited.

A building without any usable LoD2 section receives one 14.5 kWp fallback section at 45°/180°
(`fallback_capacity_kwp`, a legacy mean); every use is reported and checked against
`maximum_fallback_share`, which is 0.0 in every repository scenario.

Profiles are computed with [pvlib](https://doi.org/10.21105/joss.05994) once per binned tilt and azimuth
(`tilt_bin_degrees` 5°, `azimuth_bin_degrees` 15°); the largest observed annual-yield deviation of these
bins against exact LoD2 angles was 9.30 % for the roofs of the DSO A region. Capacity always uses the exact
surface area. Paired runs build one angle-binned profile library (`paired_pv_profile_library.h5`) before
the grid jobs start, so real and synthetic grids use identical normalized profiles.

## PV sizing

Heuristic PV capacity is computed per physical building, before heat and mobility electrification and
before timeframe selection or TSAM:

$$
P_{\mathrm{PV},i}=\min\left(\alpha_{\mathrm{PV}}\frac{E_{\mathrm{el},i}}{1000},\;P_{\mathrm{PV,max},i}\right),
\qquad \alpha_{\mathrm{PV}}=2.0\ \text{kWp per MWh/a (demand\_multiplier)},
$$

where $E_{\mathrm{el},i}$ is annual appliance-and-lighting electricity in kWh/a. Public consumer
recommendations span different ambitions: [Enpal](https://www.enpal.de/photovoltaik) starts from about
1 MWh per kWp and recommends some oversizing,
[Vattenfall](https://www.vattenfall.de/infowelt-energie/solar/pv-anlage-dimensionierung) recommends
1.5–2 kWp per annual MWh, and [1KOMMA5°](https://1komma5.com/de/solaranlage/dimensionierung-pv-anlage/)
publishes 2.5. The scenario selects 2.0 as a central compromise (upper end of Vattenfall, below
1KOMMA5°); these are consumer recommendations, not normative design rules. Applying the coefficient to all
building types, one shared system per building, and excluding heat-pump and mobility demand (so the PV
inventory stays independent of the later electrification realization) are study choices; the LoD2
potential is the hard upper bound.

Roof bins are filled in descending specific yield until the target is met; one capacity-weighted profile
represents the building system. Heuristic capacity is fixed in urbs (`inst-cap` = `cap-up`) with zero
investment cost because sizing happened upstream. Optimized sizing uses one process per building with zero
installed capacity and the physical maximum as `cap-up`, so the fixed PV investment cost is charged once per
building; the optimized process scales the building's roof mix proportionally and does not choose single
orientations. In paired runs, `post-hems-optimized` aggregates roof sections of one angle bin into one urbs
process bounded by their summed LoD2 `cap-up`.

## Stationary-battery sizing

Only buildings selected for `pv_battery` are battery candidates. With annual base
electricity $E_{\mathrm{el},i}^{\mathrm{MWh}}$ in MWh/a and PV capacity $P_{\mathrm{PV},i}$ in kWp, the
heuristic is

$$
C_{\mathrm{bat},i}^{\mathrm{use}}=
\begin{cases}
0, & P_{\mathrm{PV},i}\leq 0.5\,E_{\mathrm{el},i}^{\mathrm{MWh}},\\
\min\left(\alpha_{\mathrm{bat,PV}}P_{\mathrm{PV},i},\ \alpha_{\mathrm{bat,E}}E_{\mathrm{el},i}^{\mathrm{MWh}}\right), & \text{otherwise,}
\end{cases}
$$

with the threshold 0.5 kWp per MWh/a (`minimum_pv_kwp_per_annual_mwh`) and
$\alpha_{\mathrm{bat,PV}}=\alpha_{\mathrm{bat,E}}=1.0$ (`heuristic_usable_kwh_per_pv_kwp`,
`heuristic_usable_kwh_per_annual_mwh`).

Figure 22 of the [HTW Stromspeicher-Inspektion 2025](https://solar.htw-berlin.de/wp-content/uploads/HTW-Stromspeicher-Inspektion-2025.pdf)
supplies the eligibility threshold and recommends 1.5 kWh/kWp and 1.5 kWh/MWh as upper limits for usable
home-storage capacity (the loader rejects coefficients above 1.5). The study selects 1.0 as a less aggressive
central value because it extrapolates a home-storage recommendation to all residential building types and
uses appliance-and-lighting demand only; 0.75 and 1.5 are study-defined sensitivities. Open sizing
heuristics support the central value: [HTW Berlin (2014)](https://solar.htw-berlin.de/publikationen/auslegung-pv-speicher-einfamilienhaus/)
finds 1 kWh usable per kWp sensible for high self-sufficiency, the
[Bavarian LfU/C.A.R.M.E.N. guide (2022)](https://www.carmen-ev.de/wp-content/uploads/2022/02/Zukunftsloesungen-fuer-PV-Anlagen.pdf)
recommends about 0.7–1.0 kWh/kWp and at most 1 kWh per MWh of household demand, and
[Vattenfall](https://www.vattenfall.de/infowelt-energie/solar/lohnt-sich-pv-anlage) describes 1 kWh per kWp
and per MWh as a frequent rule. [HTW Berlin (2022)](https://solar.htw-berlin.de/publikationen/auslegung-von-solarstromspeichern/)
warns that a PV-only 1:1 rule can oversize storage; the minimum of both terms follows that two-sided logic.
It is not a DIN or VDI standard. For multi-household buildings the result is one shared system, audited per
building and per household.

A study-defined 2 h energy-to-power ratio (`energy_to_power_hours`) sets symmetric charge and discharge
power $P^{\mathrm{ch,max}}=P^{\mathrm{dch,max}}=C^{\mathrm{use}}/2\,\mathrm{h}$.

In both heuristic cases $P_{\mathrm{PV},i}$ is the fixed heuristic PV capacity, and usable energy is fixed
(`inst-cap-c` = `cap-up-c`) without investment cost. In `post-hems-optimized` the battery upper bound uses
the building's LoD2 maximum PV potential (the optimized PV capacity is unknown during input preparation) and
the coefficients 1.5 (`optimized_upper_*`) define only `cap-up-c`; `inst-cap-c` is zero and urbs chooses the
capacity. The heuristic capacity and the optimized upper bound are intentionally different quantities.

The optimized case uses 300 EUR/kWh (`technologies.storages.stationary_battery.investment_cost_eur_per_kwh`).
An earlier thesis of this project used 976 per kWh; the cited
[IRENA 2022 report](https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2022/Mar/IRENA_Tech_Innovation_Indicators_2022_.pdf)
gives that value as the 2021 German median installed price in 2020 USD/kWh, not a 2045 euro projection.
Scenario variants should use 250 and 365 EUR/kWh as low and high sensitivities; they span the projected 2040
household range of the [JRC 2018 report](https://op.europa.eu/en/publication-detail/-/publication/e65c072a-f389-11e8-9982-01aa75ed71a1).
This is an explicit extrapolation to 2045 and to shared multi-household systems.

INFLEX (`post-inflex-heuristic`, paired runs) operates the fixed battery with causal local self-consumption
control without forecasts: PV first supplies simultaneous demand, surplus charges the battery, stored energy
later covers residual demand; no grid charging and no battery export.

## Residential heat assets

### Scope and demand representation

The heat method applies to `SFH`, `TH`, `MFH` and `AB` buildings and represents one central heat system per
physical building. Synthetic runs keep non-residential electricity but add no commercial heat demand or heat
pumps; paired validation uses residential heat-pump inventory rows only.

The space-heat source is `asset_sizing.heat.space_heat_source`:

- `infdb_ro_heat` (DSO A region scenarios): building-level hourly heating loads from the INFDB `ro_heat` schema
  (1R1C model), converted from W to hourly kWh. The source holds 8,736 contiguous hours (1 January to
  30 December 2023); as a preliminary rule the last available day is duplicated to form 8,760 hours, and every
  affected building is audited. The library must be rebuilt when a complete export is available. A missing
  single series does not abort preparation: the adapter takes the available building of the same type with the
  nearest total floor area and scales by the area ratio, then broadens to the nearest-area building of the whole
  region (paired preparation uses the tiers same grid and type, same grid and use, same grid, same type, same
  use, region; it uses the LoD2 footprint as size proxy). Source building, match scope and scale are stored.
  These are approved pragmatic fallbacks, not exact simulations.
- `teaser` (Schweinfurt and joint DSO A + DSO B scenarios): TEASER `tabula_de` archetypes with the component's
  heated reference area `heated_area_ratio` × the component's residential effective floor area (footprint ×
  `floor_number`, residential share) as net leased area; the number of floors passed to TEASER does not change
  the result. The TEASER envelope design load is not used for sizing (see below). Occupancy (richardsonpy) and hot
  water draw on seeded random states per building. Standalone synthetic runs simulate with a PVGIS TMY at the
  grid's transformer. Paired runs pass the provider weather file (`--weather-hdf`), so heat, COP, PV and sizing
  share one TMY; a TMY takes each month from a different year, so TMYs of two nearby locations are unrelated
  hour by hour. The 5R1C model heats to a constant 20 °C. The envelope, area and solar assumptions are set by
  `asset_sizing.heat.teaser` (see [TEASER envelope and assumptions](#teaser-envelope-and-assumptions)); with
  `preset: legacy` the area is the full gross area and heating is off on days 135–258, so all buildings
  restart together on 16 September, which can set the auxiliary-heater size.
- `internal`: corrected ro-heat envelope coefficients,
  weather and gains feed a 1R1C temperature balance. HEMS determines space heating
  within comfort bounds; InFlex uses an independent forward thermostat reference.
  See [Internal space heat and building preheating](#internal-space-heat-and-building-preheating)
  for the effective heated zone, source provenance and boundaries.

#### TEASER envelope and assumptions

`asset_sizing.heat.teaser` (read only by `space_heat_source: teaser`) sets the assumptions of the TEASER path.
`preset: corrected` is the default; `preset: legacy` reproduces the TEASER profiles of SurroGrid 2.0.0 bit for
bit (test `tests/allocation/test_teaser_heat.py`). Single keys can be set on top of either preset. The
internal 1R1C model does not read this block.

| Key | Corrected (default) | Legacy | Why |
|---|---|---|---|
| `heated_area_ratio` | 0.55 | 1.0 | TEASER scales every element, the air volume and the internal masses with `net_leased_area`, which is TABULA's conditioned reference area (≈ 1.0–1.1 × living area). InfDB's gross area counts floors from the ridge height (≈ 4 m per floor, attics as full storeys); Bavaria-wide the occupied living area is ≈ 0.55 × this gross area (below) |
| `envelope_source` | `lod2` | `tabula` | walls = Σ LoD2 WallSurface − shared walls − windows (TEASER's orientation split), roof = Σ LoD2 RoofSurface, ground floor = footprint, as in InfDB ro_heat; buildings without stored surfaces keep the TABULA envelope |
| `subtract_shared_walls` | true | – | LoD2 WallSurface includes party walls (Bavarian sample, residential neighbours: TH 31 %, AB 22 %, MFH 12 %, SFH 11 % of the wall area) |
| `attic`, `attic_ceiling_b`, `pitched_roof_area_ratio` | `unheated`, 0.8, 1.05 | – | the LoD2 hull encloses the attic, TABULA's envelope only the heated zone: for a pitched roof (roof surface > 1.05 × footprint) the top-floor ceiling (= footprint, b = 0.8 as for a ceiling to an unheated attic, no solar or sky radiation) replaces the roof surface and the gable walls above the eaves leave the walls; flat roofs keep the roof. Eaves as in InfDB `05_fill_floor_number.sql`: lowest point of the roof surfaces at the roof-area median above the lowest ground point; a wall's area above it is its polygon projected onto its vertical plane and clipped. `heated` keeps the full hull |
| `lod2_window_ratio` | null (TEASER's TABULA ratio on the heated area) | – | a number sets windows = ratio × heated area (ro_heat: 0.2) |
| `sfh_roof_fix` | true | false | TEASER files the tilted SFH roof under orientations 0/90; legacy counted it as wall and added a roof of 1.2 × ground floor |
| `solar_beam` | `horizontal` (GHI − DHI) | `normal` (DNI) | the Perez/Duffie–Beckman code expects the beam irradiance on the horizontal |
| `window_shading_factor`, `window_frame_fraction`, `blind_threshold_w_per_m2` | 0.6, 0.3, null | 1.0, 0.0, 100 | TABULA shading instead of blinds closing above 100 W/m² all year |
| `ground_floor_b` | 0.5 | 1.0 | temperature reduction factor of the ground floor (DIN V 4108-6, TABULA) |
| `age_mapping` | `stock_share` | `band_end` | the TABULA class is drawn within the Zensus band by the IWU 2011 building counts (Loga et al. 2015, Tab. 4) per type; legacy took the last year of the band, i.e. the youngest class |
| `apartment_block_after_1978` | `multi_family_house` | `cap_1978` | TEASER has no AB after 1978; corrected uses the MFH archetype of the block's class, hot water stays AB |
| `air_change_per_h`, `ventilation_height_m` | 0.4, 2.5 | 0.5, null (TEASER volume = area × 3.125 m) | TABULA hygienic air change on the heated area × 2.5 m clear height |
| `heating_limit_c`, `fixed_heating_off_days` | 12, null | null, [135, 259] | TABULA heating limit on the daily mean outdoor temperature instead of a fixed heating break |
| `retrofit_share_before_1979`, `retrofit_share_1979_1994` | 0, 0 | 0, 0 | option S7: seeded share of level-0 buildings simulated with the TABULA retrofit variant |
| `consumption_factor` | `none` | `none` | option S8 `iwu`: IWU consumption calibration (Loga et al. 2015, App. C.3) on the annual energy of level-0 buildings, peak kept |

Source of the 0.55: InfDB holds 1,152 Mio m² of gross residential area for 6.39 M households in Bavaria
(`pylovo.buildings_result` v1). Bavaria had 650.5 Mio m² of living area in residential buildings and 665.6 Mio m²
in all buildings on 31.12.2022 (Bayerisches Landesamt für Statistik 2023, Statistischer Bericht F2400C 202200,
Tab. 3 and 4; 97.8 m² per dwelling in StMWi Bayern, Daten zur Raumbeobachtung, Tab. 5.5), i.e. 0.56–0.58 of the
InfDB gross area, and just over 4 % of the dwellings were vacant (Zensus 2022), which gives ≈ 0.55. The ratio is
a Bavarian average, not per type, on a Zensus 2011-based stock update; the two Bavarian publications differ by
≈ 2 %. It is below the usual living/gross ratio of 0.7–0.8 (DIN 277) because of the ridge-based floor count. An
InfDB version with eaves-based floors (`floor_source: eaves`) needs ≈ 0.73.

LoD2 surfaces reach the grid inputs as the optional h5 table `raw_data/building_surfaces` (`objectid`,
`wall_surface_area_m2`, `roof_surface_area_m2`, `shared_wall_area_m2`, `wall_above_eaves_m2`,
`shared_wall_above_eaves_m2`), written by
`scripts/lod2_building_surfaces.py` from a read-only InfDB query. Database-backed grids have no such table yet and
use the TABULA envelope.

#### TEASER refurbishment level

`asset_sizing.heat.teaser_retrofit_level` (default 0) selects the TABULA DE variant relative to each
building's own construction-year class: 0 "standard" (as built), 1 "retrofit" (usual full-envelope
refurbishment of wall, roof, floor and windows), 2 "advanced retrofit". Approximate single-family U-values of
TEASER's TABULA data (layer conduction plus 0.17 m²K/W surface resistance), W/m²K:

| Construction years | Wall standard | Wall retrofit | Roof standard | Roof retrofit |
|---|---:|---:|---:|---:|
| 1860–1918 | 1.71 | 0.24 | 1.38 | 0.36 |
| 1969–1978 | 1.01 | 0.21 | 0.49 | 0.21 |
| 1984–1994 | 0.48 | 0.17 | 0.36 | 0.36 |
| 2016+ | 0.15 | 0.15 | 0.15 | 0.11 |

The joint 2045 scenario uses level 1:

1. It matters for the old stock only: for recent classes standard and retrofit are nearly identical, so level 1
   means every building built before about 1995 has a complete envelope refurbishment by 2045.
2. It is an upper bound of refurbishment consistent with climate-neutral 2045 paths. IWU (2018) observed about
   1.4 %/a (single- and two-family) and 1.6 %/a (multi-family) refurbishment of pre-1979 houses in 2010–2016,
   about 1.1–1.2 %/a of the whole stock (windows about 2.5 %/a, roofs 2.3 %/a, facades 1.1 %/a, floors below
   1 %/a, as reported by Prognos, Öko-Institut, Wuppertal Institut 2021 and Prognos et al. 2022). Target paths
   raise the rate to about 1.75 %/a (2030–2045; Prognos, Öko-Institut, Wuppertal Institut 2021) and to
   1.6–1.7 % (2030) and 1.8–2.0 % (2045) in the BMWK KNG scenario (Prognos et al. 2022). In KNG about 25 % of
   the 2045 floor area is built from 2000 on, about 55 % refurbished since 2000 and about 20 % refurbished
   before 2000 or unrefurbished. Level 1 for all buildings therefore overstates the 2045 refurbished share by up
   to roughly a fifth of the floor area relative to that path, and by much more relative to current trends.
3. Its depth matches the target paths: on a median reference grid (TMY weather) level 1 gives a median of
   62 kWh/m² space heat and 36 W/m² peak (level 0: 149 kWh/m², 74 W/m²); Klimaneutrales
   Deutschland 2045 assumes about 60 kWh/m² after full refurbishment of a single- or two-family house (about
   KfW-Effizienzhaus 70) and 40–45 kWh/m² for multi-family houses.
4. Consequence: level 1 yields lower heat-pump and heating-rod peaks than a stock refurbished along today's
   trend, so grid stress from space heating is a lower-end estimate. A share-based variant (level 1 for about
   70–80 % of pre-2000 buildings, deterministically selected) is the natural sensitivity.

A comparison with INFDB `ro_heat` on the reference buildings gave TEASER level 0 about 1.9× INFDB in annual space
heat and 3× in peak.

Sources: Cischinsky, H.; Diefenbach, N. (2018): *Datenerhebung Wohngebäudebestand 2016.* IWU (as reported
below). Prognos, Öko-Institut, Wuppertal Institut (2021): *Klimaneutrales Deutschland 2045.* Long version for
Stiftung Klimaneutralität, Agora Energiewende and Agora Verkehrswende. Prognos AG (Thamling, N.; Rau, D.) with
FIW München, ITG Dresden, ifeu, Öko-Institut, adelphi, BBH, dena and EY Law (2022): *Hintergrundpapier zur
Gebäudestrategie Klimaneutralität 2045*, for BMWK, Table 7 and Section 4.

#### Domestic hot water

Residential hot water comes from OpenDHW for either space-heat source: stochastic tapping events are resampled
hourly and converted to thermal demand with seasonally varying cold- and mixed-water temperatures, and the
hourly result is kept unchanged as direct `water_heat` demand. Each building-flat call runs in a locally seeded
and restored random context, so the same buildings and `profile_seed` reproduce the realization independently
of call order and worker assignment.

No DHW tank is modelled. The heat pump and auxiliary heater must supply each hour's OpenDHW demand, and urbs
cannot shift DHW production. This keeps realistic timing but can overestimate generator peak power compared
with a thermostatically controlled tank. The urbs `heat_storage` is a space-heating buffer only; it stores
`space_heat` and cannot serve or shift `water_heat`.

### Climate inputs and full-load hours

$T_{\mathrm{NAT}}$ is the postcode-specific norm outside temperature `T_ne` in
`data/statistics/general/site_data.txt` (PLZ 99999: −12.6 °C). It is a design condition, not the minimum of
the weather year; only an exact postcode entry is accepted. The inherited table should eventually be replaced
or annotated with traceable DIN/TS 12831-1 data; until then −12.6 °C is an inherited input.

The regional full-load-hour proxy counts heating days with the DWD/VDI 3807 convention: days with a daily mean
outdoor temperature $\bar T_{\mathrm{out},d}$ below the heating limit $T_{\mathrm{HG}}=15$ °C
(`heating_limit_temperature_c`) count with the deficit to a base temperature $T_{\mathrm{b}}$
(`degree_day_base_temperature_c`, default the indoor temperature $T_{\mathrm{i}}=20$ °C,
`indoor_design_temperature_c`), as documented by the
[German Weather Service](https://opendata.dwd.de/climate_environment/CDC/derived_germany/techn/daily/heating_degreedays/hdd_3807/recent/).
Full-load hours are computed once per complete weather year, before timeframe selection and TSAM:

$$
\mathrm{GT}=\sum_{d:\,\bar T_{\mathrm{out},d}<T_{\mathrm{HG}}}\left(T_{\mathrm{b}}-\bar T_{\mathrm{out},d}\right),
\qquad
h_{\mathrm{FLH}}=\frac{24\,\mathrm{GT}}{T_{\mathrm{i}}-T_{\mathrm{NAT}}}.
$$

With $T_{\mathrm{b}}=T_{\mathrm{i}}$ (Gradtagzahl G20/15) the DSO A region's weather year gives about 2,619 h/a. This
counts the part of the indoor–outdoor difference that internal and solar gains cover, while the simulated annual
demand is net of these gains: the design-load proxy then reaches only about 0.7 of the simulated winter peak, and
the heat pump covers about 45 % of it. The joint scenario therefore uses the heating limit as base
($T_{\mathrm{b}}=15$ °C, Heizgradtage G15): 1,721 h/a for the DSO A region, inside the 1,500–1,800 h reported for
one- and two-family houses from new to unrenovated
([IER Stuttgart, 2016](https://www.ier.uni-stuttgart.de/forschung/modelle/heizkostenvergleich/pdf/16-05-09-IER_Waermekostenrechner_-_Dokumentation.pdf)),
and a design proxy of about 1.1 times the simulated peak. Converting annual simulated energy to a design-load
proxy this way is a study choice, not a DIN EN 12831 load calculation.

### Heat-pump and auxiliary sizing

Annual space-heating and hot-water energies of building $i$ become a design-load proxy

$$
P_{\mathrm{design},i}^{\mathrm{th}}
=\frac{E_{\mathrm{space},i}^{\mathrm{annual}}}{h_{\mathrm{FLH}}}
+\frac{E_{\mathrm{DHW},i}^{\mathrm{annual}}}{N_h},
$$

with $N_h$ the number of hours of the modelled year (mean DHW power is a study simplification; the hourly
OpenDHW series stays in dispatch and still sets the auxiliary peak). The bivalent air-source heat pump is

$$
P_{\mathrm{HP},i}^{\mathrm{th}}=s_{\mathrm{HP}}P_{\mathrm{design},i}^{\mathrm{th}},\quad s_{\mathrm{HP}}=0.65\ (\texttt{heat\_pump\_design\_share}),
\qquad
P_{\mathrm{HP},i}^{\mathrm{el}}=\frac{P_{\mathrm{HP},i}^{\mathrm{th}}}{\mathrm{COP}_i(T_{\mathrm{NAT}})},
$$

where the design COP is the building COP at the weather hour closest to $T_{\mathrm{NAT}}$ (radiator or
floor-heating sink temperature; air-source COP from `temp_air`). The design hour, the degree days, the COP and the
heat demand must come from one weather series: paired runs generate the heat library with the provider weather
file (`--weather-hdf`), which the PV library and the sizing also use. 0.65 is the central value of the 50–80 % range
for modulating monoenergetic air-to-water systems in the
[BWP dimensioning guide (2025)](https://www.waermepumpe.de/fileadmin/user_upload/waermepumpe/07_Publikationen/BWP_LF_WPDimensionierung.pdf).
That guide calls for a normative heat-load calculation under
[DIN EN 12831-1](https://www.dinmedia.de/de/norm/din-en-12831-1/261292587); the annual-energy/full-load-hour
proxy cannot reconstruct its transmission and ventilation terms and must not be called DIN EN 12831 compliant.
Applying 0.65 to `MFH` and `AB` central systems is an explicit extrapolation of a one- and two-family guide.

Direct electric auxiliary heat covers the exact positive full-year residual

$$
P_{\mathrm{aux},i}^{\mathrm{el}}=\max_t\left[\dot Q_{\mathrm{space},i,t}+\dot Q_{\mathrm{DHW},i,t}-\mathrm{COP}_{i,t}P_{\mathrm{HP},i}^{\mathrm{el}}\right]^+ .
$$

The heat pump is always capacity-limited and the auxiliary heater supplies every remaining peak. Heuristic cases
write equal installed and upper capacities and zero investment costs. `post-hems-optimized` keeps zero installed
capacities and active costs with building-specific bounds: the monovalent design capacity for the heat pump, the
observed heat peak for the auxiliary heater, and the physical buffer bound.

### Space-heating buffer

The buffer volume is tied to installed thermal heat-pump output,
$V_{\mathrm{buf},i}=v_{\mathrm{buf}}P_{\mathrm{HP},i}^{\mathrm{th}}$ (`buffer_volume_l_per_kw_th`), and its usable
energy follows from the water heat capacity and the usable spread $\Delta T_{\mathrm{buf}}$
(`buffer_usable_temperature_spread_k`):

$$
C_{\mathrm{buf},i}=\frac{V_{\mathrm{buf},i}\cdot 1.163\ \mathrm{Wh/(L\,K)}\cdot\Delta T_{\mathrm{buf}}}{1000}\ \mathrm{kWh_{th}},
\qquad
P_{\mathrm{buf},i}^{\mathrm{ch,max}}=P_{\mathrm{buf},i}^{\mathrm{dch,max}}=P_{\mathrm{HP},i}^{\mathrm{th}} .
$$

A buffer that holds $t_{\mathrm{buf}}$ hours of thermal heat-pump output has
$v_{\mathrm{buf}}=t_{\mathrm{buf}}\cdot 1000/(1.163\,\Delta T_{\mathrm{buf}})$.

- **Hydraulic buffer** (DSO A region scenarios): $v_{\mathrm{buf}}=20$ L/kW<sub>th</sub> and 5 K, i.e. about 7 minutes.
  This follows the [VDI 4645](https://www.dinmedia.de/de/technische-regel/vdi-4645/364873293) value for runtime
  optimisation, inside the 12–35 L/kW range of [DIN EN 15450](https://www.dinmedia.de/de/norm/din-en-15450/98862901)
  ([Weck-Ponten, 2023, Sec. 3.3.8](https://publications.rwth-aachen.de/record/969286/files/969286.pdf)). Such
  series buffers secure the minimum run time and defrosting and are not meant to bridge blocking times (Dimplex
  planning manual). They are no load-shifting store.
- **Flexibility buffer** (joint scenario): $t_{\mathrm{buf}}=1$ h (2 h as high case) at $\Delta T_{\mathrm{buf}}=10$ K,
  i.e. $v_{\mathrm{buf}}=86$ L/kW<sub>th</sub>. No DIN, EN or EU norm prescribes a time-based buffer rule. The
  closest is the bridging rule of VDI 4645 and the BWP hydraulics guide (2016): 30–40 L per kW and hour of
  blocking time, i.e. about one hour of rated output per blocking hour at their larger spread (as applied in the
  Viessmann Vitocal 150-A planning manual, 2026, p. 301). The 10 K spread is the SG Ready setpoint raise of the
  buffer ([BWP SG Ready interface 1.1](https://www.waermepumpe.de/fileadmin/user_upload/bwp_service/SG_ready/SG_Ready_Schnittstelle_1.1.pdf)).
  FfE for Agora Energiewende (2023) give every heat pump a 700 L (single-family) to 1,500 L (multi-family)
  combined tank at 55–65 °C because building mass is not modelled; the 1 h rule stores a similar energy
  (about 6 kWh for a 6 kW<sub>th</sub> heat pump against about 8 kWh). It is a scenario assumption, not a picture of
  today's stock: installers bridge blocking times with building mass and heat-pump oversizing, and under §14a EnWG
  blocking has become dimming.
- **Standing loss**: `technologies.storages.thermal_storage.self_discharge_per_timestep` = 0.005 per hour of the
  stored energy, about an ErP class-A tank of 700 L under EU 812/2013 at 30 K above ambient (real products
  0.8–0.9 %/h; FfE/Agora use 0.03 %/h).
- **Charge efficiency** (`buffer_charge_efficiency_method`): `technology` uses the storage's `charge_efficiency`;
  `cop_curve` gives each building the COP penalty of charging $\Delta T_{\mathrm{buf}}$ above the normal sink,
  $\eta_{\mathrm{ch},i}=\sum_t \dot Q_{\mathrm{space},i,t}\,\mathrm{COP}(\Delta\vartheta_{i,t}+\Delta T_{\mathrm{buf}})/\mathrm{COP}(\Delta\vartheta_{i,t})\big/\sum_t \dot Q_{\mathrm{space},i,t}$,
  with the lift $\Delta\vartheta_{i,t}$ recovered from the hourly COP on the model's COP curve (15–90 K). For
  +10 K this gives 0.86 (EN 14511 product data 0.80–0.89 per 10 K; when2heat regression 0.83).

In heuristic cases the reference is the fixed heat-pump capacity; in the optimized case a linear urbs constraint
ties the maximum buffer energy to the heat-pump capacity actually installed. With the flexibility buffer
(0.0116 kWh<sub>th</sub>/L at 10 K) a 5 kW<sub>th</sub> single-family heat pump gets about 430 L and a 20-flat block
(57 kW<sub>th</sub>) about 4,900 L, within the footprint of the oil tanks they replace (3,000–6,000 L single-family,
20,000–50,000 L apartment block).

`post-inflex-heuristic` and `post-hems-heuristic` consume the identical heat asset plan. INFLEX ignores buffer
flexibility, limits heat-pump heat to $\mathrm{COP}_{i,t}P_{\mathrm{HP},i}^{\mathrm{el}}$ and supplies the
residual with the fixed auxiliary heater; HEMS optimizes dispatch of the same capacities.

## Mobility

Vehicle ownership, model and schedule are sampled per household with the stable seeds above; emobpy parameters
(`mobility.*`) come from the scenario YAML. Home charging points are 11 kW
(`technologies.processes.home_charger`); a building can have several.

- Synthetic runs assign pregenerated profiles of the legacy pool `emobpy_pool_v1`
  (`data/statistics/general/mobility_profile_pool_old/`, deadline-clipped) and model charging in urbs with a
  mobility storage buffer.
- Paired runs use the conservation-preserving session pool `emobpy_pool_v2_sessions`
  (`data/statistics/general/mobility_profile_pool/`, generated with `generate_mobility_profile_pool --mode
  session` and frozen with `--freeze-manifest`) and write dedicated EV sessions (`urbs_in/ev_sessions`), which
  INFLEX needs. This is why synthetic runs cannot run INFLEX (open question for the user).

**HEMS session spreading.** Under flat prices every charging hour inside a session costs the same unless
local PV is available, so the HEMS optimum is degenerate: the LP returns full-power bursts that coincide
across buildings by solver choice, not by modelled behaviour. `mobility.hems_session_power_factor` (α ≥ 1)
caps each session in Step 3 at α × its average required power (session energy over connected hours) in
every hour without local PV surplus, i.e. where the site's PV potential (capacity bound × availability)
does not exceed its fixed electricity demand; PV-surplus hours keep the charger rating. The cap only
tightens the session's charging fraction (`hems_available_fraction`), adds no variables and keeps every
session feasible. InFlex charging is unchanged.

## Prices and temporal aggregation

Import price (0.398 EUR/kWh) and PV feed-in tariff (0.0 EUR/kWh in every repository scenario; a zero tariff keeps
physical export without remuneration) are scenario assumptions. TSAM (`time_aggregation`) selects representative
periods from ambient temperature and irradiation only (6 periods of 168 h, hierarchical clustering, medoid
representation, extreme periods for minimum mean temperature and maximum mean irradiation that replace cluster
centres); its settings are scenario assumptions, CPU and worker counts are run settings. TSAM is enabled in the
synthetic scenarios and disabled in the paired and aligned scenarios, which run the chronological full year:
Step 3 refuses TSAM together with EV sessions and asserts that full-year inputs carry no type-period weights.
With TSAM, Step 4 simulates the representative hours (6 × 168 = 1,008) instead of 8,760.

## Commercial and public (GHD) electricity

**Model.** The component manifest (`gridexpand.common.building_components`) gives a building at most one
residential and one non-residential component. The non-residential component (`Commercial` or `Public`, pylovo's
`nonresidential_use`) has the effective floor area of the InfDB split: LoD2 footprint × `floor_number` × the
non-residential share (`floor_number` is the LoD2 storey count, otherwise the height divided by the median storey
height of the building function). Step 2 draws one synPRO use type per component from the IWU shares of its
category and multiplies the type's hourly kWh/m² profile by the area (`allocation/functions/electricity.py`);
paired runs keep that annual energy and reshape it with the category-mean synPRO shape. There is no net-floor-area
factor and no BDEW standard load profile.

**Unknown is non-demand.** An InfDB building with function 31001_9998 ("not specifiable from the source") has the
non-residential use `Unknown`. It passes validation, but its non-residential part yields no component and no load;
a mixed building with an Unknown part keeps only its residential component. pylovo 8afedcc dimensions such parts
with the Public load values; GridExpand models no demand for them. The SQL view `surrogrid.grid_building_component`
applies the same rule (run `gridexpand db migrate --apply` once to update an existing database).

**Rules of the `ghd:` scenario block** (`gridexpand.common.ghd`, all off by default and in every repository
scenario):

| rule | effect |
|---|---|
| `single_volume_one_storey` | a fully non-residential single-volume building counts one storey (GHD area = footprint) |
| `osm_levels` | other fully non-residential buildings use OSM `building:levels` instead of `floor_number` |
| `activity_gating` | a GHD component needs a specific non-residential function or OSM activity evidence |

*Single volume.* A specific OSM `building` value of the footprint decides (church, chapel, cathedral, mosque,
synagogue, temple, shrine, religious, train_station, transportation, castle, parking, garage, garages, carport); a
missing or generic value (`yes`) falls back to the ALKIS function list: 3040–3043 and 3045–3047 (religious
buildings, without 3044 Gemeindehaus), 3031 Schloss, 3038 Burg, 3090, 3091, 3094, 3095, 3097 (station buildings,
without 3092 airport terminals), 2460–2465 (parking). The LoD2-height storeys otherwise count a 20 m nave as five or
six floors. Mixed buildings keep their split (the InfDB assigns the ground floor to the non-residential part).

*Activity gating.* A component is active if its ALKIS function names a non-residential use (every function except
31001_2000 generic commercial, the residential functions 31001_1xxx and 31001_9998), or if an OSM feature with an
activity tag lies on or near the footprint. Generic commercial buildings and the shop part of residential-function
buildings therefore need evidence; an Unknown building with evidence becomes a GHD component of the evidence's
category. Activity tags: every `shop`, `craft` and `office` value (government offices count as Public), every
`healthcare` value, `amenity` services (food and drink, banks, pharmacies, cinemas, fuel, schools, kindergartens,
libraries, town halls, police and fire stations, post offices, community and social facilities, places of worship,
hospitals, clinics, doctors, dentists, nursing homes), `tourism` accommodation, museums and galleries, and indoor
`leisure` (fitness and sports centres, sports halls, bowling, dance, sauna, arcades, ice rinks). Street furniture
and outdoor amenities (parking, benches, waste, vending, post boxes, toilets, shelters, charging points, ATMs) and
vacant or disused shops are not evidence. The rationale: a DSO supplies a building-bound business or public service
through its own connection, while a cadastral Commercial/Public polygon need not host one. In the reference area
only a minority of the buildings with modelled GHD had a GHD connection of the DSO; on those the floor-area model
matched the metered energy closely, while much of the metered GHD energy sat on connections with household meters
or a placeholder load.

*Evidence matching* (`match_osm_evidence`): a point (or a line's representative point) belongs to the footprint
that contains it, otherwise to the nearest footprint within `point_buffer_m` (default 5 m); a polygon belongs to
every footprint it covers by at least `polygon_min_overlap` (default 50 %) of the footprint area, otherwise to the
footprint of its representative point; an OSM building polygon gives its `building` value and `building:levels`
to the footprint it covers most (at least the same share). Sources: PostGIS tables next to the footprints with
configurable tables and columns (pgosm-flex layout by default), a CSV of per-building evidence, or a frame.
MV-direct components stay as pylovo classified them.

**Audits.** Aligned datasets write `paired_ghd_policy_audit.csv` (one row per building with a non-residential
part: decision, storey rule, source and GHD area, matched tags, OSM building value and levels) and a summary in
`paired_scenario_metadata.json` (`ghd_policy`); Step 2 records the summary in the run assumptions while a rule is
on.

## Paired validation

The real-vs-synthetic validation (pylovo alignment bundles, scenario units on two networks, status-quo test,
publication criteria) is not part of the scenario toolchain; its contract is in
[validation/METHOD.md](../validation/METHOD.md).

## Audit tables and acceptance checks

Step 2 writes building-level audit tables with every run: `raw_data/pv_asset_audit`, `battery_asset_audit`
(kWh per building, household, kWp and annual MWh, zero-capacity and upper-bound counts), `heat_asset_audit`
(annual heat, climate inputs, full-load hours, design COP, thermal and electric capacity, auxiliary bound, buffer
litres and kWh<sub>th</sub>, peak-coverage validity), `demand_component_audit` and the electrification assignment.
Database-mode hand-offs keep these compact tables although the bulky raw inputs stay in the database.

Before a result set is accepted (in particular a paired publication run), check, per physical building, scenario
unit, target bus and target:

1. heuristic PV equals $\min(E_{\mathrm{base}}\cdot 2.0/1000,\ P_{\mathrm{PV,max}})$; optimized PV lies in
   $[0, P_{\mathrm{PV,max}}]$;
2. heuristic batteries reproduce the HTW rule and the 2 h E/P ratio; optimized batteries stay within their bounds;
3. heuristic heat pump, auxiliary heater and buffer reproduce the rules above; optimized heat assets stay within
   their finite bounds; installed equals upper capacity for every heuristic row;
4. base electricity, space heat, DHW and mobility energy are conserved through scenario-unit and bus aggregation;
5. PV generation equals the normalized profile times the selected capacity;
6. shared buildings or buses introduce no duplicated physical asset;
7. real and synthetic contracts are identical before bus projection; heuristic HEMS and INFLEX asset plans are
   identical;
8. solvers terminated optimally (`urbs_out/solver_audit`), and power-flow convergence is accounted for.

Use an absolute tolerance of 1e-6 for deterministic capacity and energy transformations and report a separate
relative tolerance for solver results. These checks are not automated end to end: the fixed-grid smoke runners
that ran part of them were removed on 2026-09-25.

### Internal space heat and building preheating

`asset_sizing.heat.space_heat_source: internal` activates a chronological 1R1C
model in both urbs/Pyomo and PyPSA/Linopy. For each physical residential building:

```
C / dt * (T[t] - T[t-1]) = Q_room[t] + G_internal[t] + G_solar[t]
                         - (H_transmission + H_ventilation) * (T[t] - T_out[t])
```

The discretization is implicit Euler, following urbs-lvds. Units are kWh/K,
kW/K, kW, hours and °C. Delivered room heat is nonnegative and replaces fixed
space-heat demand in HEMS. Buildings sharing an electrical site retain separate
temperature states. HEMS optimizes the complete horizon with perfect foresight.
Internal+TSAM and discontinuous period selection are rejected.

The coefficient source is corrected `ro_heat.buildings_rc`: resistance K/W and
capacitance J/K. Its upstream 0.8 heated-area fraction is already embedded in R/C
and is not applied again. A mixed building's residential component share scales
conductance and capacitance once. The thermal floor area is 80% of residential
component area; ventilation uses this area and the configured zone height.
Household electricity, occupancy and direct DHW are not reduced by 20%.
The stored refurbishment years and RC changelog are exported; these coefficients
do not imply an additional 2045 refurbishment projection.

Internal gains reuse Richardson occupancy with the existing nighttime-presence
rule (70 W per present person), plus 0.362 of residential household electricity.
Residents are split into households with the household-size rule of the TEASER
path (destatis distribution, at most five per household). Seeds use physical IDs. Civil-time occupancy and DHW are mapped to fixed CET.
Solar irradiance uses the existing DistrictGenerator cardinal-plane calculation.
The database's total window area is divided equally between four cardinal
orientations because the available RC envelope lacks window orientations.
Window area follows the same residential/thermal area convention. The initial
solar approximation includes transmitted window gains with configurable glazing
transmittance (0.6), and blinds transmitting 0.15 above 100 W/m². It excludes
opaque-surface absorption and long-wave sky exchange; this is a documented
simplification relative to the TEASER 5R1C envelope gain calculation.

A forward, year-round thermostat supplies only enough heat to maintain the
minimum temperature (20°C by default), allowing passive overheating. Its annual
cycle is warmed up until the temperature boundary closes to 1e-9 K. HEMS starts
and ends at that recorded state. The upper bound is the larger of the baseline
passive temperature and minimum plus permitted preheating (0/1/2 K scenarios).
There is no cooling, comfort slack or artificial weekly temperature reset.

The independent reference sizes the fixed HP/rod/buffer with the configured
full-load-hour rule ([Climate inputs and full-load hours](#climate-inputs-and-full-load-hours),
including `degree_day_base_temperature_c`, e.g. Heizgradtage G15). Separate room,
DHW and tank charging COP curves consume one shared electrical HP capacity; the
rod likewise has one capacity. Only tank charging raises its sink curve, by the
usable buffer spread (`buffer_usable_temperature_spread_k`). That COP route
carries the charging penalty, so the internal tank's storage charge efficiency is
1 and `buffer_charge_efficiency_method: cop_curve` is rejected; the tank keeps the
configured discharge efficiency and standing loss. Tank charging must enter
storage; DHW remains direct. Fixed COP curves do not change with optimized room
uplift. The buffer and building mass are distinct stores. Service routing adds no
new investment or integer decisions.

InFlex uses the independent thermostat reference, fixed capacities and an
instantaneous HP dispatch to the highest-COP service first, with the rod supplying
remaining heat. It has no preheating or forecast. Its EV and greedy battery rules
are unchanged. The `heatpump_air` parent throughput exports total HP electricity
for Step-4 active/reactive demand, while service routes retain separate flows.

Physical references are cached by envelope provenance, area, occupancy,
residential electricity, weather, seed, thermal settings, generator source
fingerprints and dependency versions, independently of
target topology and permitted uplift. The 0/1/2 K cases reuse the same reference
and equipment. `raw_data/thermal_building_parameters`,
`raw_data/internal_heat_reference` and `urbs_in/building_thermal_*` record the
inputs. Results include `building_temperature`, `building_heat`,
`building_gains`, `building_loss` and `building_energy_residual`.

Internal preparation does not require TEASER heat/COP libraries. TEASER's
80%-area alignment and matched rerun remain deferred. Archived TEASER results
are historical context; grid expansion savings require a matched full-year
Step-4 comparison and cannot be inferred from building operating cost savings.
