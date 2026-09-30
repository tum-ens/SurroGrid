# Scientific method

This is the record of the scientific choices shared by the synthetic pipeline and the paired
validation. Adjustable values live in the scenario YAMLs (`config/scenarios/`, key reference in
[configuration.md](configuration.md)); the reasoning and the cross-stage contracts live here. Values
quoted below are those of every repository scenario YAML unless a table says otherwise.

| Scenario YAML | used by | heat source | adoption (heat / mobility / PV+battery) | TSAM |
|---|---|---|---|---|
| `forchheim_2045_synthetic.yaml` | synthetic runs | `infdb_ro_heat` | deterministic 1.0 / 1.0 / 1.0 | on |
| `schweinfurt_2045.yaml` | synthetic runs | `teaser` (level 0) | deterministic 0.75 / 0.75 / 0.75 | on |
| `forchheim_2045_full_year.yaml` | paired SWF runs | `infdb_ro_heat` | source inventory (SWF) | off |
| `joint_2045_full_year.yaml` | aligned SWF + ÜZW runs | `teaser` (level 1) | deterministic 0.8 / 1.0 / 0.8 | off |
| `00_scenario_template.yaml` | template | `infdb_ro_heat` | deterministic 1.0 / 1.0 / 1.0 | on |

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

`electrification.<heat|mobility|pv_battery>.adoption_mode` selects, per technology:

- `deterministic_share`: eligible physical buildings are ranked by a stable seed and the exact rounded
  `building_share` is selected;
- `source_inventory`: only buildings backed by explicit source evidence (the SWF 2045 inventory) are
  selected; `building_share` is not allowed.

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
bins against exact LoD2 angles was 9.30 % for the Forchheim roofs. Capacity always uses the exact
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

Only buildings selected for `pv_battery` are battery candidates. Battery rows of a source inventory (SWF)
are location evidence only; their capacities do not set the scenario capacity. With annual base
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

- `infdb_ro_heat` (Forchheim scenarios): building-level hourly heating loads from the INFDB `ro_heat` schema
  (1R1C model), converted from W to hourly kWh. The source holds 8,736 contiguous hours (1 January to
  30 December 2023); as a preliminary rule the last available day is duplicated to form 8,760 hours, and every
  affected building is audited. The library must be rebuilt when a complete export is available. A missing
  single series does not abort preparation: the adapter takes the available building of the same type with the
  nearest total floor area and scales by the area ratio, then broadens to the nearest-area building of the whole
  region (paired preparation uses the tiers same grid and type, same grid and use, same grid, same type, same
  use, region; it uses the LoD2 footprint as size proxy). Source building, match scope and scale are stored.
  These are approved pragmatic fallbacks, not exact simulations.
- `teaser` (Schweinfurt and joint SWF + ÜZW scenarios): TEASER `tabula_de` archetypes with the component's
  residential effective floor area (footprint × `floor_number`, residential share) as total net leased area;
  the number of floors passed to TEASER does not change the result, and no blanket heated-area factor is
  applied. The TEASER envelope design load is not used for sizing (see below). Occupancy (richardsonpy) and hot
  water draw on seeded random states per building. Standalone synthetic runs simulate with a PVGIS TMY at the
  grid's transformer. Paired runs pass the provider weather file (`--weather-hdf`), so heat, COP, PV and sizing
  share one TMY; a TMY takes each month from a different year, so TMYs of two nearby locations are unrelated
  hour by hour. The 5R1C model heats to a constant 20 °C and switches heating off on days 135–258; all
  buildings restart together on 16 September, which can set the auxiliary-heater size.

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
3. Its depth matches the target paths: on a median SWF grid (67 residential buildings, Forchheim TMY) level 1
   gives a median of 62 kWh/m² space heat and 36 W/m² peak (level 0: 149 kWh/m², 74 W/m²); Klimaneutrales
   Deutschland 2045 assumes about 60 kWh/m² after full refurbishment of a single- or two-family house (about
   KfW-Effizienzhaus 70) and 40–45 kWh/m² for multi-family houses.
4. Consequence: level 1 yields lower heat-pump and heating-rod peaks than a stock refurbished along today's
   trend, so grid stress from space heating is a lower-end estimate. A share-based variant (level 1 for about
   70–80 % of pre-2000 buildings, deterministically selected) is the natural sensitivity.

A comparison with INFDB `ro_heat` on 1,203 SWF buildings (TEASER level 0 about 1.9× INFDB in annual space heat
and 3× in peak) is kept as a research note:
[2026-09-24_teaser_vs_infdb_ro_heat.md](research/2026-09-24_teaser_vs_infdb_ro_heat.md).

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
`data/statistics/general/site_data.txt` (PLZ 91301: −12.6 °C). It is a design condition, not the minimum of
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

With $T_{\mathrm{b}}=T_{\mathrm{i}}$ (Gradtagzahl G20/15) the Forchheim weather year gives about 2,619 h/a. This
counts the part of the indoor–outdoor difference that internal and solar gains cover, while the simulated annual
demand is net of these gains: the design-load proxy then reaches only about 0.7 of the simulated winter peak, and
the heat pump covers about 45 % of it. The joint scenario therefore uses the heating limit as base
($T_{\mathrm{b}}=15$ °C, Heizgradtage G15): 1,721 h/a for Forchheim, inside the 1,500–1,800 h reported for
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

- **Hydraulic buffer** (Forchheim scenarios): $v_{\mathrm{buf}}=20$ L/kW<sub>th</sub> and 5 K, i.e. about 7 minutes.
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

## Prices and temporal aggregation

Import price (0.398 EUR/kWh) and PV feed-in tariff (0.0 EUR/kWh in every repository scenario; a zero tariff keeps
physical export without remuneration) are scenario assumptions. TSAM (`time_aggregation`) selects representative
periods from ambient temperature and irradiation only (6 periods of 168 h, hierarchical clustering, medoid
representation, extreme periods for minimum mean temperature and maximum mean irradiation that replace cluster
centres); its settings are scenario assumptions, CPU and worker counts are run settings. TSAM is enabled in the
synthetic scenarios and disabled in the paired and aligned scenarios, which run the chronological full year:
Step 3 refuses TSAM together with EV sessions and asserts that full-year inputs carry no type-period weights.
With TSAM, Step 4 simulates the representative hours (6 × 168 = 1,008) instead of 8,760.

## Paired validation contract

The paired pipelines compare real DSO grids (SWF; SWF and ÜZW in aligned runs) with synthetic pylovo grids under
the same post-electrification scenario. They hold the physical demand and technology realization constant and
change only the electrical network and the mapping of scenario units to buses. Paired validation may project a
compiled scenario onto real and synthetic buses; it may not redefine scenario assumptions.

| Layer | Shared between real and synthetic | Network-specific |
|---|---|---|
| Scope | physical buildings retained by both network models, same minimum-building rule | grid partition |
| Base demand | household rows, measured annual household energy, calibrated GHD energy, sampled profile realization | allocation bus |
| Sector assets | deduplicated source inventory (SWF 2045 PV, EV, heat pump) or deterministic selection | allocation bus |
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

**One HEMS per building.** Where several source connections of one LV grid match the same LoD2 building, their
annual demand and asset evidence are summed and assigned to the connection with the largest annual base
electricity (ties: lowest bus id). A building matched to connections in different LV grids is excluded (no single
connection preserves both transformer assignments). Both counts are recorded in the scope audit and the dataset
metadata.

**Reference-grid scope and topology (SWF).** The Forchheim reference population uses the curated `swf_clear`
dataset: 49 station workbooks with exactly one transformer each, so no multi-transformer station splitting is
needed. The source grids are meshed (combined cycle rank 112); their validated radialized derivatives (cycle rank 0,
no unsupplied load buses) are used for power flow and reinforcement. `station_split_manifest.csv` and
`station_radialization_manifest.csv` below `GRID_DATA_PATH` record the eligible stations and the derived workbooks;
allocation and real-grid power flow select the same entries. They describe provenance, not scenario parameters.

**Source inventory (SWF).** Heat-pump rows are location evidence only (no positive capacities); the three component
records of one installation (`load_type` `hp`, `heat`, `dhw`, same `Baujahr`) form one physical system. PV rows with
`Baujahr <= 2045` (and legacy rows without a usable year) mark a building as a PV location; the SWF PV capacity is
not an installation limit. Charging-point capacities are fixed.

### GHD and mixed-use evidence rules

pylovo's open building layer contains many more Commercial/Public polygons than a DSO has GHD customers; an ALKIS
polygon is not necessarily an active electricity customer, and one connection can serve a mixed-use building. The
paired scenario therefore:

1. retains DSO GHD demand only where the GHD row matches a physical building;
2. keeps DSO household rows that match a Commercial/Public polygon as mixed-use household proxies;
3. adds pylovo per-square-metre GHD defaults to neither network;
4. excludes an unmatched GHD row individually and audits it, without rejecting its otherwise valid LV grid.

The Forchheim evidence table behind these rules is in
[2026-07-21_forchheim_ghd_calibration_v5.md](research/2026-07-21_forchheim_ghd_calibration_v5.md). pylovo grids are
still dimensioned with the open-data GHD assumptions; an open-data rule for load-active generic commercial buildings
is a separate topology sensitivity and must not use DSO matching (that would leak the reference network into the
synthetic generation).

### Shared profile libraries

Physical heat and COP series are generated once per profile assumption set and stored in a network-independent
regional HDF5 library keyed by `building_objectid`; readiness checks coverage against it and the urbs inputs of
both targets read the same profiles before projecting them to buses. Topology changes can reuse a library; changes
of refurbishment, weather or other heat assumptions need a new profile-set id (aligned runs derive it as
`<paired_dataset_id>_teaser_heat_<scenario hash 12>`). Diagnostic heat fallbacks
(`--allow-diagnostic-heat-fallback`) are not publication results. The PV profile library is built the same way
(see "Rooftop PV potential").

### Publication criteria

A publication run requires:

1. equal paired-plan buildings and demand totals;
2. no diagnostic heat-profile fallbacks;
3. identical scenario-unit inputs and optimization settings for both targets;
4. an identical temporal horizon and, with TSAM, identical representative periods for both targets (the runner
   records one canonical mapping and rejects results that do not reproduce it);
5. no silently skipped power-flow timesteps: non-convergence is reported (real grids with failed timesteps are
   `incomplete` in the expansion analysis, not zero-cost);
6. a separate meshed-versus-radial sensitivity for critical real grids where radialization materially changes
   stress.

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
