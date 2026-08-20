# Scenario method

This document is the central record of scientific choices shared by the normal
scenario pipeline and paired validation. Short comments live beside values in
the scenario YAML; the reasoning and cross-stage contracts live here.

## Model cases

| Case                      | Asset sizing                | Operation                    |
| ------------------------- | --------------------------- | ---------------------------- |
| `pre`                   | Reference assets            | Reference electricity demand |
| `post-inflex-heuristic` | Shared heuristic asset plan | Rule-based dispatch          |
| `post-hems-optimized`   | Endogenous urbs sizing      | Optimized dispatch           |
| `post-hems-heuristic`   | Shared heuristic asset plan | Optimized dispatch           |

The controlled comparison of flexibility strategies is
`post-inflex-heuristic` versus `post-hems-heuristic`. In paired validation,
both dispatches are reconstructed from the same materialized heuristic asset
plan, so their PV, battery, heat-pump, auxiliary-heater, and buffer capacities
are exactly identical; only operation differs. Independently launched ordinary
scenario runs use the same sizing equations but are comparable only when they
also reuse the same input realization and seed. Paired validation may project a
compiled scenario onto real and synthetic buses, but it may not redefine
scenario assumptions.

PV, stationary-battery, and residential heat sizing are implemented in both
heuristic and optimized modes. The heat method compiles one central system per
physical residential building. Commercial heat-pump sizing is deliberately
outside the present method and is not assigned the residential rule.

## Configuration ownership

The scenario YAML is the single source of truth for mobility behavior and all
parameters written into urbs process and storage tables. This applies equally
to ordinary scenario runs and paired validation. Python configuration retains
only implementation references such as data locations and API endpoints.

## Rooftop PV potential

All active pipelines use CityDB LoD2 roof sections joined through the pylovo
building object ID. The required CityDB properties are `Flaeche`,
`Dachneigung`, and `Dachorientierung`. Random roof-type, tilt, or azimuth
sampling is not an accepted production data source.

Available section capacity is LoD2 surface area multiplied by the configured
usable roof fraction and module peak capacity per square metre. Flat and
slanted roofs use separate usable fractions. A configurable 14.5 kWp fallback
is allowed only for buildings without a usable LoD2 section; every use is
reported and checked against `maximum_fallback_share`.

pvlib profiles are cached by binned tilt and azimuth. The Forchheim defaults of
5 degrees tilt and 15 degrees azimuth were empirically checked against exact
LoD2 angles: the maximum observed annual-yield deviation was 9.30% for the
studied roofs.

## PV sizing

Heuristic PV capacity is calculated per physical building, before heat and
mobility electrification and before timeframe selection or TSAM:

```text
P_PV [kWp] = min(E_annual,electricity-only [kWh] * 2.5 / 1000,
                 P_PV,max [kWp])
```

Roof bins are filled in descending annual specific yield until the target is
met. A capacity-weighted normalized profile represents the resulting system.
In urbs, heuristic capacity is fixed with equal `inst-cap` and `cap-up`; PV
investment costs are zero because sizing occurred upstream. Optimized sizing
keeps separate LoD2 angle-bin processes with zero installed capacity and the
physical maximum as `cap-up`.

`location_mode: predefined` limits PV to locations found in the source SWF
model when that inventory is part of the input. `all_buildings` selects one
primary electricity connection for every physical building.

## Stationary-battery sizing

All buildings with PV are battery candidates. When a source inventory such as
SWF contains battery rows, paired validation uses those rows only as location
evidence; their reported capacities do not determine the scenario capacity.
With annual base electricity `E` in MWh/a and the relevant PV capacity `P_PV`
in kWp, the extrapolated HTW 2025 rule is:

```Python
C_battery,use [kWh] = 0,                              if P_PV <= 0.5 * E
C_battery,use [kWh] = min(1.5 * P_PV, 1.5 * E),       otherwise
```

The rule is applied to every building type as an explicit extrapolation from
HTW's single-family-home recommendation. In both heuristic cases, `P_PV` is
the fixed heuristic PV capacity; the resulting usable battery energy is fixed
with equal `inst-cap-c` and `cap-up-c`. In `post-hems-optimized`, `P_PV` is
instead the building's LoD2 maximum PV potential (`pv_max_kwp`), because the
PV capacity selected by urbs is not known during input preparation. The HTW
equation therefore provides a potential-based battery `cap-up-c`, while
`inst-cap-c` remains zero and urbs chooses the installed battery capacity
endogenously. It does not imply that heuristic and optimized cases have the
same numerical battery bound, nor does it couple optimized battery capacity to
the PV capacity ultimately selected by urbs. A configurable energy-to-power
ratio of 2 h sets maximum charge and discharge power to usable energy divided
by two. Fixed heuristic assets carry no investment cost because sizing occurs
upstream.

## Residential heat assets

### Scope and demand representation

The first heat-asset method applies to `SFH`, `TH`, `MFH`, and `AB` buildings.
It represents one central heat system per physical building. Ordinary scenario
runs retain non-residential electricity but do not add commercial heat demand
or heat pumps. Paired validation likewise uses residential heat-pump inventory
rows only. A separate commercial sizing method is required before that scope is
extended.

Space heat continues to use the DistrictGenerator/TEASER-derived building
profiles. No additional blanket heated-floor-area factor (such as 0.8) is
applied downstream: that would rescale an already generated thermal demand
without building-specific evidence. Floor area is retained only for the
specific-design-load audit. The TEASER envelope design load is not used to size
the HP; annual space-heat energy and regional full-load hours provide the
explicit sizing rule below. Residential domestic-hot-water demand continues
to originate from OpenDHW. OpenDHW generates stochastic tapping events,
which DistrictGenerator resamples to the hourly model resolution and converts
to thermal demand using the
seasonally varying cold- and mixed-water temperatures. This hourly profile is
retained unchanged as direct `water_heat` demand.

No DHW tank is modeled, either explicitly or implicitly. Consequently, the HP
and auxiliary heater must supply the hourly OpenDHW demand in its corresponding
time step, and urbs cannot shift DHW production. This preserves the realistic
timing produced by OpenDHW but can overestimate generator peak power compared
with a real thermostatically controlled DHW tank; a tank state-of-charge and
reheating controller would be required to resolve that effect physically.

The explicit urbs `heat_storage` is therefore a space-heating buffer only. It
stores the `space_heat` commodity and cannot serve or shift `water_heat`.

### Climate inputs and full-load hours

`T_NAT` is the postcode-specific norm outside temperature `T_ne` in
`site_data.txt`; postcode 91301 currently resolves to -12.6 degrees C. It is a
design condition, not the minimum of the TMY. Only an exact postcode entry is
accepted. Numeric proximity between postcodes is not a geographic fallback.
The inherited table must eventually be replaced or annotated with a directly
traceable DIN/TS 12831-1 source.

The heating-limit temperature is a separate building-operation assumption.
For the existing residential stock, the scenario uses the DWD/VDI 20/15
convention: 20 degrees C indoors and a 15 degrees C heating limit. Full-load
hours are calculated once per complete regional weather year, before timeframe
selection and TSAM. Daily mean ambient temperature selects heating days:

```text
GTZ = sum_d(20 - mean(T_out,d))  for mean(T_out,d) < T_HG
h_FLH = 24 * GTZ / (T_inside - T_NAT)
```

For the audited Forchheim pilot weather year this gives approximately 2,619 h/a.

### Heat-pump and auxiliary sizing

For each building:

```text
P_space,design,th = E_space,annual / h_FLH
P_DHW,allowance,th = E_DHW,annual / hours_in_year
P_design,th = P_space,design,th + P_DHW,allowance,th
P_HP,heuristic,th = heat_pump_design_share * P_design,th
P_HP,heuristic,el = P_HP,heuristic,th / COP(T_NAT)
```

The design COP is the building COP at the full-year weather step closest to
`T_NAT`, preserving the existing radiator/floor-heating sink-temperature
assumption. The Forchheim share is 0.65. This lies within the 50--80% range for an
air-source bivalent system in the [BWP heat-pump dimensioning guide](https://www.waermepumpe.de/fileadmin/user_upload/waermepumpe/07_Publikationen/BWP_LF_WPDimensionierung.pdf);
peak load is intentionally supplied by direct electric auxiliary heating.
This transparent scenario rule is informed by the practical guide, but it is
not asserted to be a complete DIN EN 15450 or VDI 4645 compliance calculation.

The fixed auxiliary capacity is the largest full-year positive residual:

```text
P_aux,el = max_t[Q_space(t) + Q_DHW(t) - COP(t) * P_HP,el]^+
```

Thus HP output is always capacity-limited. The auxiliary heater supplies every
remaining peak in both heuristic cases; no code path may let the HP operate
beyond its installed electrical capacity.

Heuristic cases write equal installed and upper capacities and zero investment
costs. `post-hems-optimized` retains zero installed capacities and active costs,
but replaces generic 2,000 kW bounds with building-specific bounds: monovalent
calculated HP design capacity, the observed heat peak for auxiliary heating,
and the physical buffer bound.

### Space-heating buffer

The building buffer volume is:

```text
V_buffer [l] = 20 l/kWth * P_HP,reference [kWth]
C_buffer [kWhth] = V_buffer * 1.163 Wh/(l K) * delta_T_usable / 1000
```

Forchheim uses a 5 K usable temperature spread. The 20 l/kWth value follows the
VDI 4645 headline rule discussed for the scenario; the temperature spread is an
explicit modeling assumption needed to convert volume into usable energy.
Charge and discharge power equal the reference thermal HP capacity, so the E/P
ratio is derived rather than independently configured.

`post-inflex-heuristic` and `post-hems-heuristic` consume the identical fixed
heat asset plan. INFLEX dispatch ignores buffer flexibility, limits HP heat to
`COP(t) * P_HP,el`, and supplies the residual with the fixed auxiliary heater.
`post-hems-heuristic` optimizes dispatch of the same capacities.

The production audit tables record annual heat energies, climate inputs,
full-load hours, design COP, thermal and electrical capacity, auxiliary bound,
buffer litres and kWhth, and the representation choices. Database-backed Step
2 handoffs retain these compact tables even though bulky raw inputs remain in
the database.

One implemented full-year realization on Forchheim grid
`9474126-07_91301_1_12` (`post-hems-heuristic`, residential scope) produced 20
central systems and 477.874 MWhth/a of useful heat. Its checks found 2,619.23
full-load hours, 39.221 kWel of fixed HP capacity, 258.182 kWel of fixed
auxiliary capacity, 1,906.4 litres (11.086 kWhth) of explicit space-heating
buffer, an exact 20 l/kWth ratio, retained within-day OpenDHW variation, and
zero uncovered heat in INFLEX dispatch. The coincident auxiliary peak was
216.587 kW and the auxiliary heater supplied 10.54% of annual useful heat. The
high auxiliary peak is consistent with the documented absence of DHW-tank
buffering. Building design-load intensities
ranged from 13.7 to 109.6 W/m2 (median 53.1 W/m2), which is retained in the
audit for outlier review rather than silently clipped. Stochastic input profile
realizations can shift these aggregate pilot values; the sizing and coverage
invariants are the acceptance criteria. The optimized-mode check wrote
zero installed capacity, positive costs, and finite building-specific bounds
for HP, auxiliary heater, and buffer.

## Prices and temporal aggregation

Import price and PV feed-in tariff are scenario assumptions. Forchheim 2045
currently uses a feed-in tariff of zero. TSAM period length, period count,
features, weights, clustering representation, and extreme-period policy are
also scenario assumptions. CPU and worker counts are run settings.
