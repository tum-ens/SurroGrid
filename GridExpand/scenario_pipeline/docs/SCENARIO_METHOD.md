# Scenario method

This document is the central record of scientific choices shared by the normal
scenario pipeline and paired validation. Short comments live beside values in
the scenario YAML; the reasoning and cross-stage contracts live here.

## Model cases

| Case | Asset sizing | Operation |
| --- | --- | --- |
| `pre` | Reference assets | Reference electricity demand |
| `post-inflex-heuristic` | Shared heuristic asset plan | Rule-based dispatch |
| `post-hems-optimized` | Endogenous urbs sizing | Optimized dispatch |
| `post-hems-heuristic` | Shared heuristic asset plan | Optimized dispatch |

The comparison of flexibility strategies is
`post-inflex-heuristic` versus `post-hems-heuristic`: their asset capacities
must be identical. Paired validation may project a compiled scenario onto real
and synthetic buses, but it may not redefine scenario assumptions.

Heat-pump sizing currently exposes an explicit `temporary_placeholder`
adapter. It documents the incomplete method without inventing fixed physical
capacity. PV and stationary-battery sizing are implemented in both heuristic
and optimized modes.

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

```text
C_battery,use [kWh] = 0,                              if P_PV <= 0.5 * E
C_battery,use [kWh] = min(1.5 * P_PV, 1.5 * E),       otherwise
```

The rule is applied to every building type as an explicit extrapolation from
HTW's single-family-home recommendation. Heuristic cases fix usable energy
with equal `inst-cap-c` and `cap-up-c`. `post-hems-optimized` retains zero
installed capacity and uses the same result only as its endogenous sizing
upper bound. A configurable energy-to-power ratio of 2 h sets maximum charge
and discharge power to usable energy divided by two. Fixed heuristic assets
carry no investment cost because sizing occurs upstream.

## Prices and temporal aggregation

Import price and PV feed-in tariff are scenario assumptions. Forchheim 2045
currently uses a feed-in tariff of zero. TSAM period length, period count,
features, weights, clustering representation, and extreme-period policy are
also scenario assumptions. CPU and worker counts are run settings.
