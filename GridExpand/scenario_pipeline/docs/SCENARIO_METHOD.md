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
are exactly identical; only operation differs. Ordinary scenario runs use the
same deterministic realization contract: all requested model cases receive the
same run-YAML `profile_seed`, and stochastic choices are keyed by physical
building ID, component, and household or vehicle index. Paired validation may
project a compiled scenario onto real and synthetic buses, but it may not
redefine scenario assumptions.

PV, stationary-battery, and residential heat sizing are implemented in both
heuristic and optimized modes. The heat method compiles one central system per
physical residential building. Commercial heat-pump sizing is deliberately
outside the present method and is not assigned the residential rule.

## Configuration ownership

The scenario YAML is the single source of truth for mobility behavior and all
parameters written into urbs process and storage tables. This applies equally
to ordinary scenario runs and paired validation. The run YAML owns the selected
input topology version, selected model cases, profile-realization seed, and
execution resources. The seed selects a Monte Carlo realization; it does not
change the scientific distributions defined by the scenario. For paired
datasets, that declared version must match the immutable preparation metadata. Python
configuration retains only implementation references such as data locations
and API endpoints.


## Reproducible physical-profile realization

Controlled model-case comparisons separate stochastic input realization from
asset sizing and dispatch. The ordinary and paired pipelines derive deterministic
sub-seeds from the run-level seed and topology-independent physical building ID.
Independent sub-seeds cover household occupancy, annual household electricity,
non-residential use type, heat-system attributes, each OpenDHW flat, vehicle
ownership, vehicle model and schedule, and mobility-profile selection. Bus IDs
and DataFrame row order are deliberately excluded.

Every Step 2 HDF handoff records `profile_seed`, `profile_realization_id`, the
realization-contract version, and fingerprints for each component generated in
that run: base electricity and, where selected, space heat, hot water,
heat-pump COP, mobility demand, and mobility availability.
The realization ID depends on the seed and physical building inventory, but not
on model case. Multi-case smoke runs fail before interpretation when controlled
fingerprints differ. The two heuristic cases additionally require identical PV,
battery, and heat asset plans; optimized capacities may differ by design.

## Rooftop PV potential

All active pipelines use CityDB LoD2 roof sections joined through the pylovo
building object ID. The required CityDB properties are `Flaeche`,
`Dachneigung`, and `Dachorientierung`. Random roof-type, tilt, or azimuth
sampling is not an accepted production data source.

For roof section \(r\) of building \(i\), available peak capacity is

\[
P_{\mathrm{PV,max},i,r}=A_{i,r}\,u_r\,\rho_{\mathrm{PV}},
\qquad
P_{\mathrm{PV,max},i}=\sum_r P_{\mathrm{PV,max},i,r},
\]

where \(A_{i,r}\) is the LoD2 roof-surface area, \(u_r\) is the usable-area
fraction, and \(\rho_{\mathrm{PV}}\) is module peak power per roof area. The
scenario uses \(u_r=0.27\) for flat and \(u_r=0.58\) for sloped roofs, following
[Mainzer et al. (2014)](https://doi.org/10.1016/j.solener.2014.04.015), and
\(\rho_{\mathrm{PV}}=0.202\) kWp/m² from the module-performance statistics of
[Kräling et al. (2022)](https://doi.org/10.4229/WCPEC-82022-3BO.14.1). Because
\(A_{i,r}\) is already the inclined LoD2 surface, no footprint-to-roof or tilt
correction is applied.

A configurable 14.5 kWp legacy mean is a study-defined fallback only for
buildings without any usable LoD2 section; every use is reported and checked
against `maximum_fallback_share`. The publication scenario requires a zero
fallback share.

Orientation-dependent profiles are calculated with
[pvlib](https://doi.org/10.21105/joss.05994) and cached by binned tilt and
azimuth. The Forchheim defaults of 5° tilt and 15° azimuth were empirically
checked against exact LoD2 angles: the maximum observed annual-yield deviation
was 9.30% for the studied roofs.

## PV sizing

Heuristic PV capacity is calculated per physical building, before heat and
mobility electrification and before timeframe selection or TSAM:

\[
P_{\mathrm{PV},i}
=\min\left(
\alpha_{\mathrm{PV}}\frac{E_{\mathrm{el},i}}{1000},
P_{\mathrm{PV,max},i}
\right),
\qquad \alpha_{\mathrm{PV}}=2.0,
\]

where \(E_{\mathrm{el},i}\) is annual appliance-and-lighting electricity in
kWh/a. Public consumer recommendations span different sizing ambitions:
[Enpal](https://www.enpal.de/photovoltaik) starts from an approximate annual
yield of 1 MWh per kWp and recommends some oversizing,
[Vattenfall](https://www.vattenfall.de/infowelt-energie/solar/pv-anlage-dimensionierung)
recommends 1.5–2 kWp per annual MWh, and
[1KOMMA5°](https://1komma5.com/de/solaranlage/dimensionierung-pv-anlage/)
publishes a factor of 2.5. The scenario selects 2.0 kWp/MWh as its central
compromise: it is the upper end of Vattenfall's range and remains below the
more expansion-oriented 1KOMMA5° recommendation. These are public consumer
recommendations rather than normative design rules. Selecting the central
coefficient, applying it to all chosen building types, using one shared system
per building, and defining \(E_{\mathrm{el},i}\) as appliance-and-lighting
demand only are study choices. Heat-pump and mobility demand are excluded so
that the PV inventory remains independent of the later electrification
realization. The LoD2 potential supplies the hard physical upper bound.

Roof bins are filled in descending annual specific yield until the target is
met. One capacity-weighted normalized LoD2 profile represents the resulting
building-level system. In urbs, heuristic capacity is fixed with equal
`inst-cap` and `cap-up`; PV investment costs are zero because sizing occurred
upstream. Optimized sizing also uses one process per physical building, with
zero installed capacity and the physical maximum as `cap-up`. Consequently,
the fixed PV investment cost is charged once per building rather than once per
roof-angle bin. The optimized process scales the building's capacity-weighted
roof mix proportionally; it does not choose individual roof orientations.

`location_mode: predefined` limits PV to locations found in the source SWF
model when that inventory is part of the input. `all_buildings` selects one
primary electricity connection for every physical building.

## Stationary-battery sizing

All buildings with PV are battery candidates. When a source inventory such as
SWF contains battery rows, paired validation uses those rows only as location
evidence; their reported capacities do not determine the scenario capacity.
With annual base electricity \(E_{\mathrm{el},i}^{\mathrm{MWh}}\) in MWh/a and
PV capacity \(P_{\mathrm{PV},i}\) in kWp, the central heuristic is

\[
C_{\mathrm{bat},i}^{\mathrm{use}}=
\begin{cases}
0,
&P_{\mathrm{PV},i}\leq 0.5\,E_{\mathrm{el},i}^{\mathrm{MWh}},\\
\min\left(
\alpha_{\mathrm{bat,PV}}P_{\mathrm{PV},i},
\alpha_{\mathrm{bat,E}}E_{\mathrm{el},i}^{\mathrm{MWh}}
\right),
&P_{\mathrm{PV},i}>0.5\,E_{\mathrm{el},i}^{\mathrm{MWh}},
\end{cases}
\]

with \(\alpha_{\mathrm{bat,PV}}=1.0\) kWh/kWp and
\(\alpha_{\mathrm{bat,E}}=1.0\) kWh/MWh in the central case.

Figure 22 of the
[HTW Stromspeicher-Inspektion 2025](https://solar.htw-berlin.de/wp-content/uploads/HTW-Stromspeicher-Inspektion-2025.pdf)
supplies the eligibility threshold and recommends 1.5 kWh/kWp and 1.5 kWh/MWh
as upper limits for usable home-storage capacity. It does not recommend a
0.75–1.5 range. This study selects 1.0 as a less aggressive central
building-level heuristic because it extrapolates a home-storage recommendation
to all residential building types and uses appliance-and-lighting demand only.
Coefficients of 0.75 and 1.5 are study-defined low and high sensitivities; 1.5
is the HTW upper limit. The central coefficient is supported by open sizing
heuristics, not by a DIN or VDI standard. [HTW Berlin (2014)](https://solar.htw-berlin.de/publikationen/auslegung-pv-speicher-einfamilienhaus/)
identifies 1 kWh of usable capacity per kWp as sensible for high self-sufficiency,
while the [Bavarian LfU/C.A.R.M.E.N. guide (2022)](https://www.carmen-ev.de/wp-content/uploads/2022/02/Zukunftsloesungen-fuer-PV-Anlagen.pdf)
recommends approximately 0.7--1.0 kWh/kWp and at most 1 kWh per MWh of annual
household demand. Current [Vattenfall practitioner guidance](https://www.vattenfall.de/infowelt-energie/solar/lohnt-sich-pv-anlage)
also describes 1 kWh per kWp and per MWh as a frequently applied rule. However,
[HTW Berlin (2022)](https://solar.htw-berlin.de/publikationen/auslegung-von-solarstromspeichern/)
warns that a PV-only 1:1 rule can overdimension storage and recommends considering
both PV capacity and annual demand. The implemented minimum of both terms follows
that two-sided logic and is more restrictive than either isolated 1:1 rule; the
eligibility threshold can additionally set capacity to zero. For multi-household
buildings, the result represents one shared PV-battery system and is audited both
per building and per household.

A study-defined 2 h energy-to-power ratio sets symmetric charge and discharge
power:

\[
P_{\mathrm{bat},i}^{\mathrm{ch,max}}
=P_{\mathrm{bat},i}^{\mathrm{dch,max}}
=\frac{C_{\mathrm{bat},i}^{\mathrm{use}}}{2\ \mathrm{h}}.
\]

In both heuristic cases, \(P_{\mathrm{PV},i}\) is the fixed heuristic PV
capacity and usable battery energy is fixed with equal `inst-cap-c` and
`cap-up-c`. In `post-hems-optimized`, the building's LoD2 maximum PV potential
is used when constructing the battery upper bound because optimized PV capacity
is not known during input preparation. The HTW coefficients of 1.5 then define
only `cap-up-c`; `inst-cap-c` remains zero and urbs chooses installed capacity.
The heuristic capacity and optimized upper bound are intentionally not the same
quantity. Fixed heuristic assets carry no investment cost because sizing occurs
upstream.

Elias's thesis used 976 per kWh as a linear battery investment cost. The cited
[IRENA 2022 report](https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2022/Mar/IRENA_Tech_Innovation_Indicators_2022_.pdf)
identifies this value as the 2021 German median installed price in 2020 USD/kWh,
rather than a euro-denominated 2045 projection. The present 2045 optimized case
therefore uses 300 EUR/kWh as its central assumption. Scenario variants should
use 250 and 365 EUR/kWh as low and high sensitivities. These values span the
projected 2040 household stationary-system range in the
[JRC 2018 report](https://op.europa.eu/en/publication-detail/-/publication/e65c072a-f389-11e8-9982-01aa75ed71a1).
This remains an explicit extrapolation to 2045 and to shared multi-household
systems; the plausibility audit is required alongside optimized results.

## Residential heat assets

### Scope and demand representation

The first heat-asset method applies to `SFH`, `TH`, `MFH`, and `AB` buildings.
It represents one central heat system per physical building. Ordinary scenario
runs retain non-residential electricity but do not add commercial heat demand
or heat pumps. Paired validation likewise uses residential heat-pump inventory
rows only. A separate commercial sizing method is required before that scope is
extended.

The space-heat source is selected by `asset_sizing.heat.space_heat_source`.
For the current Forchheim 2045 scenario it is `infdb_ro_heat`: building-level
hourly heating-load series from the INFDB `ro_heat` schema, generated with its
1R1C model. Values stored as W are converted to hourly kWh. The source currently
contains 8,736 contiguous hours from 1 January through 30 December 2023. As an
explicit preliminary rule, the last available 24-hour day is duplicated to form
8,760 hours. The audit records every building affected by this calendar rule;
the library must be rebuilt when a complete export becomes available.

A missing individual `ro_heat` series must not abort regional preparation.
The adapter first selects the available building of the same type with nearest
total floor area and scales its heat demand by the target/source area ratio. If
that set is empty, it broadens to the nearest-area building in the full regional
pool. A missing target area uses a deterministic regional source without
scaling. Paired preparation uses progressively broader tiers: same grid and
type, same grid and use, same grid, same type, same use, then the full region.
Because the paired allocation currently carries LoD2 footprint area but not
floor count, this layer uses that footprint as its nearest-size and scaling
proxy. The chosen source building, match scope, and scale are stored in the
catalog and physical library. These are approved pragmatic fallbacks, not
claims that the borrowed profile is an exact physical-building simulation.

The alternative `teaser` source remains available for a later comparison. For
that route, pylovo `floor_area` is the LoD2 footprint and is multiplied by
`floor_number` before being passed as TEASER total net leased area. Neither heat
source applies an additional blanket heated-area factor such as 0.8. The TEASER
envelope design load is not used to size the HP; annual space-heat energy and
regional full-load hours provide the explicit sizing rule below.

Residential domestic-hot-water demand remains separate from either space-heat
source and originates from OpenDHW. OpenDHW generates stochastic tapping events
at its native resolution; the pipeline resamples these events hourly and
converts them to thermal demand using seasonally varying cold- and mixed-water
temperatures. The hourly result is retained unchanged as direct `water_heat` demand. Each
building-flat OpenDHW call is executed in a locally seeded and restored random
context. Rebuilding a run with the same physical buildings and `profile_seed`
therefore reproduces its DHW realization without coupling it to call order or
parallel worker assignment.

No DHW tank is modeled, either explicitly or implicitly. Consequently, the HP
and auxiliary heater must supply the hourly OpenDHW demand in its corresponding
time step, and urbs cannot shift DHW production. This preserves the realistic
timing produced by OpenDHW but can overestimate generator peak power compared
with a real thermostatically controlled DHW tank; a tank state-of-charge and
reheating controller would be required to resolve that effect physically.

The explicit urbs `heat_storage` is therefore a space-heating buffer only. It
stores the `space_heat` commodity and cannot serve or shift `water_heat`.

### Climate inputs and full-load hours

\(T_{\mathrm{NAT}}\) is the postcode-specific norm outside temperature `T_ne`
in `site_data.txt`; postcode 91301 currently resolves to \(-12.6\,^\circ\)C.
It is a design condition, not the minimum of the weather year. Only an exact
postcode entry is accepted. Numeric proximity between postcodes is not a
geographic fallback. The inherited table must eventually be replaced or
annotated with directly traceable DIN/TS 12831-1 climate data; until then,
\(-12.6\,^\circ\)C is explicitly an inherited input rather than a newly
derived study result.

The regional full-load-hour proxy uses the DWD/VDI 3807 degree-day convention
of \(T_{\mathrm{i}}=20\,^\circ\)C and a heating limit
\(T_{\mathrm{HG}}=15\,^\circ\)C. The
[German Weather Service](https://opendata.dwd.de/climate_environment/CDC/derived_germany/techn/daily/heating_degreedays/hdd_3807/recent/)
documents this 20/15 convention for degree-day calculations under VDI 3807.
Full-load hours are calculated once per complete regional weather year, before
timeframe selection and TSAM. With daily mean outdoor temperature
\(\bar T_{\mathrm{out},d}\),

\[
\mathrm{GTZ}
=\sum_{d:\,\bar T_{\mathrm{out},d}<T_{\mathrm{HG}}}
\left(T_{\mathrm{i}}-\bar T_{\mathrm{out},d}\right),
\]

\[
h_{\mathrm{FLH}}
=\frac{24\,\mathrm{GTZ}}
{T_{\mathrm{i}}-T_{\mathrm{NAT}}}.
\]

For the audited Forchheim pilot weather year this gives approximately
2,619 h/a. This is a study-defined conversion of annual simulated energy to a
design-load proxy; the 20/15 degree-day inputs are sourced, but this
full-load-hour sizing equation is not a DIN EN 12831 load calculation.

### Heat-pump and auxiliary sizing

For each residential building \(i\), annual space-heating and domestic-hot-water
energies are converted to a common design-load proxy:

\[
P_{\mathrm{space,design},i}^{\mathrm{th}}
=\frac{E_{\mathrm{space},i}^{\mathrm{annual}}}{h_{\mathrm{FLH}}},
\]

\[
P_{\mathrm{DHW,allow},i}^{\mathrm{th}}
=\frac{E_{\mathrm{DHW},i}^{\mathrm{annual}}}{N_h},
\]

\[
P_{\mathrm{design},i}^{\mathrm{th}}
=P_{\mathrm{space,design},i}^{\mathrm{th}}
+P_{\mathrm{DHW,allow},i}^{\mathrm{th}},
\]

where \(N_h\) is the number of hours in the modeled year. Using mean DHW power
for capacity sizing is a study-defined simplification; the actual hourly
OpenDHW series remains in dispatch and therefore still determines auxiliary
peak capacity.

The bivalent air-source heat pump is then sized as

\[
P_{\mathrm{HP},i}^{\mathrm{th}}
=s_{\mathrm{HP}}P_{\mathrm{design},i}^{\mathrm{th}},
\qquad s_{\mathrm{HP}}=0.65,
\]

\[
P_{\mathrm{HP},i}^{\mathrm{el}}
=\frac{P_{\mathrm{HP},i}^{\mathrm{th}}}
{\mathrm{COP}_i(T_{\mathrm{NAT}})}.
\]

The design COP is the building COP at the full-year weather step closest to
\(T_{\mathrm{NAT}}\), preserving the existing radiator/floor-heating
sink-temperature assumption. The 65% share is the selected central value from
the 50–80% range recommended for modulating monoenergetic air-to-water systems
in the
[BWP heat-pump dimensioning guide (2025)](https://www.waermepumpe.de/fileadmin/user_upload/waermepumpe/07_Publikationen/BWP_LF_WPDimensionierung.pdf).
That guide calls for a normative heat-load calculation under
[DIN EN 12831-1](https://www.dinmedia.de/de/norm/din-en-12831-1/261292587).
Our annual-energy/full-load-hour proxy is study-defined because the pipeline
does not possess all inputs needed for a complete room-by-room normative
calculation; it must not be described as DIN EN 12831 compliant. DIN EN
12831-1 derives room and building design loads from transmission and ventilation
heat losses under design boundary conditions, with applicable heat-up
allowances. Dividing annual simulated heat by regional full-load hours cannot
reconstruct those terms and therefore provides a transparent peak-load proxy,
not the normative result. The guide addresses modulating air-to-water systems
in one- and two-family houses; applying the 0.65 share to `MFH` and `AB`
central systems is an explicit study extrapolation.

Direct electric auxiliary heat covers the exact positive full-year residual:

\[
P_{\mathrm{aux},i}^{\mathrm{el}}
=\max_t
\left[
\dot Q_{\mathrm{space},i,t}
+\dot Q_{\mathrm{DHW},i,t}
-\mathrm{COP}_{i,t}P_{\mathrm{HP},i}^{\mathrm{el}}
\right]^+.
\]

Bivalent peak coverage follows the BWP design principle; using the maximum
modeled hourly residual as the installed auxiliary capacity is the study's
implementation rule. Thus HP output is always capacity-limited and the
auxiliary heater supplies every remaining peak in both heuristic cases; no code
path may let the HP operate beyond its installed electrical capacity.

Heuristic cases write equal installed and upper capacities and zero investment
costs. `post-hems-optimized` retains zero installed capacities and active costs,
but replaces generic 2,000 kW bounds with building-specific bounds: monovalent
calculated HP design capacity, the observed heat peak for auxiliary heating,
and the physical buffer bound.

### Space-heating buffer

The explicit buffer represents only the space-heating circuit. Its volume is
tied to installed thermal heat-pump output:

\[
V_{\mathrm{buf},i}
=v_{\mathrm{buf}}P_{\mathrm{HP},i}^{\mathrm{th}},
\qquad
v_{\mathrm{buf}}=20\ \mathrm{L/kW_{th}}.
\]

The 20 L/kWth ratio follows the
[VDI 4645](https://www.dinmedia.de/de/technische-regel/vdi-4645/364873293)
recommendation for runtime optimization. An independent review of buffer-sizing
approaches reports this value and places it within the broader 12–35 L/kW range
attributed to
[DIN EN 15450](https://www.dinmedia.de/de/norm/din-en-15450/98862901)
([Weck-Ponten, 2023, Sec. 3.3.8](https://publications.rwth-aachen.de/record/969286/files/969286.pdf)).
The standard-inspired ratio is applied here at the building-system level,
including multi-family buildings; that extrapolation and the assumption of one
central system per building are study choices.

Usable thermal energy follows from water heat capacity and a study-defined
usable temperature spread \(\Delta T_{\mathrm{buf}}=5\) K:

\[
C_{\mathrm{buf},i}
=\frac{
V_{\mathrm{buf},i}\,
1.163\ \mathrm{Wh/(L\,K)}\,
\Delta T_{\mathrm{buf}}
}{1000}.
\]

Charge and discharge power are set to thermal heat-pump capacity,

\[
P_{\mathrm{buf},i}^{\mathrm{ch,max}}
=P_{\mathrm{buf},i}^{\mathrm{dch,max}}
=P_{\mathrm{HP},i}^{\mathrm{th}},
\]

so the E/P ratio is derived rather than independently configured. In heuristic
cases the reference is the fixed heat-pump thermal capacity. In the optimized
case a linear urbs constraint ties maximum buffer energy to the heat-pump
capacity actually installed. The optimizer therefore cannot exploit the former
independent full-design-load buffer bound.

`post-inflex-heuristic` and `post-hems-heuristic` consume the identical fixed
heat asset plan. INFLEX dispatch ignores buffer flexibility, limits HP heat to
\(\mathrm{COP}_{i,t}P_{\mathrm{HP},i}^{\mathrm{el}}\), and supplies the
residual with the fixed auxiliary heater. `post-hems-heuristic` optimizes
dispatch of the same capacities. No DHW tank is represented, and the
space-heating buffer cannot serve DHW.

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
audit for outlier review rather than silently clipped. Deliberately changing the
run-level profile seed can shift these aggregate pilot values; the sizing and coverage
invariants are the acceptance criteria. The optimized-mode check wrote
zero installed capacity, positive costs, and finite building-specific bounds
for HP, auxiliary heater, and buffer.

## Capacity plausibility audit

The deterministic smoke runner writes one building-level capacity audit for
each post-electrification case. Battery results report kWh per building, per
household, per kWp and per annual MWh, plus zero-capacity and upper-bound
counts. Heat results report installed HP and auxiliary capacity, buffer kWhth
and m3, litres per installed kWth, and peak-coverage validity. A run fails if a
battery exceeds its configured or HTW maximum, if the heat generator bounds do
not cover the input peak, or if a buffer exceeds 20 l per installed kWth. These
checks assess scenario plausibility transparently; they do not impose an
undocumented product-size cap on shared systems.

## Prices and temporal aggregation

Import price and PV feed-in tariff are scenario assumptions. Forchheim 2045
currently uses a feed-in tariff of zero. TSAM period length, period count,
features, weights, clustering representation, and extreme-period policy are
also scenario assumptions. CPU and worker counts are run settings.
