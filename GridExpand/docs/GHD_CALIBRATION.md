# GHD Calibration for the Paired SWF Comparison

## Decision

The paired comparison preserves measured SWF household and GHD demand and projects the same physical demand units onto the real and synthetic networks. A pylovo Commercial/Public polygon is not treated as an independent GHD consumer unless an SWF GHD row provides direct load evidence. An SWF HH row matched to a Commercial/Public polygon is retained as a mixed-use household proxy.

Unmatched SWF GHD rows are excluded individually and reported. They no longer exclude their complete LV grid, because doing so also removes valid household and matched GHD demand from both comparison networks.

## Forchheim Audit

The full building calibration for PLZ 91301 and pylovo version 5 contains the following evidence classes:

| Evidence class | Buildings | Floor area | Synthetic default GHD | SWF/calibrated GHD |
|---|---:|---:|---:|---:|
| Direct SWF GHD | 273 | 103,249 m² | 4.657 GWh/a | 10.484 GWh/a |
| Direct SWF HH mixed-use proxy without independent GHD evidence | 623 | 182,246 m² | 10.752 GWh/a | 0 GWh/a |
| Generic commercial structure without SWF load evidence | 2,991 | 250,429 m² | 22.396 GWh/a | 0 GWh/a |
| Other building without SWF load evidence | 201 | 77,761 m² | 3.873 GWh/a | 0 GWh/a |
| Unmatched SWF GHD row | 0 | 0 m² | 0 GWh/a | 0.200 GWh/a excluded |

The 2,991 generic structures use ALKIS code `31001_2000` (building for economy or commerce). Of the 2,961 omitted Commercial buildings in the selected synthetic grids, 2,899 have no address and their median floor area is approximately 40 m². The code identifies a cadastral building function, not an active independent electricity customer. The German Heat Atlas similarly treats this class as uncertain with respect to whether it is an energetically relevant building and excludes it from heat-demand calculation.

The current SWF matcher already allocates household demand to synthetic non-residential polygons where the SWF household connection lands there. Across the full source audit, 719 Commercial/Public buildings receive 1,647 HH rows. In the revised paired scope, 693 such buildings receive 1,565 HH rows and 5.686 GWh/a. This is the supported mixed-use contribution. Assigning HH profiles to thousands of additional nearby GHD polygons would duplicate demand or remove profiles from explicitly residential buildings and is therefore not used.

## Revised Paired Scope

| Quantity | Real SWF | Synthetic |
|---|---:|---:|
| Target grids | 88 | 91 |
| Physical buildings | 7,643 | 7,643 |
| Scenario units | 7,994 | 7,994 |
| HH rows | 14,398 | 14,398 |
| HH annual demand | 40.847 GWh | 40.847 GWh |
| Calibrated GHD annual demand | 9.147 GWh | 9.147 GWh |

Pylovo version 5 covers every physical building retained by the paired scope; the unmapped-building audit is empty.

## Interpretation

The apparent difference of roughly 3,000 GHD buildings is not evidence for 3,000 missing SWF customers. Most are generic, unaddressed cadastral structures for which neither SWF GHD nor SWF HH demand provides direct load evidence. The fair paired power-flow comparison therefore uses equal measured/calibrated demand totals rather than pylovo's raw per-square-metre default for every polygon.

This does not erase the topology-generation limitation: pylovo version 5 was generated with `residential_only_generation = false`, so transformer and cable dimensioning still reflects the original open-data GHD assumptions. A later topology sensitivity should test an open-data-only rule for identifying load-active generic commercial buildings. That rule must not use SWF matching, because doing so would leak the reference network into synthetic grid generation.

## Sources

- [Bavarian Surveying Administration: ALKIS LoD2 building-function codes](https://www.ldbv.bayern.de/mam/ldbv/dateien/alkis-shape_datenformatbeschreibung.pdf)
- [ifeu: Wärmeatlas Deutschland 3.0 model description](https://www.ifeu.de/fileadmin/uploads/Publikationen/Energie/ifeu_gef_geomer_Modellbeschreibung_Waermeatlas_Deutschland_3.0_01_2024.pdf)
