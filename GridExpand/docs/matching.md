# Matching the connection points of given grids to buildings

A grid you bring yourself usually has no consumers in it: it holds connection points (the buses its loads hang on,
often with an address), not buildings. `gridexpand match` decides for every connection point which InfDB building it
supplies, so that the scenario can model the demand of exactly those buildings. It is the first part of
`pipeline: real`; for now it runs on its own and writes its decisions for inspection.

Code: `src/gridexpand/grids/` (`matching.py` rules, `sources.py` inputs, `match.py` command).

## Inputs

- **Grids:** pandapower JSON files, one per LV grid; the file stem is the grid id. Every row of `net.load` is a
  connection point at the coordinates of its bus (`net.bus.geo`). Optional load columns: `address` (street and
  house number in one field), `on_building` (true: mounted on a wall, false: free-standing), `exclude` and
  `exclude_reason` (known non-building supplies such as street lights or telecom cabinets); the load `name` is
  kept as a remark. Line geometries (`net.line.geo`, else straight segments) are used for the list of buildings
  near the grid without a connection point.
- **Connection points as a CSV** (`--points`): `grid_id, bus, x, y` and the same optional columns plus `point_id`.
  Use it when the load buses of a grid are street nodes rather than the points where the service cables reach the
  buildings.
- **Coordinates:** `--crs` (default EPSG:25832, the CRS of InfDB); other CRS are transformed.
- **Buildings:** InfDB `basedata.buildings` around the points (footprint, address, use, households, residential
  floor area) and, for the pylovo version (`--pylovo-version-id`, default `PYLOVO_VERSION_ID` of the `.env`), the
  buildings pylovo connects straight to MV (`pylovo.building_exclusion`, reason `mv_direct`). Generate the region
  with pylovo first; postcodes without a pylovo result are named in the report, and their MV-direct buildings
  count as LV. All queries run in read-only transactions.

## Rules

The rules are tried in this order; a stronger rule wins when two points name the same building.

| rule | the point ... |
|---|---|
| `excluded_point` | is marked `exclude`: no building |
| `address_and_footprint` | has the exact address of exactly one demand building that covers it (a wall-mounted point may lie up to `facade_offset_m` outside) |
| `unique_address` | has the normalised address (`str.` written out, first house number with a letter directly after it) of exactly one building within `candidate_radius_m` |
| `covering_footprint` | lies in exactly one footprint, which is residential or carries the wall-mounted point |
| `building_without_demand` | names a building without a demand model (use not in `demand_uses`, or a use id in `non_demand_use_ids`, small utility buildings): no load |
| `mv_direct_building` | names a building that pylovo connects to MV: no LV load |
| `second_supply` | names a building that a stronger match already supplies from another bus: no load (several points on one bus may share a building, e.g. the meters of an apartment house) |
| `nearest_residential` | takes the nearest residential building within `nearest_residential_m` that no other bus supplies; closest points first; never a garage or shed |
| `no_residential_in_reach` | finds no such building: no load (`unmatched` when the fallback is switched off) |

A point names at most one building. Residential buildings near the grid without a point of their own (parts of a
building complex, a second house on a plot) carry no load; `unconnected_buildings.csv` lists them.

## Configuration

`--config <yaml>` reads the `matching:` block of a YAML file (later the run YAML of `pipeline: real`). All keys are
optional; the defaults are the rules and distances of the real-vs-synthetic validation, chosen on a non-public
real-grid dataset.

| key | default | meaning |
|---|---|---|
| `address_and_footprint`, `unique_address`, `covering_footprint` | `true` | switch an evidence rule off |
| `nearest_residential_m` | 20 | fallback distance in metres; `null` switches the fallback off |
| `candidate_radius_m` | 30 | buildings considered around a point (the fallback distance must not exceed it) |
| `facade_offset_m` | 2 | how far outside its building a wall-mounted point may lie |
| `exclude_mv_direct` | `true` | leave pylovo's MV-direct buildings without LV load |
| `demand_uses` | Residential, Commercial, Public, Mixed | InfDB uses with a demand model |
| `non_demand_use_ids` | `31001_2523` | InfDB use ids without a demand model (an ALKIS utility-building function) |
| `suspect_pattern` | none | regular expression over address and remark; a point it finds never gets the exact-address rule |
| `unconnected_radius_m` | 20 | distance to the grid lines for `unconnected_buildings.csv` |

## Outputs

`gridexpand match <grids...> --output <dir>` writes:

| file | content |
|---|---|
| `point_matches.csv` | per point: rule, `matched`/`excluded`, the supplied building and its distance; for exclusions the named building and, for a second supply, the bus that holds it |
| `building_supply.csv` | per supplied building: grid, bus, strongest rule, its points |
| `unconnected_buildings.csv` | demand buildings near the grid lines without a supply: `residential_without_a_point`, `outbuilding` or `mv_direct_building` |
| `grid_quality.csv` | per grid: points by evidence (`strong` address, `weak` geometry only, the exclusion reasons) and the weak and excluded shares |
| `matching_report.json` | configuration, inputs, counts per rule, pylovo coverage |

```bash
uv run gridexpand match ~/data/my_grids --output work/matching/my_grids --pylovo-version-id 3
uv run gridexpand match ~/data/my_grids --points points.csv --crs EPSG:31468 --config my_run.yaml --output out/
```
