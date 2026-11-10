#!/usr/bin/env python3
"""Store per-building LoD2 surface sums with a grid's h5 inputs (key ``raw_data/building_surfaces``).

The TEASER LoD2 envelope (``asset_sizing.heat.teaser.envelope_source: lod2``) reads these sums. Step 1:
run ``SURFACE_SQL`` read-only (psql) on the InfDB that holds the LoD2 surfaces, with ``:ids`` replaced by the
quoted objectids of the grid, and save the result as CSV. Step 2:

    uv run python scripts/lod2_building_surfaces.py GRID.h5 SURFACES.csv

The query follows InfDB ro_heat ``sql/01_get_building_surface_data.sql``: wall and roof areas are the sums
of the InfDB ``basedata.building_surface_area`` rows. ``shared_wall_area_m2`` is the WallSurface area whose
midline lies within 0.3 m of the footprint of a residential or mixed neighbour; LoD2 WallSurface includes
such party walls (Bavarian sample 2026-11-10: 21-27 % of the wall area of every building type).
"""

import sys

import pandas as pd

SURFACE_SQL = """
with ids as (select unnest(array[:ids]) oid),
sums as (
  select a.building_objectid b,
         sum(a.area) filter (where a.classname = 'WallSurface') wall_surface_area_m2,
         sum(a.area) filter (where a.classname = 'RoofSurface') roof_surface_area_m2
  from basedata.building_surface_area a join ids on ids.oid = a.building_objectid group by 1),
walls as (
  select s.surface_gmlid, s.building_objectid b, s.geometry g,
         ST_Centroid(ST_Boundary(ST_Force2D(s.geometry))) c
  from opendata.building_surface s join ids on ids.oid = s.building_objectid
  where s.surface_type = 'WallSurface'),
pts as (
  select w.surface_gmlid, d.path[1] p, d.path[2] r, d.path[3] v, ST_X(d.geom) x, ST_Y(d.geom) y, ST_Z(d.geom) z
  from walls w, ST_DumpPoints(w.g) d),
seg as (
  select surface_gmlid, p, r, x, y, z, lead(x) over w nx, lead(y) over w ny, lead(z) over w nz
  from pts window w as (partition by surface_gmlid, p, r order by v)),
ring as (  -- Newell vector area of each ring
  select surface_gmlid, p, r,
         0.5 * sqrt(power(sum(y * nz - z * ny), 2) + power(sum(z * nx - x * nz), 2)
                    + power(sum(x * ny - y * nx), 2)) a
  from seg where nx is not null group by 1, 2, 3),
wall_area as (select surface_gmlid, sum(case when r = 1 then a else -a end) a from ring group by 1),
heated_neighbour as (
  select distinct w.surface_gmlid
  from walls w join basedata.buildings n
    on n.objectid <> w.b and n.building_use in ('Residential', 'Mixed')
   and n.geom && ST_Expand(w.c, 0.3) and ST_DWithin(w.c, n.geom, 0.3)),
shared as (
  select w.b, sum(wa.a) wall_m2, coalesce(sum(wa.a) filter (where h.surface_gmlid is not null), 0) shared_m2
  from walls w join wall_area wa using (surface_gmlid) left join heated_neighbour h using (surface_gmlid)
  group by 1)
select s.b objectid, s.wall_surface_area_m2, s.roof_surface_area_m2,
       s.wall_surface_area_m2 * sh.shared_m2 / nullif(sh.wall_m2, 0) shared_wall_area_m2
from sums s left join shared sh on sh.b = s.b;
"""

COLUMNS = ["objectid", "wall_surface_area_m2", "roof_surface_area_m2", "shared_wall_area_m2"]


def attach_surfaces(h5_path, surfaces):
    """Write the surface rows of the grid's buildings to ``raw_data/building_surfaces``; return the row count."""
    buildings = pd.read_hdf(h5_path, key="raw_data/buildings")
    table = surfaces[COLUMNS].copy()
    table["objectid"] = table["objectid"].astype(str)
    table = table[table["objectid"].isin(buildings["objectid"].astype(str))]
    table = table.drop_duplicates("objectid").reset_index(drop=True)
    table.to_hdf(h5_path, key="raw_data/building_surfaces", mode="a", format="table")
    return len(table)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    n = attach_surfaces(sys.argv[1], pd.read_csv(sys.argv[2]))
    print(f"stored {n} building surface rows in {sys.argv[1]}")
