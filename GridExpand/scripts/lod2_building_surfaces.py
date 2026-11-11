#!/usr/bin/env python3
"""Store per-building LoD2 surface sums with a grid's h5 inputs (key ``raw_data/building_surfaces``).

The TEASER LoD2 envelope (``asset_sizing.heat.teaser.envelope_source: lod2``) reads these sums. Step 1:
run ``SURFACE_SQL`` read-only (psql) on the InfDB that holds the LoD2 surfaces, with ``:ids`` replaced by the
quoted objectids of the grid, and save the result as CSV. Step 2:

    uv run python scripts/lod2_building_surfaces.py GRID.h5 SURFACES.csv

The query follows InfDB ro_heat ``sql/01_get_building_surface_data.sql``: wall and roof areas are the sums
of the InfDB ``basedata.building_surface_area`` rows. ``shared_wall_area_m2`` is the WallSurface area whose
midline lies within 0.3 m of the footprint of a residential or mixed neighbour; LoD2 WallSurface includes
such party walls (Bavarian sample 2026-11-10: 11-31 % of the wall area per building type).
The eaves follow InfDB 05_fill_floor_number.sql (lowest point of the roof surfaces at the roof-area
median, above the lowest ground point); a wall's area above the eaves is its polygon projected onto its
vertical plane and clipped there. The TEASER unheated-attic rule removes these gable areas.
"""

import sys

import pandas as pd

SURFACE_SQL = """
with ids as (select unnest(array[:ids]) oid),
sums as (
  select a.building_objectid b,
         sum(a.area) filter (where a.classname='WallSurface') wall_surface_area_m2,
         sum(a.area) filter (where a.classname='RoofSurface') roof_surface_area_m2
  from basedata.building_surface_area a join ids on ids.oid=a.building_objectid group by 1),
surf as (select s.* from opendata.building_surface s join ids on ids.oid=s.building_objectid),
lv_roofs as (
  select building_objectid b, ST_ZMin(geometry) z_min,
         sum(ST_Area(geometry)) over (partition by building_objectid order by ST_ZMin(geometry), ST_Area(geometry) rows unbounded preceding) up_to,
         sum(ST_Area(geometry)) over (partition by building_objectid) area_all
  from surf where surface_type='RoofSurface'),
levels as (
  select g.b, g.z_ground, r.z_eaves, g.z_top
  from (select building_objectid b, min(ST_ZMin(geometry)) z_ground, max(ST_ZMax(geometry)) z_top from surf group by 1) g
  left join (select b, min(z_min) filter (where up_to >= 0.5*area_all) z_eaves from lv_roofs group by 1) r using (b)),
walls as (
  select s.surface_gmlid, s.building_objectid b, s.geometry g, ST_Centroid(ST_Boundary(ST_Force2D(s.geometry))) c
  from surf s where s.surface_type='WallSurface'),
pts as (
  select w.surface_gmlid, d.path[1] p, d.path[2] r, d.path[3] v, ST_X(d.geom) x, ST_Y(d.geom) y, ST_Z(d.geom) z
  from walls w, ST_DumpPoints(w.g) d),
seg as (
  select surface_gmlid, p, r, v, x, y, z, lead(x) over w nx, lead(y) over w ny, lead(z) over w nz
  from pts window w as (partition by surface_gmlid, p, r order by v)),
part as (  -- horizontal Newell components of each outer ring (wall normal)
  select surface_gmlid, p,
         sum(y*nz - z*ny) filter (where r=1) ax, sum(z*nx - x*nz) filter (where r=1) ay
  from seg where nx is not null group by 1,2),
ring_area as (
  select surface_gmlid, p, r, 0.5*sqrt(power(sum(y*nz - z*ny),2) + power(sum(z*nx - x*nz),2) + power(sum(x*ny - y*nx),2)) a
  from seg where nx is not null group by 1,2,3),
part_area as (select surface_gmlid, p, sum(case when r=1 then a else -a end) a from ring_area group by 1,2),
proj as (  -- outer ring in (u, z); u = horizontal direction of the wall plane
  select s.surface_gmlid, s.p,
         ST_MakeValid(ST_MakePolygon(ST_MakeLine(array_agg(
           ST_MakePoint(s.x * (-pa.ay) / nullif(sqrt(pa.ax*pa.ax + pa.ay*pa.ay),0) + s.y * pa.ax / nullif(sqrt(pa.ax*pa.ax + pa.ay*pa.ay),0), s.z)
           order by s.v)))) g2
  from seg s join part pa using (surface_gmlid, p) where s.r = 1 group by 1,2),
wall_part as (
  select w.surface_gmlid, w.b, pa.a,
         case when ST_Area(pr.g2) > 0 and l.z_eaves is not null
              then ST_Area(ST_Intersection(pr.g2, ST_MakeEnvelope(-1e9, l.z_eaves, 1e9, 1e9))) / ST_Area(pr.g2) else 0 end frac_above
  from walls w join part_area pa using (surface_gmlid) join proj pr on pr.surface_gmlid=pa.surface_gmlid and pr.p=pa.p
       join levels l on l.b=w.b),
heated_neighbour as (
  select distinct w.surface_gmlid from walls w join basedata.buildings n
    on n.objectid <> w.b and n.building_use in ('Residential','Mixed') and n.geom && ST_Expand(w.c, 0.3) and ST_DWithin(w.c, n.geom, 0.3)),
per as (
  select wp.b, sum(wp.a) wall_m2, sum(wp.a) filter (where h.surface_gmlid is not null) shared_m2,
         sum(wp.a*wp.frac_above) above_m2, sum(wp.a*wp.frac_above) filter (where h.surface_gmlid is not null) shared_above_m2
  from wall_part wp left join heated_neighbour h using (surface_gmlid) group by 1)
select b.objectid, s.wall_surface_area_m2, s.roof_surface_area_m2,
       s.wall_surface_area_m2 * coalesce(p.shared_m2,0) / nullif(p.wall_m2,0) shared_wall_area_m2,
       s.wall_surface_area_m2 * coalesce(p.above_m2,0) / nullif(p.wall_m2,0) wall_above_eaves_m2,
       s.wall_surface_area_m2 * coalesce(p.shared_above_m2,0) / nullif(p.wall_m2,0) shared_wall_above_eaves_m2,
       l.z_eaves - l.z_ground eaves_height_m
from ids join basedata.buildings b on b.objectid=ids.oid left join sums s on s.b=b.objectid
     left join per p on p.b=b.objectid left join levels l on l.b=b.objectid;
"""

COLUMNS = ["objectid", "wall_surface_area_m2", "roof_surface_area_m2", "shared_wall_area_m2",
           "wall_above_eaves_m2", "shared_wall_above_eaves_m2"]


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
