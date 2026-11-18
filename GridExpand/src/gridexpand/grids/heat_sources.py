"""Per-building heat profiles of a scenario dataset with a fixed heat source (``infdb_ro_heat`` or ``teaser``).

The validation regenerates these profiles by running the former synthetic Step 2 on the pylovo grids of the same
buildings; own grids have no such grids. This module computes them building by building with the same functions
and seeds (statistics, space heat, hot water, heat-pump COP; daylight saving shifts as in Step 2):

- ``infdb_ro_heat``: space heat from the InfDB ``ro_heat`` series, hot water from OpenDHW;
- ``teaser``: TEASER/DistrictGenerator with the building's own electricity profile as internal gains.

It writes one source HDF per grid (``urbs_in/demand`` with ``space_heat``/``water_heat`` and ``urbs_in/eff_factor``
with ``heatpump_air``, one pseudo bus per building = its scenario unit), the physical heat-profile library and the
final ``paired_heat_profile_catalog.csv`` (the catalog function of the validation, so both read alike).

``uv run python -m gridexpand.grids.heat_sources --paired-dir <dataset> --scenario-config <yaml> --weather-hdf <file>
--sources-dir <dir> --library <file.h5> --profile-set-id <id>``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gridexpand.allocation.functions import heat
from gridexpand.allocation.functions.dst import dst_shift_input, dst_shift_output
from gridexpand.scenario.config_loader import load_scenario_config

SOURCES = ("infdb_ro_heat", "teaser")


def _residential(paired_dir: Path) -> pd.DataFrame:
    """The heat-eligible residential buildings with one pseudo bus each (their scenario unit)."""
    buildings = pd.read_csv(paired_dir / "paired_physical_buildings.csv", dtype={"building_objectid": str},
                            low_memory=False)
    assignment = pd.read_csv(paired_dir / "paired_electrification_assignment.csv", dtype={"building_objectid": str})
    eligible = set(assignment.loc[assignment["technology"].eq("heat") & assignment["eligible"].astype(bool),
                                  "building_objectid"])
    area = pd.to_numeric(buildings["residential_effective_floor_area_m2"], errors="coerce")
    residential = buildings[buildings["building_objectid"].isin(eligible) & area.gt(0)].copy()
    residential = residential.rename(columns={"building_objectid": "objectid"})
    residential["objectid"] = residential["objectid"].astype(str)
    residential["bus"] = residential["scenario_unit_id"].astype(int)
    gross = (pd.to_numeric(residential["floor_area"], errors="coerce")
             * pd.to_numeric(residential["floor_number"], errors="coerce"))
    residential["residential_area_share"] = area.loc[residential.index] / gross
    return residential.reset_index(drop=True)


def _electricity_by_bus(group: pd.DataFrame, profile_seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Residential electricity of each building at its pseudo bus (TEASER internal gains) and its occupants per
    household (``occ_list``), sampled as in the dataset."""
    import gridexpand.allocation.functions.electricity as electricity
    from gridexpand.common.building_components import build_building_components

    components = build_building_components(group)
    components = components[components["component_category"].eq("Residential")]
    if components.empty or not components["included_in_lv"].astype(bool).any():
        return pd.DataFrame(), pd.Series(dtype=object)
    sampled, by_bus, _ = electricity.profile_components(components, base_seed=profile_seed)
    occupants = sampled.drop_duplicates("objectid").set_index(sampled["objectid"].astype(str))["occ_list"]
    return by_bus, occupants


def grid_heat_profiles(residential: pd.DataFrame, weather: pd.DataFrame, *, source: str, plz: int,
                       profile_seed: int, teaser=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(demand with space_heat/water_heat, COP, source audit)`` of the buildings of one grid."""
    residential = heat.sample_statistics(residential.copy(), profile_seed)
    audit: dict = {}
    if source == "infdb_ro_heat":
        from gridexpand.allocation.functions.infdb_ro_heat import generate_opendhw, load_space_heat

        space, audit = load_space_heat(residential)
        water = generate_opendhw(residential, base_seed=profile_seed)
    else:
        electricity, occupants = _electricity_by_bus(residential, profile_seed)
        residential["occ_list"] = [value if isinstance(value, list) else []
                                   for value in residential["objectid"].map(occupants)]
        columns = [(bus, "electricity") for bus in residential["bus"]]
        electricity_input = dst_shift_input(electricity.reindex(columns=columns, fill_value=0.0)) \
            if not electricity.empty else pd.DataFrame(0.0, index=range(len(weather)),
                                                      columns=pd.MultiIndex.from_tuples(columns))
        space, water = heat.generate_heat_demands(
            residential, electricity_input, heat.teaser_weather_arrays(dst_shift_input(weather), teaser),
            plz, profile_seed, teaser,
        )
    heat.require_heat_profiles(residential["bus"], space, water)
    space, water = dst_shift_output(space), dst_shift_output(water)
    cop = heat.generate_hp_cop(residential, space, water, weather)
    return pd.concat([space, water], axis=1).sort_index(axis=1), cop, dict(audit)


def build_heat_sources(paired_dir: Path, *, scenario_config: Path, weather_hdf: Path, sources_dir: Path,
                       library: Path, profile_set_id: str) -> pd.DataFrame:
    """Write the per-grid source HDFs, the heat-profile library and the final catalog; return the catalog."""
    from gridexpand.allocation.config import config as grid_config
    from gridexpand.allocation.scenario_calibration.profiles.paired_profile_readiness import (
        build_heat_profile_catalog,
        profile_readiness_summary,
    )
    from gridexpand.allocation.scenario_calibration.profiles.physical_heat_profile_library import (
        create_physical_heat_profile_library,
    )

    scenario, _ = load_scenario_config(scenario_config)
    source = scenario.heat.space_heat_source
    if source not in SOURCES:
        raise ValueError(f"Per-building heat sources are for {SOURCES}; the scenario uses {source!r}.")
    grid_config.apply_scenario(scenario)
    metadata = json.loads((paired_dir / "paired_scenario_metadata.json").read_text(encoding="utf-8"))
    profile_seed, plz = int(metadata["profile_seed"]), int(metadata["plz"])
    weather = pd.read_hdf(weather_hdf, key="raw_data/weather").reset_index(drop=True)
    residential = _residential(paired_dir)
    if residential.empty:
        raise ValueError(f"{paired_dir} has no heat-eligible residential building.")
    sources_dir.mkdir(parents=True, exist_ok=True)
    rows, audits = [], []
    for grid, group in residential.groupby("grid_number", sort=True):
        name = f"heat_grid_{int(grid):04d}.h5"
        demand, cop, audit = grid_heat_profiles(group.reset_index(drop=True), weather, source=source, plz=plz,
                                                profile_seed=profile_seed, teaser=scenario.heat.teaser)
        target = sources_dir / name
        target.unlink(missing_ok=True)
        with pd.HDFStore(target, mode="w") as store:
            store.put("urbs_in/demand", demand)
            store.put("urbs_in/eff_factor", cop)
        audits.append({"grid": int(grid), **audit})
        rows.extend({"building_objectid": row.objectid, "synthetic_bridge_filename": name,
                     "synthetic_bus": int(row.bus)} for row in group.itertuples())
    sources = pd.DataFrame(rows)
    plan = pd.read_csv(paired_dir / "paired_building_scenario_plan.csv", dtype={"building_objectid": str})
    eligible = set(sources["building_objectid"])
    # The library takes the publication-ready exact profiles of a source catalog.
    attributes = plan.drop_duplicates("building_objectid").set_index("building_objectid")
    source_catalog = sources.assign(
        building_use=sources["building_objectid"].map(attributes["building_use"]),
        building_type=sources["building_objectid"].map(attributes["building_type"]),
        building_floor_area=sources["building_objectid"].map(attributes["residential_effective_floor_area_m2"]),
        exact_source_hdf=sources["synthetic_bridge_filename"], exact_source_bus=sources["synthetic_bus"],
        publication_ready=True,
    )
    source_catalog_path = paired_dir / "paired_heat_source_catalog.csv"
    source_catalog.to_csv(source_catalog_path, index=False)
    create_physical_heat_profile_library(source_catalog=source_catalog_path, source_hdf_dir=sources_dir,
                                         output=library, profile_set_id=profile_set_id, source_mode="exact")
    # The final catalog with the validation's function; the allocation carries the sources in place of the
    # synthetic grids the validation takes its profiles from.
    allocation = pd.read_csv(paired_dir / "paired_real_bus_allocation_plan.csv", dtype={"building_objectid": str})
    allocation = allocation.drop(columns=["synthetic_bridge_filename", "synthetic_bus"], errors="ignore").merge(
        sources, on="building_objectid", how="inner")
    catalog = build_heat_profile_catalog(allocation, plan, synthetic_input_dir=sources_dir,
                                         heat_profile_library=library, building_ids=eligible)
    catalog.to_csv(paired_dir / "paired_heat_profile_catalog.csv", index=False)
    profile_readiness_summary(catalog).to_csv(paired_dir / "paired_heat_profile_readiness_summary.csv", index=False)
    (paired_dir / "paired_heat_source_audit.json").write_text(json.dumps(audits, indent=2, default=str),
                                                              encoding="utf-8")
    return catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gridexpand.grids.heat_sources", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paired-dir", type=Path, required=True)
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument("--weather-hdf", type=Path, required=True)
    parser.add_argument("--sources-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--profile-set-id", required=True)
    args = parser.parse_args(argv)
    catalog = build_heat_sources(args.paired_dir.resolve(), scenario_config=args.scenario_config.resolve(),
                                 weather_hdf=args.weather_hdf.resolve(), sources_dir=args.sources_dir.resolve(),
                                 library=args.library.resolve(), profile_set_id=args.profile_set_id)
    ready = int(catalog["publication_ready"].astype(bool).sum())
    print(f"{ready} of {len(catalog)} heat profiles ready; library {args.library}")
    return 0 if ready == len(catalog) else 1


if __name__ == "__main__":
    raise SystemExit(main())
