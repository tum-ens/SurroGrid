"""Materialize the same physical-building scenario on either target network."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..paths import GRIDALLOC_DIR, GRIDEXPAND_DIR

DEFAULT_PAIRED_DIR = (
    GRIDALLOC_DIR
    / "outputs"
    / "scenario_calibration"
    / "swf_2045_paired_v5_91301_station_hybrid_v2"
)
DEFAULT_OUTPUT_DIR = GRIDEXPAND_DIR / "3.urbs" / "Input"

for path in (GRIDEXPAND_DIR, GRIDALLOC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.timeframe import build_full_year_metadata, write_hdf_metadata  # noqa: E402
from src.scenario_calibration.profiles.paired_profiles import (  # noqa: E402
    build_paired_base_electric_demand,
    build_paired_sector_urbs_inputs,
    load_electricity_module,
)
from src.scenario_calibration.profiles.physical_heat_profile_library import (  # noqa: E402
    PhysicalHeatProfileLibrary,
)
from src.scenario_calibration.pipeline.urbs_input_tables import (  # noqa: E402
    buy_sell_price,
    read_or_create_weather,
    urbs_static_tables,
)


def _plan_path(paired_dir: Path, target_network: str) -> Path:
    filename = (
        "paired_real_bus_allocation_plan.csv"
        if target_network == "real_swf"
        else "paired_synthetic_bus_allocation_plan.csv"
    )
    return paired_dir / filename


def _target_plan(
    paired_dir: Path,
    target_network: str,
    target_grid_id: int,
) -> pd.DataFrame:
    plan = pd.read_csv(_plan_path(paired_dir, target_network))
    target = plan[
        pd.to_numeric(plan["target_grid_id"], errors="coerce")
        .astype("Int64")
        .eq(int(target_grid_id))
    ].copy()
    if target.empty:
        raise ValueError(
            f"No {target_network} allocation rows for target grid {target_grid_id}."
        )
    return target


def _output_name(
    allocation: pd.DataFrame,
    target_network: str,
    target_grid_id: int,
    scenario_label: str,
) -> str:
    if target_network == "real_swf":
        prefix = f"paired_real_swf_LV_{int(target_grid_id):03d}"
    else:
        bridge_names = (
            allocation["synthetic_bridge_filename"].dropna().astype(str).unique()
        )
        if len(bridge_names) != 1:
            raise ValueError("A synthetic target grid must have one bridge filename.")
        prefix = f"paired_synthetic_{Path(bridge_names[0]).stem}"
    return f"{prefix}_{scenario_label}.h5"


def _combine_static_tables(
    active_buses: list[int],
    sector_inputs,
    electricity_module,
) -> dict[str, pd.DataFrame]:
    tables = urbs_static_tables(
        active_buses,
        electricity_module,
        include_generic_battery=False,
    )
    for key in ("process", "commodity", "process_commodity", "storage"):
        sector_table = getattr(sector_inputs, key)
        if not sector_table.empty:
            tables[key] = pd.concat(
                [tables[key], sector_table],
                ignore_index=True,
                sort=False,
            )
    return tables


def materialize_paired_urbs_input(
    *,
    paired_dir: Path,
    output_dir: Path,
    target_network: str,
    target_grid_id: int,
    scenario_label: str,
    seed: int,
    weather_source_hdf: Path,
    heat_profile_library: Path | None,
    allow_diagnostic_heat_fallback: bool,
) -> Path:
    """Write one paired full-year Step-3 input HDF."""
    paired_dir = paired_dir.resolve()
    allocation = _target_plan(
        paired_dir,
        target_network,
        target_grid_id,
    )
    if "scenario_unit_id" not in allocation:
        raise ValueError(
            "Paired allocation is missing scenario_unit_id. Regenerate it with "
            "paired_allocation before materializing URBS inputs."
        )
    allocation["_profile_site_id"] = allocation["scenario_unit_id"].astype(int)
    catalog_path = paired_dir / "paired_heat_profile_catalog.csv"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Run paired_profile_readiness first: {catalog_path}")
    heat_catalog = pd.read_csv(catalog_path)
    library_sources = (
        heat_catalog.get("profile_source_kind", pd.Series(dtype=str))
        .astype(str)
        .eq("physical_heat_library")
    )
    if library_sources.any():
        if heat_profile_library is None:
            raise ValueError(
                "The paired heat catalog requires --heat-profile-library."
            )
        library = PhysicalHeatProfileLibrary(heat_profile_library)
        expected_profile_sets = (
            heat_catalog.loc[library_sources, "profile_set_id"]
            .dropna()
            .astype(str)
            .unique()
        )
        profile_set_matches = (
            len(expected_profile_sets) == 1
            and library.profile_set_id == expected_profile_sets[0]
        )
        if not profile_set_matches:
            raise ValueError(
                "Heat-profile library mismatch: paired catalog expects "
                f"{expected_profile_sets.tolist()}, got {library.profile_set_id!r}."
            )
    demand, demand_audit = build_paired_base_electric_demand(
        allocation,
        seed=seed,
    )
    sector_inputs = build_paired_sector_urbs_inputs(
        allocation,
        hours=len(demand),
        seed=seed,
        weather_source_hdf=weather_source_hdf.resolve(),
        heat_profile_catalog=heat_catalog,
        heat_profile_library=heat_profile_library,
        allow_diagnostic_heat_fallback=allow_diagnostic_heat_fallback,
    )
    if not sector_inputs.demand.empty:
        demand = pd.concat(
            [demand, sector_inputs.demand],
            axis=1,
        ).sort_index(axis=1)
    active_buses = sorted(
        int(bus) for bus in demand.columns.get_level_values(0).unique()
    )
    electricity_module = load_electricity_module()
    static_tables = _combine_static_tables(
        active_buses,
        sector_inputs,
        electricity_module,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / _output_name(
        allocation,
        target_network,
        target_grid_id,
        scenario_label,
    )
    if output.exists():
        output.unlink()
    heat_audit = sector_inputs.audit[
        sector_inputs.audit.get("sector", pd.Series(dtype=str)).eq("heat")
    ]
    metadata = {
        **build_full_year_metadata(),
        "source": "paired_swf_2045",
        "target_network": target_network,
        "target_grid_id": int(target_grid_id),
        "scenario_label": scenario_label,
        "scenario_scope": "paired_full_local_demand",
        "profile_contract": "physical_building_paired_v1",
        "optimization_space": "scenario_unit",
        "paired_dir": str(paired_dir),
        "profile_seed": int(seed),
        "physical_buildings": int(allocation["building_objectid"].nunique()),
        "hh_rows": float(allocation["residential_equivalent_hh_rows"].sum()),
        "hh_annual_kwh": float(
            allocation["residential_equivalent_hh_annual_kwh"].sum()
        ),
        "ghd_annual_kwh": float(allocation["calibrated_annual_ghd_kwh"].sum()),
        "heat_profile_fallback_buildings": int(
            heat_audit.loc[
                heat_audit.get("profile_method", "").ne("exact_physical_building"),
                "building_objectid",
            ].nunique()
        )
        if not heat_audit.empty
        else 0,
        "publication_ready": bool(
            heat_audit.empty
            or heat_audit.get(
                "profile_method",
                pd.Series(dtype=str),
            )
            .eq("exact_physical_building")
            .all()
        ),
        **sector_inputs.metadata,
    }
    write_hdf_metadata(output, metadata)
    with pd.HDFStore(
        output,
        mode="a",
        complib="blosc",
        complevel=9,
    ) as store:
        store.put("raw_data/allocation_plan", allocation.reset_index(drop=True))
        store.put(
            "raw_data/demand_profile_audit",
            demand_audit.reset_index(drop=True),
        )
        store.put(
            "raw_data/sector_profile_audit",
            sector_inputs.audit.reset_index(drop=True),
        )
        store.put("urbs_in/demand", demand)
        store.put("urbs_in/supim", sector_inputs.supim)
        store.put("urbs_in/eff_factor", sector_inputs.eff_factor)
        store.put(
            "urbs_in/buy_sell_price",
            buy_sell_price(len(demand), electricity_module),
        )
        store.put(
            "urbs_in/weather",
            read_or_create_weather(
                weather_source_hdf.resolve(),
                len(demand),
            ),
        )
        for key, table in static_tables.items():
            store.put(f"urbs_in/{key}", table)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-dir", type=Path, default=DEFAULT_PAIRED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-network",
        choices=["real_swf", "synthetic"],
        required=True,
    )
    parser.add_argument("--target-grid-id", type=int, required=True)
    parser.add_argument(
        "--scenario-label",
        default="swf_2045_paired_full_local",
    )
    parser.add_argument("--seed", type=int, default=91301)
    parser.add_argument("--weather-source-hdf", type=Path, required=True)
    parser.add_argument("--heat-profile-library", type=Path)
    parser.add_argument(
        "--allow-diagnostic-heat-fallback",
        action="store_true",
        help=(
            "Permit area-scaled proxy heat profiles for buildings whose "
            "current residential-only synthetic source HDF has no exact heat "
            "profile. Such outputs are marked publication_ready=false."
        ),
    )
    args = parser.parse_args()
    output = materialize_paired_urbs_input(
        paired_dir=args.paired_dir,
        output_dir=args.output_dir,
        target_network=args.target_network,
        target_grid_id=args.target_grid_id,
        scenario_label=args.scenario_label,
        seed=args.seed,
        weather_source_hdf=args.weather_source_hdf,
        heat_profile_library=(
            args.heat_profile_library.resolve()
            if args.heat_profile_library is not None
            else None
        ),
        allow_diagnostic_heat_fallback=(args.allow_diagnostic_heat_fallback),
    )
    print(output)


if __name__ == "__main__":
    main()
