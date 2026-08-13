"""Materialize real SWF scenario allocation plans as Step-3 URBS HDF inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..paths import GRIDALLOC_DIR, GRIDEXPAND_DIR

DEFAULT_ALLOCATION_PLAN = (
    GRIDALLOC_DIR
    / "outputs"
    / "scenario_calibration"
    / "swf_2045_building_match_91301"
    / "swf_2045_full_local_demand_bus_allocation_plan.csv"
)
DEFAULT_OUTPUT_DIR = GRIDEXPAND_DIR / "3.urbs" / "Input"
DEFAULT_SCENARIO_CONFIG = (
    GRIDEXPAND_DIR / "scenario_pipeline" / "config" / "scenarios"
    / "forchheim_2045.yaml"
)
DEFAULT_SCOPE = "full_local_demand_recommended"
DEFAULT_SCENARIO_LABEL = "real_swf_2045_full_local_base_electricity"

for path in (GRIDEXPAND_DIR, GRIDALLOC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import config  # noqa: E402
from scenario_pipeline.config_loader import load_scenario_config  # noqa: E402
from common.timeframe import build_full_year_metadata, write_hdf_metadata  # noqa: E402
from src.scenario_calibration.profiles.real_swf_electricity_profiles import (  # noqa: E402
    DEFAULT_MEASURED_PROFILE_BAND_PCT,
    DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES,
    MEASURED_PROFILE_SELECTION_CHOICES,
    MEASURED_PROFILE_SELECTION_RANDOM_BAND,
    build_scenario_base_electric_demand,
    load_electricity_module,
    profile_selection_summary,
    read_allocation_plan,
)
from src.scenario_calibration.profiles.real_swf_sector_profiles import (  # noqa: E402
    build_real_swf_sector_urbs_inputs,
)
from src.scenario_calibration.pipeline.urbs_input_tables import (  # noqa: E402
    buy_sell_price,
    empty_timeseries,
    read_or_create_weather,
    urbs_static_tables,
)


def _default_weather_source(plz: int, require_supim: bool = False) -> Path | None:
    input_dir = GRIDEXPAND_DIR / "3.urbs" / "Input"
    candidates = sorted(input_dir.glob(f"*_{plz}_*.h5"))
    for path in candidates:
        try:
            pd.read_hdf(path, key="urbs_in/weather")
            if require_supim:
                supim = pd.read_hdf(path, key="urbs_in/supim")
                if supim.empty:
                    continue
        except Exception:
            continue
        return path
    return None


def _allocation_totals(
    allocation: pd.DataFrame, demand_audit: pd.DataFrame
) -> dict[str, float | int]:
    return {
        "allocation_plan_rows": int(len(allocation)),
        "allocation_plan_buses": int(allocation["allocation_bus"].nunique()),
        "allocation_plan_buildings": int(allocation["building_match_id"].nunique()),
        "allocation_hh_rows": int(
            pd.to_numeric(allocation["residential_equivalent_hh_rows"], errors="coerce")
            .fillna(0.0)
            .sum()
        ),
        "allocation_hh_annual_kwh": float(
            pd.to_numeric(
                allocation["residential_equivalent_hh_annual_kwh"], errors="coerce"
            )
            .fillna(0.0)
            .sum()
        ),
        "allocation_ghd_annual_kwh": float(
            pd.to_numeric(allocation["calibrated_annual_ghd_kwh"], errors="coerce")
            .fillna(0.0)
            .sum()
        ),
        "generated_profile_energy_kwh": float(
            pd.to_numeric(demand_audit["annual_demand_kwh"], errors="coerce")
            .fillna(0.0)
            .sum()
        ),
    }


def materialize_one_real_swf_hdf(
    *,
    allocation_plan_path: Path,
    output_dir: Path,
    lv_id: int,
    plz: int,
    scenario_label: str,
    scope: str,
    seed: int,
    weather_source_hdf: Path | None,
    measured_profile_selection: str,
    measured_profile_band_pct: float,
    measured_profile_min_candidates: int,
    include_sector_assets: bool = False,
    scenario_config_path: Path = DEFAULT_SCENARIO_CONFIG,
) -> Path:
    scenario, _scenario_hash = load_scenario_config(scenario_config_path)
    config.apply_scenario(scenario)
    plan = read_allocation_plan(allocation_plan_path, scope=None)
    allocation = plan[plan["lv_id"].astype(int).eq(int(lv_id))].copy()
    if scope and "scenario_scope" in allocation.columns:
        allocation = allocation[
            allocation["scenario_scope"].astype(str).eq(scope)
        ].copy()
    if allocation.empty:
        raise ValueError(
            f"Allocation plan contains no rows for LV {lv_id} and scope={scope!r}."
        )

    demand, demand_audit = build_scenario_base_electric_demand(
        allocation,
        seed=seed,
        measured_profile_selection=measured_profile_selection,
        measured_profile_band_pct=measured_profile_band_pct,
        measured_profile_min_candidates=measured_profile_min_candidates,
        include_reactive=False,
    )
    demand.index.name = "t"
    hours = len(demand)
    electricity_module = load_electricity_module()
    electricity_module.config.apply_scenario(scenario)
    weather_source_hdf = weather_source_hdf or _default_weather_source(
        plz, require_supim=include_sector_assets
    )
    sector_inputs = None
    if include_sector_assets:
        sector_inputs = build_real_swf_sector_urbs_inputs(
            allocation,
            hours=hours,
            seed=seed,
            source_hdf=weather_source_hdf,
            include_pv=True,
            include_mobility=True,
        )
        if not sector_inputs.demand.empty:
            demand = pd.concat([demand, sector_inputs.demand], axis=1).sort_index(
                axis=1
            )

    active_buses = set(int(bus) for bus in demand.columns.get_level_values(0).unique())
    if sector_inputs is not None:
        for table in [
            sector_inputs.process,
            sector_inputs.commodity,
            sector_inputs.storage,
        ]:
            if not table.empty and "Site" in table.columns:
                active_buses.update(
                    int(bus)
                    for bus in pd.to_numeric(table["Site"], errors="coerce")
                    .dropna()
                    .astype(int)
                )
    static_tables = urbs_static_tables(sorted(active_buses), electricity_module)
    if sector_inputs is not None:
        for key in ["process", "commodity", "process_commodity", "storage"]:
            sector_table = getattr(sector_inputs, key)
            if not sector_table.empty:
                static_tables[key] = pd.concat(
                    [static_tables[key], sector_table], ignore_index=True, sort=False
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"real_swf_LV_{int(lv_id):03d}_{scenario_label}.h5"
    if output_path.exists():
        output_path.unlink()

    metadata = {
        **build_full_year_metadata(),
        "source": "real_swf",
        "scenario_label": scenario_label,
        "scenario_scope": scope,
        "plz": int(plz),
        "lv_id": int(lv_id),
        "allocation_plan_path": str(allocation_plan_path),
        "weather_source_hdf": None
        if weather_source_hdf is None
        else str(weather_source_hdf),
        "sector_coupling_assets_simulated": bool(
            include_sector_assets
            and sector_inputs is not None
            and sector_inputs.metadata.get("sector_assets_simulated")
        ),
        "demand_handover": "HH and calibrated GHD active electricity plus optional explicit PV/EV sector assets; same Step-3 URBS HDF contract as synthetic inputs.",
        **_allocation_totals(allocation, demand_audit),
        **profile_selection_summary(demand_audit),
        **({} if sector_inputs is None else sector_inputs.metadata),
    }

    write_hdf_metadata(output_path, metadata)
    with pd.HDFStore(output_path, mode="a", complib="blosc", complevel=9) as store:
        store.put("raw_data/allocation_plan", allocation.reset_index(drop=True))
        store.put("raw_data/demand_profile_audit", demand_audit.reset_index(drop=True))
        store.put("urbs_in/demand", demand)
        store.put(
            "urbs_in/supim",
            empty_timeseries(hours) if sector_inputs is None else sector_inputs.supim,
        )
        store.put(
            "urbs_in/eff_factor",
            empty_timeseries(hours)
            if sector_inputs is None
            else sector_inputs.eff_factor,
        )
        if sector_inputs is not None and not sector_inputs.audit.empty:
            store.put(
                "raw_data/sector_profile_audit",
                sector_inputs.audit.reset_index(drop=True),
            )
        store.put("urbs_in/buy_sell_price", buy_sell_price(hours, electricity_module))
        store.put("urbs_in/weather", read_or_create_weather(weather_source_hdf, hours))
        for key, table in static_tables.items():
            store.put(f"urbs_in/{key}", table)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize real SWF scenario allocation plans as URBS HDF inputs."
    )
    parser.add_argument("--allocation-plan", type=Path, default=DEFAULT_ALLOCATION_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lv-id", type=int, required=True)
    parser.add_argument("--plz", type=int, default=91301)
    parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIO_CONFIG)
    parser.add_argument("--scenario-label", default=DEFAULT_SCENARIO_LABEL)
    parser.add_argument("--scope", default=DEFAULT_SCOPE)
    parser.add_argument("--seed", type=int, default=91301)
    parser.add_argument("--weather-source-hdf", type=Path, default=None)
    parser.add_argument(
        "--measured-profile-selection",
        choices=MEASURED_PROFILE_SELECTION_CHOICES,
        default=MEASURED_PROFILE_SELECTION_RANDOM_BAND,
    )
    parser.add_argument(
        "--measured-profile-band-pct",
        type=float,
        default=DEFAULT_MEASURED_PROFILE_BAND_PCT,
    )
    parser.add_argument(
        "--measured-profile-min-candidates",
        type=int,
        default=DEFAULT_MEASURED_PROFILE_MIN_CANDIDATES,
    )
    parser.add_argument(
        "--include-sector-assets",
        action="store_true",
        help="Add explicit PV, EV, and heat/WP sector assets using shared synthetic profile-library assumptions.",
    )
    args = parser.parse_args()

    output_path = materialize_one_real_swf_hdf(
        allocation_plan_path=args.allocation_plan,
        output_dir=args.output_dir,
        lv_id=args.lv_id,
        plz=args.plz,
        scenario_label=args.scenario_label,
        scope=args.scope,
        seed=args.seed,
        weather_source_hdf=args.weather_source_hdf,
        measured_profile_selection=args.measured_profile_selection,
        measured_profile_band_pct=args.measured_profile_band_pct,
        scenario_config_path=args.scenario_config,
        measured_profile_min_candidates=args.measured_profile_min_candidates,
        include_sector_assets=args.include_sector_assets,
    )
    print(output_path)


if __name__ == "__main__":
    main()
