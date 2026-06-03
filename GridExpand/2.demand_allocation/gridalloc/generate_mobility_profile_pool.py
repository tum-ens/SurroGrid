#!/usr/bin/env python3
"""Generate a persistent CSV pool of pregenerated emobpy mobility profiles."""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

GRIDALLOC_DIR = Path(__file__).resolve().parent
os.chdir(GRIDALLOC_DIR)
if str(GRIDALLOC_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDALLOC_DIR))

from config import config
import src.functions.mobility as mbl
import src.functions.weather as wth


SCHEDULES = ["commuter", "non-commuter"]


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return slug[:60]


def _profile_id(weather_key: str, model_index: int, model: str, schedule: str, sample_index: int) -> str:
    return (
        f"{weather_key}_m{model_index:02d}_{_slug(model)}_"
        f"{schedule.replace('-', '_')}_s{sample_index:04d}"
    )


def _pool_seed(model_index: int, schedule_index: int, sample_index: int) -> int:
    return int(f"{model_index + 1:02d}{schedule_index + 1}{sample_index + 1:04d}")


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _load_or_fetch_weather(weather_csv: Path) -> pd.DataFrame:
    if weather_csv.exists():
        return pd.read_csv(weather_csv)

    weather_csv.parent.mkdir(parents=True, exist_ok=True)
    df_weather, _altitude, _selected_months = wth.get_pvgis_tmy_sarah3_dataframe(
        config.MOBILITY_PROFILE_POOL_LAT,
        config.MOBILITY_PROFILE_POOL_LON,
    )
    df_weather["dew_point"] = wth.get_dew_point(
        df_weather["temp_air"],
        df_weather["relative_humidity"],
    )
    cols = ["temp_air", "pressure", "dew_point", "relative_humidity"]
    df_weather[cols].to_csv(weather_csv, index=False)
    return df_weather[cols]


def _read_existing_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _validate_output_paths(paths: list[Path], append: bool) -> None:
    if append:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Pool CSV already exists: {names}. Use --append to extend it.")


def _select_models(market_share_threshold: float, explicit_models: list[str] | None) -> list[tuple[int, str]]:
    all_models = config.CAR_MODEL_DISTRIBUTION["model"].dropna().astype(str).tolist()
    if explicit_models:
        selected = set(explicit_models)
        missing = selected - set(all_models)
        if missing:
            raise ValueError(f"Unknown model(s): {sorted(missing)}")
        return [(idx, model) for idx, model in enumerate(all_models) if model in selected]

    df_models = config.CAR_MODEL_DISTRIBUTION.copy()
    df_models["cum_probability"] = df_models["probability"].cumsum()
    n_models = int((df_models["cum_probability"] < market_share_threshold).sum() + 1)
    return [(idx, str(model)) for idx, model in enumerate(all_models[:n_models])]


def _planned_sample_indexes(
    existing: pd.DataFrame,
    *,
    model: str,
    schedule: str,
    weather_key: str,
    target_count: int,
) -> list[int]:
    if existing.empty:
        present = set()
    else:
        subset = existing[
            (existing["model"] == model)
            & (existing["schedule"] == schedule)
            & (existing["weather_key"] == weather_key)
        ]
        present = set(subset["sample_index"].astype(int).tolist())
    return [idx for idx in range(target_count) if idx not in present]


def _generate_profile(task: dict, weather_records: dict[str, list]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weather = mbl.prepare_weather_input(pd.DataFrame(weather_records))
    profile_id = task["profile_id"]
    vehicle_key = (0, 0)
    cfg = {"model": task["model"], "schedule": task["schedule"], "seed": task["pool_seed"]}
    mob_demand, availability, battery_dict = mbl.get_mobility_demand({vehicle_key: cfg}, weather)
    demand = mob_demand.iloc[:, 0].reset_index(drop=True)
    avai = availability.iloc[:, 0].reset_index(drop=True)
    battery_cap = float(battery_dict[vehicle_key])

    metadata_row = pd.DataFrame([
        {
            "profile_id": profile_id,
            "schedule": task["schedule"],
            "model": task["model"],
            "sample_index": int(task["sample_index"]),
            "pool_seed": int(task["pool_seed"]),
            "weather_key": task["weather_key"],
            "weather_source": config.MOBILITY_PROFILE_POOL_WEATHER_SOURCE,
            "battery_cap_kwh": battery_cap,
            "total_hours": int(config.TOTAL_HOURS),
            "emobpy_timestep_h": float(config.MBL_TIME_STEP_LENGTH),
            "output_timestep_h": 1.0,
            "ref_year": int(config.REF_YEAR),
            "demand_sum_kwh": float(demand.sum()),
            "availability_hours": float(avai.sum()),
            "generation_version": config.MOBILITY_PROFILE_POOL_GENERATION_VERSION,
        }
    ])
    demand_rows = pd.DataFrame(
        {"profile_id": profile_id, "t": range(len(demand)), "demand_kwh": demand.to_numpy()}
    )
    availability_rows = pd.DataFrame(
        {"profile_id": profile_id, "t": range(len(avai)), "availability": avai.to_numpy()}
    )
    return metadata_row, demand_rows, availability_rows


def generate_pool(args: argparse.Namespace) -> None:
    metadata_path = Path(args.metadata_csv or config.MOBILITY_PROFILE_POOL_METADATA_PATH)
    demand_path = Path(args.demand_csv or config.MOBILITY_PROFILE_POOL_DEMAND_PATH)
    availability_path = Path(args.availability_csv or config.MOBILITY_PROFILE_POOL_AVAILABILITY_PATH)
    weather_csv = Path(args.weather_csv or config.MOBILITY_PROFILE_POOL_WEATHER_PATH)
    weather_key = args.weather_key or config.MOBILITY_PROFILE_POOL_WEATHER_KEY

    _validate_output_paths([metadata_path, demand_path, availability_path], args.append)
    existing = _read_existing_metadata(metadata_path) if args.append else pd.DataFrame()

    models = _select_models(args.market_share_threshold, args.models)
    schedules = args.schedules or SCHEDULES
    planned = []
    for model_index, model in models:
        for schedule_index, schedule in enumerate(SCHEDULES):
            if schedule not in schedules:
                continue
            for sample_index in _planned_sample_indexes(
                existing,
                model=model,
                schedule=schedule,
                weather_key=weather_key,
                target_count=args.profiles_per_stratum,
            ):
                seed = _pool_seed(model_index, schedule_index, sample_index)
                planned.append(
                    {
                        "profile_id": _profile_id(weather_key, model_index, model, schedule, sample_index),
                        "model_index": model_index,
                        "model": model,
                        "schedule_index": schedule_index,
                        "schedule": schedule,
                        "sample_index": sample_index,
                        "pool_seed": seed,
                        "weather_key": weather_key,
                    }
                )

    print(
        f"Planning {len(planned)} profile(s) for {len(models)} model(s), "
        f"{len(schedules)} schedule(s), target {args.profiles_per_stratum} per stratum."
    )
    if args.dry_run or not planned:
        return

    df_weather = _load_or_fetch_weather(weather_csv)
    weather_records = df_weather[["temp_air", "pressure", "dew_point", "relative_humidity"]].to_dict(orient="list")

    if args.n_cpu == 1:
        for position, task in enumerate(planned, start=1):
            print(f"[{position}/{len(planned)}] Generating {task['profile_id']}")
            metadata_row, demand_rows, availability_rows = _generate_profile(task, weather_records)
            _append_csv(metadata_row, metadata_path)
            _append_csv(demand_rows, demand_path)
            _append_csv(availability_rows, availability_path)
        return

    with ProcessPoolExecutor(max_workers=args.n_cpu) as executor:
        futures = {executor.submit(_generate_profile, task, weather_records): task for task in planned}
        for position, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            print(f"[{position}/{len(planned)}] Finished {task['profile_id']}")
            metadata_row, demand_rows, availability_rows = future.result()
            _append_csv(metadata_row, metadata_path)
            _append_csv(demand_rows, demand_path)
            _append_csv(availability_rows, availability_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pregenerated emobpy mobility profile CSV pool.")
    parser.add_argument("--profiles-per-stratum", type=int, default=1)
    parser.add_argument("--market-share-threshold", type=float, default=0.80)
    parser.add_argument("--n_cpu", type=int, default=1)
    parser.add_argument("--append", action="store_true", help="Generate missing sample indexes up to the target count.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned profile count without generating profiles.")
    parser.add_argument("--models", nargs="*", help="Optional exact EV model names to generate.")
    parser.add_argument("--schedules", nargs="*", choices=SCHEDULES, help="Optional schedules to generate.")
    parser.add_argument("--weather-key", help="Pool weather key stored in metadata.")
    parser.add_argument("--weather-csv", help="Weather CSV to use or create.")
    parser.add_argument("--metadata-csv", help="Metadata CSV output path.")
    parser.add_argument("--demand-csv", help="Demand CSV output path.")
    parser.add_argument("--availability-csv", help="Availability CSV output path.")
    args = parser.parse_args()
    if args.profiles_per_stratum < 1:
        raise ValueError("--profiles-per-stratum must be at least 1")
    if not 0 < args.market_share_threshold <= 1:
        raise ValueError("--market-share-threshold must be in (0, 1]")
    if args.n_cpu < 1:
        raise ValueError("--n_cpu must be at least 1")
    return args


if __name__ == "__main__":
    generate_pool(parse_args())
