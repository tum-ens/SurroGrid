#!/usr/bin/env python3
"""Generate a persistent CSV pool of pregenerated emobpy mobility profiles.

Two generation modes exist and must never be mixed in one directory:

``deadline`` (``emobpy_pool_v1``)
    The legacy hourly deadline-demand pool. It reallocates non-home charging,
    collapses each home block onto its final hour and clips against the charger
    and battery capacities, so it does not conserve energy.

``session`` (``emobpy_pool_v2_sessions``)
    The full-year chronological reference. It retains the raw 0.5 h emobpy
    records, a stage-wise energy ledger and the dedicated charging-session
    tables described in FULL_YEAR_REFERENCE_DESIGN.md. No energy is clipped;
    an infeasible session is reported, never repaired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from gridexpand.allocation.config import config
import gridexpand.allocation.functions.mobility as mbl
import gridexpand.common.weather as wth
from gridexpand.common.ev_sessions import build_sessions_from_source
from gridexpand.paths import SCENARIO_CONFIG_DIR
from gridexpand.scenario.config_loader import load_scenario_config

DEFAULT_SCENARIO_CONFIG = SCENARIO_CONFIG_DIR / "forchheim_2045_full_year.yaml"


SCHEDULES = ["commuter", "non-commuter"]

SESSION_GENERATION_VERSION = "emobpy_pool_v2_sessions"
SESSION_POOL_DIRNAME = "mobility_profile_pool"
LEGACY_POOL_DIRNAME = "mobility_profile_pool_old"
SOURCE_RECORDS_FILENAME = "mobility_source_records.h5"
SESSIONS_FILENAME = "mobility_sessions_pool.csv"
SESSION_HOURS_FILENAME = "mobility_session_hours_pool.csv"
LEDGER_FILENAME = "mobility_energy_ledger.csv"
POOL_MANIFEST_FILENAME = "mobility_pool_manifest.json"
POOL_MANIFEST_FORMAT = 1


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
    df_weather, _altitude = wth.get_pvgis_tmy_sarah3_dataframe(
        config.MOBILITY_PROFILE_POOL_LAT,
        config.MOBILITY_PROFILE_POOL_LON,
        reference_year=config.REF_YEAR,
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


def _init_session_worker(scenario_config_path: str) -> None:
    """Apply the scenario configuration inside a freshly spawned worker.

    Workers are recycled to bound emobpy's memory growth, so a new process does
    not inherit the scenario applied in __main__ and must load it itself.
    """
    scenario, _hash = load_scenario_config(Path(scenario_config_path))
    config.apply_scenario(scenario)


def _generate_session_profile(task: dict, weather_records: dict[str, list]):
    """Generate one profile as raw records, an energy ledger and session tables."""
    weather = mbl.prepare_weather_input(pd.DataFrame(weather_records))
    profile_id = task["profile_id"]
    vehicle_key = (0, 0)
    cfg = {"model": task["model"], "schedule": task["schedule"], "seed": task["pool_seed"]}
    records_by_vehicle, battery_dict = mbl.get_mobility_source_records(
        {vehicle_key: cfg}, weather
    )
    records = records_by_vehicle[vehicle_key]
    battery_cap = float(battery_dict[vehicle_key])
    step_hours = float(config.MBL_TIME_STEP_LENGTH)

    ledger = mbl.source_energy_ledger(records, battery_cap, step_hours)
    # The reference is a periodically repeated charging-service surrogate: it
    # reproduces the sampled home charging exactly and does NOT claim that the
    # underlying vehicle realization repeats physically year on year. Joining the
    # December and January fragments conserves the sampled charge but does not
    # restore end-of-year stored energy. See FULL_YEAR_REFERENCE_DESIGN.md §3.4.
    #
    # A state-of-charge residual is therefore a recorded property of the source
    # realization in every case, whether the year ends at home or away. An
    # earlier draft blocked the away case on the grounds that "no session can
    # account for it", which silently assumed the periodicity this reference does
    # not claim. What remains hard is the energy identity below: charge in, minus
    # traction out, must equal the change in stored energy.
    residual_kwh = abs(ledger["soc_residual_kwh"])
    ends_connected = str(records["charging_point"].iloc[-1]) == "home"
    if abs(ledger["battery_balance_residual_kwh"]) > 1e-6:
        raise ValueError(
            f"Profile {profile_id} battery energy balance does not close: "
            f"charge_battery={ledger['charge_battery_kwh']:.9f} kWh, "
            f"traction={ledger['traction_kwh']:.9f} kWh, "
            f"soc_change={ledger['soc_residual_kwh']:.9f} kWh, "
            f"residual={ledger['battery_balance_residual_kwh']:.3e} kWh."
        )
    ledger["annually_periodic"] = bool(residual_kwh <= 1e-9)
    ledger["ends_connected_at_home"] = bool(ends_connected)

    sessions, hours = build_sessions_from_source(
        records,
        profile_id=profile_id,
        charger_kw=float(config.CAPACITY_HOME_CHARGING),
        battery_cap_kwh=battery_cap,
        source_timestep_hours=step_hours,
        horizon_hours=int(config.TOTAL_HOURS),
        site="pool",
        vehicle_index=0,
        generation_version=SESSION_GENERATION_VERSION,
        reference_start=pd.Timestamp(f"{int(config.REF_YEAR)}-01-01T00:00:00+00:00"),
    )

    session_index = {
        session_id: index for index, session_id in enumerate(sessions["session_id"])
    }
    # The pool is charger- and site-agnostic: identity columns are re-keyed when
    # a profile is assigned to a physical building in Step 2.
    sessions_out = sessions.drop(
        columns=["session_id", "site", "vehicle_index", "process", "profile_id"]
    ).copy()
    sessions_out.insert(0, "session_index", range(len(sessions_out)))
    sessions_out.insert(0, "profile_id", profile_id)
    hours_out = hours.copy()
    hours_out.insert(0, "session_index", hours_out["session_id"].map(session_index))
    hours_out.insert(0, "profile_id", profile_id)
    hours_out = hours_out.drop(columns=["session_id"])

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
            "emobpy_timestep_h": step_hours,
            "output_timestep_h": 1.0,
            "ref_year": int(config.REF_YEAR),
            "reference_charger_kw": float(config.CAPACITY_HOME_CHARGING),
            "sessions": int(len(sessions_out)),
            "session_energy_kwh": float(sessions_out["energy_kwh"].sum()),
            "wrap_sessions": int(sessions_out["wraps_year"].sum()),
            "merged_source_intervals": int(
                (sessions_out["merged_from"] > 1).sum()
            ),
            "max_energy_over_battery_ratio": float(
                sessions_out["energy_over_battery_ratio"].max()
            ),
            "generation_version": SESSION_GENERATION_VERSION,
        }
    ])
    ledger_row = pd.DataFrame([{"profile_id": profile_id, **ledger}])
    records_out = records.copy()
    records_out.insert(0, "profile_id", profile_id)
    return metadata_row, ledger_row, sessions_out, hours_out, records_out


def _append_source_records(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = df.copy()
    frame["source_timestamp"] = frame["source_timestamp"].astype(str)
    with pd.HDFStore(path, mode="a", complib="blosc", complevel=9) as store:
        store.append(
            "source_records",
            frame,
            format="table",
            data_columns=["profile_id"],
            min_itemsize={"profile_id": 80, "state": 24, "charging_point": 16,
                          "source_timestamp": 40},
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pool_artifact_paths(output_dir: Path) -> dict:
    return {
        "metadata": output_dir / "mobility_profile_pool_metadata.csv",
        "ledger": output_dir / LEDGER_FILENAME,
        "sessions": output_dir / SESSIONS_FILENAME,
        "session_hours": output_dir / SESSION_HOURS_FILENAME,
        "source_records": output_dir / SOURCE_RECORDS_FILENAME,
    }


def validate_pool_completeness(output_dir: Path) -> dict:
    """Check that every declared profile is complete in every artifact.

    Metadata is the completion marker, but a profile is only usable when its
    ledger, session, session-hour and raw-record rows are all present too. An
    interrupted or partially written profile is reported, never silently used.
    """
    paths = _pool_artifact_paths(output_dir)
    missing_files = [name for name, path in paths.items() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Pool at {output_dir} is missing artifacts: {sorted(missing_files)}"
        )
    metadata = pd.read_csv(paths["metadata"])
    declared = list(metadata["profile_id"].astype(str))
    if len(set(declared)) != len(declared):
        duplicates = sorted({name for name in declared if declared.count(name) > 1})
        raise ValueError(f"Pool metadata contains duplicate profiles: {duplicates[:5]}")

    ledger = pd.read_csv(paths["ledger"])
    sessions = pd.read_csv(paths["sessions"])
    hours = pd.read_csv(paths["session_hours"])
    declared_set = set(declared)

    problems = {}
    ledger_missing = declared_set - set(ledger["profile_id"].astype(str))
    if ledger_missing:
        problems["ledger"] = sorted(ledger_missing)[:5]
    # A profile with zero home sessions legitimately has no session rows, so
    # completeness is checked against the session count metadata declares.
    declared_sessions = metadata.set_index("profile_id")["sessions"].astype(int)
    actual_sessions = sessions.groupby("profile_id").size()
    for profile_id, expected in declared_sessions.items():
        actual = int(actual_sessions.get(profile_id, 0))
        if actual != int(expected):
            problems.setdefault("sessions", []).append(
                {"profile_id": profile_id, "declared": int(expected), "found": actual}
            )
    session_hour_profiles = set(hours["profile_id"].astype(str))
    hours_missing = {
        profile_id
        for profile_id, expected in declared_sessions.items()
        if int(expected) > 0 and profile_id not in session_hour_profiles
    }
    if hours_missing:
        problems["session_hours"] = sorted(hours_missing)[:5]

    with pd.HDFStore(paths["source_records"], mode="r") as store:
        record_rows = int(store.get_storer("source_records").nrows)
    steps = int(metadata["total_hours"].iloc[0] / metadata["emobpy_timestep_h"].iloc[0])
    expected_rows = len(declared) * steps
    if record_rows != expected_rows:
        problems["source_records"] = {
            "expected_rows": expected_rows,
            "found_rows": record_rows,
        }

    if problems:
        raise ValueError(
            f"Pool at {output_dir} is incomplete or inconsistent: {problems}"
        )
    return {
        "profiles": len(declared),
        "profile_ids": sorted(declared),
        "sessions": int(len(sessions)),
        "session_hours": int(len(hours)),
        "source_record_rows": record_rows,
    }


def write_pool_manifest(output_dir: Path, *, scenario_hash: str = "") -> Path:
    """Freeze the pool's content identity after validating its completeness.

    Consumers pin this manifest, so adding profiles later cannot silently change
    an existing building's vehicle assignment: the pinned identity no longer
    matches and the run is rejected rather than quietly re-paired.
    """
    summary = validate_pool_completeness(output_dir)
    paths = _pool_artifact_paths(output_dir)
    manifest = {
        "generation_version": SESSION_GENERATION_VERSION,
        "pool_format": POOL_MANIFEST_FORMAT,
        "scenario_hash": str(scenario_hash),
        "profiles": summary["profiles"],
        "profile_ids": summary["profile_ids"],
        "sessions": summary["sessions"],
        "session_hours": summary["session_hours"],
        "source_record_rows": summary["source_record_rows"],
        "artifact_sha256": {
            name: _file_sha256(path) for name, path in sorted(paths.items())
        },
    }
    manifest["pool_id"] = hashlib.sha256(
        json.dumps(
            {
                key: manifest[key]
                for key in ("generation_version", "pool_format", "profile_ids",
                            "artifact_sha256")
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    path = output_dir / POOL_MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def generate_session_pool(args: argparse.Namespace) -> None:
    """Generate the conservation-preserving session pool (``emobpy_pool_v2``)."""
    output_dir = Path(args.output_dir or (Path(config.DATA_STAT_DIR) / "general" / SESSION_POOL_DIRNAME))
    if output_dir.resolve().name == LEGACY_POOL_DIRNAME:
        raise ValueError(
            "Refusing to write session-pool artifacts into the published "
            f"'{LEGACY_POOL_DIRNAME}' directory. Use a new output namespace."
        )
    metadata_path = output_dir / "mobility_profile_pool_metadata.csv"
    ledger_path = output_dir / LEDGER_FILENAME
    sessions_path = output_dir / SESSIONS_FILENAME
    hours_path = output_dir / SESSION_HOURS_FILENAME
    records_path = output_dir / SOURCE_RECORDS_FILENAME
    weather_csv = Path(args.weather_csv or config.MOBILITY_PROFILE_POOL_WEATHER_PATH)
    weather_key = args.weather_key or config.MOBILITY_PROFILE_POOL_WEATHER_KEY

    _validate_output_paths(
        [metadata_path, ledger_path, sessions_path, hours_path, records_path],
        args.append,
    )
    existing = _read_existing_metadata(metadata_path) if args.append else pd.DataFrame()

    planned = _plan_tasks(args, existing, weather_key)
    print(
        f"Planning {len(planned)} session profile(s) into {output_dir} "
        f"at reference charger {config.CAPACITY_HOME_CHARGING} kW."
    )
    if args.dry_run or not planned:
        return

    df_weather = _load_or_fetch_weather(weather_csv)
    weather_records = df_weather[["temp_air", "pressure", "dew_point", "relative_humidity"]].to_dict(orient="list")

    def write(result) -> None:
        metadata_row, ledger_row, sessions_out, hours_out, records_out = result
        # Metadata is written LAST and is the completion marker: --append plans
        # missing work from it, so a profile must never appear there before all
        # of its other artifacts are durable. An interruption then leaves the
        # profile absent from metadata and it is simply regenerated.
        _append_csv(ledger_row, ledger_path)
        _append_csv(sessions_out, sessions_path)
        _append_csv(hours_out, hours_path)
        _append_source_records(records_out, records_path)
        _append_csv(metadata_row, metadata_path)

    if args.n_cpu == 1:
        for position, task in enumerate(planned, start=1):
            print(f"[{position}/{len(planned)}] Generating {task['profile_id']}", flush=True)
            write(_generate_session_profile(task, weather_records))
        return

    # emobpy retains memory across runs, so workers are recycled and only a
    # bounded number of results is ever in flight. Submitting all profiles at
    # once exhausted this machine's memory and pushed it into swap.
    in_flight = max(args.n_cpu * 2, args.n_cpu + 1)
    completed = 0
    with ProcessPoolExecutor(
        max_workers=args.n_cpu,
        max_tasks_per_child=8,
        initializer=_init_session_worker,
        initargs=(str(Path(args.scenario_config).resolve()),),
    ) as executor:
        pending = list(planned)
        futures = {}
        while pending or futures:
            while pending and len(futures) < in_flight:
                task = pending.pop(0)
                futures[executor.submit(_generate_session_profile, task, weather_records)] = task
            done = next(as_completed(futures))
            task = futures.pop(done)
            completed += 1
            print(
                f"[{completed}/{len(planned)}] Finished {task['profile_id']}",
                flush=True,
            )
            write(done.result())


def _plan_tasks(args: argparse.Namespace, existing: pd.DataFrame, weather_key: str) -> list[dict]:
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
                planned.append(
                    {
                        "profile_id": _profile_id(weather_key, model_index, model, schedule, sample_index),
                        "model_index": model_index,
                        "model": model,
                        "schedule_index": schedule_index,
                        "schedule": schedule,
                        "sample_index": sample_index,
                        "pool_seed": _pool_seed(model_index, schedule_index, sample_index),
                        "weather_key": weather_key,
                    }
                )
    return planned


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pregenerated emobpy mobility profile CSV pool.")
    parser.add_argument(
        "--mode",
        choices=["deadline", "session"],
        default="session",
        help=(
            "'session' generates the conservation-preserving full-year reference pool; "
            "'deadline' reproduces the legacy clipped emobpy_pool_v1."
        ),
    )
    parser.add_argument(
        "--freeze-manifest",
        action="store_true",
        help=(
            "Validate pool completeness and write the pinned content manifest. "
            "Run this once the pool is complete; consumers require it."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Session-mode output directory. Must not be the published v1 pool.",
    )
    parser.add_argument("--profiles-per-stratum", type=int, default=1)
    parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIO_CONFIG)
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
    args = parser.parse_args(argv)
    if args.profiles_per_stratum < 1:
        raise ValueError("--profiles-per-stratum must be at least 1")
    if not 0 < args.market_share_threshold <= 1:
        raise ValueError("--market-share-threshold must be in (0, 1]")
    if args.n_cpu < 1:
        raise ValueError("--n_cpu must be at least 1")
    return args


def main(argv: list[str] | None = None) -> None:
    """Generate (or freeze) the mobility profile pool; see ``--help``."""
    arguments = parse_args(argv)
    scenario, _scenario_hash = load_scenario_config(arguments.scenario_config)
    config.apply_scenario(scenario)
    if arguments.mode == "session":
        output_dir = Path(
            arguments.output_dir
            or (Path(config.DATA_STAT_DIR) / "general" / SESSION_POOL_DIRNAME)
        )
        if not arguments.freeze_manifest:
            generate_session_pool(arguments)
        if arguments.freeze_manifest:
            path = write_pool_manifest(output_dir, scenario_hash=_scenario_hash)
            print(f"Pool manifest written to {path}")
    else:
        generate_pool(arguments)


if __name__ == "__main__":
    main()
