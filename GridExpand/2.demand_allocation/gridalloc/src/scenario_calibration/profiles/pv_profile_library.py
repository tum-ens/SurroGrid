"""Build the shared pvlib profile library for LoD2 roof-angle bins."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import GRIDALLOC_DIR, GRIDEXPAND_DIR


DEFAULT_PROFILE_LIBRARY_NAME = "paired_pv_profile_library.h5"


def profile_label(tilt: float, azimuth: float) -> str:
    return f"solar_{float(tilt):g}_{float(azimuth):g}"


def required_profile_angles(
    roof_catalog: pd.DataFrame,
    allocation: pd.DataFrame,
) -> list[tuple[float, float]]:
    eligible = set(
        allocation.loc[
            allocation["pv_roof_eligible"].astype(bool), "building_objectid"
        ].astype(str)
    )
    selected = roof_catalog[
        roof_catalog["building_objectid"].astype(str).isin(eligible)
        & roof_catalog["profile_usable"].astype(bool)
    ].copy()
    return sorted(
        {
            (float(tilt), float(azimuth))
            for tilt, azimuth in selected[
                ["profile_tilt_deg", "profile_azimuth_deg"]
            ].itertuples(index=False, name=None)
        }
    )


def build_pv_profile_library(
    *,
    roof_catalog_path: Path,
    allocation_path: Path,
    weather_source_hdf: Path,
    output_path: Path,
) -> Path:
    """Create or reuse normalized annual profiles for all eligible angle bins."""
    roof_catalog = pd.read_csv(roof_catalog_path)
    allocation = pd.read_csv(allocation_path)
    angles = required_profile_angles(roof_catalog, allocation)
    if not angles:
        raise ValueError("The paired allocation contains no eligible PV roof profiles.")
    metadata = _expected_metadata(weather_source_hdf, angles)
    if output_path.exists() and _metadata_matches(output_path, metadata):
        return output_path

    weather, latitude, longitude, altitude = _load_weather_and_location(
        weather_source_hdf
    )
    solar = _load_solar_module()
    values = np.empty((len(weather), len(angles)), dtype=np.float32)
    labels = []
    for column, (tilt, azimuth) in enumerate(angles):
        generated = solar._calculate_pv_power(
            latitude,
            longitude,
            altitude,
            tilt,
            azimuth,
            weather,
        )
        values[:, column] = (
            pd.to_numeric(generated, errors="coerce").fillna(0.0).to_numpy()
            / 1000.0
        )
        labels.append(profile_label(tilt, azimuth))
    profiles = pd.DataFrame(values, columns=labels)
    profiles.index.name = "t"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.h5")
    if temporary.exists():
        temporary.unlink()
    profiles.to_hdf(
        temporary,
        key="profiles",
        mode="w",
        complevel=9,
        complib="blosc",
    )
    pd.Series({"json": json.dumps(metadata, sort_keys=True)}).to_hdf(
        temporary,
        key="metadata",
        mode="a",
    )
    temporary.replace(output_path)
    return output_path


def read_pv_profile_library(path: Path) -> pd.DataFrame:
    profiles = pd.read_hdf(path, key="profiles")
    profiles.index.name = "t"
    return profiles


def _expected_metadata(
    weather_source_hdf: Path,
    angles: list[tuple[float, float]],
) -> dict[str, object]:
    source = weather_source_hdf.resolve()
    stat = source.stat()
    angle_payload = json.dumps(angles, separators=(",", ":"))
    return {
        "weather_source_hdf": str(source),
        "weather_source_size": int(stat.st_size),
        "weather_source_mtime_ns": int(stat.st_mtime_ns),
        "angle_hash": hashlib.sha256(angle_payload.encode("utf-8")).hexdigest(),
        "profile_count": len(angles),
        "normalization": "pvlib_ac_kw_per_1kw_pdc0",
    }


def _metadata_matches(path: Path, expected: dict[str, object]) -> bool:
    try:
        stored = pd.read_hdf(path, key="metadata")
        actual = json.loads(str(stored.loc["json"]))
    except (KeyError, FileNotFoundError, ValueError, OSError):
        return False
    return actual == expected


def _load_solar_module():
    """Load the original Elias pvlib implementation with its local config."""
    return _load_gridalloc_function_module("solar")


def _load_weather_and_location(
    weather_source_hdf: Path,
) -> tuple[pd.DataFrame, float, float, float]:
    """Read cached raw weather or reproduce it for the source grid location."""
    try:
        weather = pd.read_hdf(weather_source_hdf, key="raw_data/weather")
        region = pd.read_hdf(weather_source_hdf, key="raw_data/region")
        if region.empty:
            raise ValueError(
                f"Weather source has an empty raw_data/region: {weather_source_hdf}"
            )
        row = region.iloc[0]
        return (
            weather,
            float(row["lat"]),
            float(row["lon"]),
            float(row.get("altitude", 0.0)),
        )
    except KeyError:
        pass

    if str(GRIDEXPAND_DIR) not in sys.path:
        sys.path.insert(0, str(GRIDEXPAND_DIR))
        remove_gridexpand_path = True
    else:
        remove_gridexpand_path = False
    try:
        from common.database import SurroGridDatabase

        database = SurroGridDatabase()
        grid_ref = database.resolve_grid_identifier(weather_source_hdf.name)
        region = database.read_region(grid_ref)
    finally:
        if remove_gridexpand_path:
            sys.path.remove(str(GRIDEXPAND_DIR))
    row = region.iloc[0]
    weather_module = _load_gridalloc_function_module("weather")
    weather_result = weather_module.get_pvgis_tmy_sarah3_dataframe(
        float(row["lat"]),
        float(row["lon"]),
    )
    if weather_result is None:
        raise RuntimeError(
            "PVGIS did not return TMY weather for the shared PV profile library."
        )
    weather, altitude, _ = weather_result
    return weather, float(row["lat"]), float(row["lon"]), float(altitude)


def _load_gridalloc_function_module(name: str):
    demand_dir = GRIDALLOC_DIR.resolve()
    old_cwd = Path.cwd()
    sys.path.insert(0, str(demand_dir))
    previous_config = sys.modules.pop("config", None)
    try:
        os.chdir(demand_dir)
        spec = importlib.util.spec_from_file_location(
            f"gridalloc_{name}",
            demand_dir / "src" / "functions" / f"{name}.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load gridalloc {name} module.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(demand_dir))
        except ValueError:
            pass
        if previous_config is not None:
            sys.modules["config"] = previous_config
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roof-catalog", type=Path, required=True)
    parser.add_argument("--allocation-plan", type=Path, required=True)
    parser.add_argument("--weather-source-hdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build_pv_profile_library(
        roof_catalog_path=args.roof_catalog.resolve(),
        allocation_path=args.allocation_plan.resolve(),
        weather_source_hdf=args.weather_source_hdf.resolve(),
        output_path=args.output.resolve(),
    )
    profiles = read_pv_profile_library(output)
    print(f"PV profile library ready: {output} ({len(profiles.columns)} profiles)")


if __name__ == "__main__":
    main()
