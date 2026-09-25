"""Resolve a prepared paired dataset from stable artifact conventions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import h5py
import pandas as pd

from gridexpand.paths import ALLOCATION_RESULTS_DIR, SCENARIO_CALIBRATION_OUTPUT_DIR

PAIRED_DATASET_ROOT = SCENARIO_CALIBRATION_OUTPUT_DIR
HEAT_LIBRARY_ROOT = PAIRED_DATASET_ROOT / "profile_libraries"
WEATHER_RESULT_ROOT = ALLOCATION_RESULTS_DIR


@dataclass(frozen=True)
class PairedDataset:
    dataset_id: str
    paired_dir: Path
    plz: int
    pylovo_version_id: str
    weather_source_hdf: Path
    heat_profile_library: Path


def _one_profile_set(catalog_path: Path) -> str:
    catalog = pd.read_csv(catalog_path)
    physical = catalog.get("profile_source_kind", pd.Series(dtype=str)).astype(str)
    values = (
        catalog.loc[physical.eq("physical_heat_library"), "profile_set_id"]
        .dropna()
        .astype(str)
        .unique()
    )
    if len(values) != 1:
        raise ValueError(
            f"Expected one physical heat profile_set_id in {catalog_path}, "
            f"found {values.tolist()}."
        )
    return str(values[0])


def _weather_source(pv_library: Path) -> Path:
    try:
        stored = pd.read_hdf(pv_library, key="metadata")
        metadata = json.loads(str(stored.loc["json"]))
        source_name = Path(str(metadata["weather_source_hdf"])).name
    except (KeyError, ValueError, OSError) as exc:
        raise ValueError(
            f"Cannot resolve weather source from PV library metadata: {pv_library}"
        ) from exc
    local_source = WEATHER_RESULT_ROOT / source_name
    if not local_source.exists():
        raise FileNotFoundError(
            f"Resolved paired weather source does not exist: {local_source}"
        )
    return local_source


def resolve_paired_dataset(
    dataset_id: str,
    *,
    expected_pylovo_version_id: str | None = None,
) -> PairedDataset:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", dataset_id) is None:
        raise ValueError("paired dataset ID must be one directory-safe name.")
    paired_dir = PAIRED_DATASET_ROOT / dataset_id
    metadata_path = paired_dir / "paired_scenario_metadata.json"
    pv_library = paired_dir / "paired_pv_profile_library.h5"
    heat_catalog = paired_dir / "paired_heat_profile_catalog.csv"
    for required in (metadata_path, pv_library, heat_catalog):
        if not required.exists():
            raise FileNotFoundError(required)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    plz = int(metadata["plz"])
    pylovo_version_id = str(metadata.get("pylovo_version_id", "")).strip()
    if not pylovo_version_id:
        raise ValueError(
            f"Missing pylovo_version_id in paired metadata: {metadata_path}"
        )
    if (
        expected_pylovo_version_id is not None
        and str(expected_pylovo_version_id) != pylovo_version_id
    ):
        raise ValueError(
            "Run pylovo version does not match paired dataset metadata: "
            f"{expected_pylovo_version_id!r} != {pylovo_version_id!r}."
        )
    heat_profile_set = _one_profile_set(heat_catalog)
    heat_library = HEAT_LIBRARY_ROOT / f"{heat_profile_set}.h5"
    if not heat_library.exists():
        raise FileNotFoundError(heat_library)
    with h5py.File(heat_library, "r") as store:
        actual_set = str(store.attrs.get("profile_set_id", ""))
    if actual_set != heat_profile_set:
        raise ValueError(
            f"Heat library profile_set_id mismatch: expected {heat_profile_set!r}, "
            f"got {actual_set!r}."
        )
    return PairedDataset(
        dataset_id=dataset_id,
        paired_dir=paired_dir,
        plz=plz,
        pylovo_version_id=pylovo_version_id,
        weather_source_hdf=_weather_source(pv_library),
        heat_profile_library=heat_library,
    )


def validate_prepared_dataset(
    dataset_id: str,
    *,
    pylovo_version_id: str,
    scenario_hash: str,
    ags: int | None = None,
    plz: int | None = None,
    provider: str | None = None,
    electrification=None,
    profile_seed: int | None = None,
    require_publication_ready: bool = False,
    require_exact_heat: bool = False,
    weather_hdf: Path | None = None,
) -> dict[str, object]:
    """Check that a prepared paired dataset belongs to this run; return a readiness summary.

    Always: the pylovo version and scenario hash of the metadata, the
    electrification assignment against its metadata hash, identical building
    sets of the real and synthetic plans. Optional checks (each pipeline asks
    for its own): ``ags``/``plz``/``provider`` of the metadata, the assignment
    against the scenario's electrification rules (``electrification`` with
    ``profile_seed``), every heat profile publication-ready, every heat
    profile an exact physical-library profile, the PV library's weather file.

    Raises:
        ValueError, FileNotFoundError: the dataset does not match.
    """
    from gridexpand.common.electrification import (
        assignment_manifest_hash,
        validate_electrification_assignment_config,
    )

    dataset = resolve_paired_dataset(dataset_id, expected_pylovo_version_id=pylovo_version_id)
    paired_dir = dataset.paired_dir
    metadata = json.loads((paired_dir / "paired_scenario_metadata.json").read_text(encoding="utf-8"))
    if ags is not None and int(metadata.get("ags", -1)) != int(ags):
        raise ValueError("Prepared paired dataset AGS does not match the run YAML.")
    if plz is not None and int(metadata["plz"]) != int(plz):
        raise ValueError("Prepared paired dataset PLZ does not match the run YAML.")
    if provider is not None and metadata.get("provider") != provider:
        raise ValueError(f"{paired_dir} belongs to provider {metadata.get('provider')!r}.")
    if metadata.get("scenario_hash") != scenario_hash:
        raise ValueError(f"{paired_dir} was prepared with another scenario YAML (scenario hash differs).")
    real = pd.read_csv(paired_dir / "paired_real_bus_allocation_plan.csv")
    synthetic = pd.read_csv(paired_dir / "paired_synthetic_bus_allocation_plan.csv")
    heat = pd.read_csv(paired_dir / "paired_heat_profile_catalog.csv")
    assignment_path = paired_dir / "paired_electrification_assignment.csv"
    if not assignment_path.exists():
        raise ValueError(f"Missing paired electrification assignment: {assignment_path}")
    assignment = pd.read_csv(assignment_path)
    if electrification is not None:
        validate_electrification_assignment_config(assignment, electrification, profile_seed=profile_seed)
    assignment_hash = assignment_manifest_hash(assignment)
    if metadata.get("electrification_assignment_hash") != assignment_hash:
        raise ValueError("Prepared paired electrification assignment hash does not match metadata.")
    if require_publication_ready:
        not_ready = int((~heat["publication_ready"].astype(bool)).sum())
        if not_ready:
            raise ValueError(f"Paired heat readiness failed for {not_ready} buildings.")
    if require_exact_heat:
        not_exact = heat["profile_method"].ne("exact_physical_building") | heat["profile_source_kind"].ne(
            "physical_heat_library"
        )
        if not_exact.any():
            raise ValueError(f"{int(not_exact.sum())} heat profiles are not exact library profiles.")
    real_buildings = set(real["building_objectid"].astype(str))
    if real_buildings != set(synthetic["building_objectid"].astype(str)):
        raise ValueError("Prepared real and synthetic plans contain different buildings.")
    if weather_hdf is not None and dataset.weather_source_hdf.resolve() != Path(weather_hdf).resolve():
        raise ValueError(f"PV library was built from {dataset.weather_source_hdf}, not {weather_hdf}.")
    summary: dict[str, object] = {
        "paired_dataset_id": dataset.dataset_id,
        "paired_buildings": len(real_buildings),
        "real_grids": int(real["target_grid_id"].nunique()),
        "synthetic_grids": int(synthetic["target_grid_id"].nunique()),
        "heat_profiles": int(heat["building_objectid"].nunique()),
        "electrification_assignment_hash": assignment_hash,
        "electrification_selected": assignment.groupby("technology")["selected"].sum().astype(int).to_dict(),
    }
    if provider is not None:
        summary["provider"] = provider
    for key in ("registered_pylovo_grid_cases", "registered_pylovo_buildings"):
        if key in metadata:
            summary[key] = int(metadata[key])
    return summary
