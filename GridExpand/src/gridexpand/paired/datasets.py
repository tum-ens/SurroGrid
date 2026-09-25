"""Resolve a prepared paired dataset from stable artifact conventions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import h5py
import pandas as pd

GRIDEXPAND_DIR = Path(__file__).resolve().parents[1]
GRIDALLOC_DIR = GRIDEXPAND_DIR / "2.demand_allocation" / "gridalloc"
PAIRED_DATASET_ROOT = (
    GRIDALLOC_DIR / "outputs" / "scenario_calibration"
)
HEAT_LIBRARY_ROOT = PAIRED_DATASET_ROOT / "profile_libraries"
WEATHER_RESULT_ROOT = GRIDALLOC_DIR / "results"


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
