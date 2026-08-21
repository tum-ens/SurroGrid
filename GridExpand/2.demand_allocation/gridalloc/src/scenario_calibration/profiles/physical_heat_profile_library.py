"""Network-independent physical-building heat-profile storage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..paths import SYNTHETIC_INPUT_DIR

SCHEMA_VERSION = 1
PROFILE_DATASETS = (
    "space_heat",
    "water_heat",
    "heatpump_air_cop",
)


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


class PhysicalHeatProfileLibrary:
    """Read exact physical heat profiles by stable building identifier."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(
                f"Physical heat-profile library not found: {self.path}"
            )
        with h5py.File(self.path, "r") as store:
            version = int(store.attrs.get("schema_version", 0))
            if version != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported heat-profile library schema {version} in {self.path}."
                )
            identifiers = store["building_objectid"].asstr()[:]
            methods = (
                store["profile_method"].asstr()[:]
                if "profile_method" in store
                else np.full(len(identifiers), "exact_physical_building")
            )
            source_buildings = (
                store["profile_source_building_objectid"].asstr()[:]
                if "profile_source_building_objectid" in store
                else identifiers
            )
            scales = (
                np.asarray(store["profile_scale"][:], dtype=float)
                if "profile_scale" in store
                else np.ones(len(identifiers))
            )
            fallback_scopes = (
                store["fallback_match_scope"].asstr()[:]
                if "fallback_match_scope" in store
                else np.full(len(identifiers), "")
            )
            self.profile_set_id = str(store.attrs["profile_set_id"])
            self.hours = int(store.attrs["hours"])
        self._row_by_building = {
            str(identifier): row for row, identifier in enumerate(identifiers)
        }
        self._metadata_by_building = {
            str(identifier): {
                "profile_method": str(methods[row]),
                "profile_source_building_objectid": str(source_buildings[row]),
                "profile_scale": float(scales[row]),
                "fallback_match_scope": str(fallback_scopes[row]),
            }
            for row, identifier in enumerate(identifiers)
        }

    def __contains__(self, building_objectid: object) -> bool:
        return str(building_objectid) in self._row_by_building

    @property
    def building_count(self) -> int:
        return len(self._row_by_building)

    def profile_metadata(self, building_objectid: object) -> dict[str, object]:
        """Return persisted provenance for one library entry."""
        building_id = str(building_objectid)
        try:
            return dict(self._metadata_by_building[building_id])
        except KeyError as exc:
            raise KeyError(
                f"Building {building_id!r} is absent from {self.path}."
            ) from exc

    def read(
        self,
        building_objectid: object,
        *,
        hours: int | None = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Return space heat, water heat, and heat-pump COP."""
        building_id = str(building_objectid)
        try:
            row = self._row_by_building[building_id]
        except KeyError as exc:
            raise KeyError(
                f"Building {building_id!r} is absent from {self.path}."
            ) from exc
        requested_hours = self.hours if hours is None else int(hours)
        if requested_hours > self.hours:
            raise ValueError(
                f"Requested {requested_hours} hours from a {self.hours}-hour "
                f"heat-profile library."
            )
        with h5py.File(self.path, "r") as store:
            values = [
                pd.Series(
                    np.asarray(store[name][row, :requested_hours], dtype=float)
                )
                for name in PROFILE_DATASETS
            ]
        values[2] = values[2].fillna(1.0).clip(lower=0.1)
        return values[0].fillna(0.0), values[1].fillna(0.0), values[2]


def create_physical_heat_profile_library(
    *,
    source_catalog: Path,
    source_hdf_dir: Path,
    output: Path,
    profile_set_id: str,
    source_mode: str = "profile",
    include_approved_fallbacks: bool = False,
) -> Path:
    """Extract physical or explicitly approved profiles into one library."""
    source_columns = {
        "profile": ("profile_source_hdf", "profile_source_bus"),
        "exact": ("exact_source_hdf", "exact_source_bus"),
    }
    try:
        source_hdf_column, source_bus_column = source_columns[source_mode]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported source mode {source_mode!r}; choose from "
            f"{sorted(source_columns)}."
        ) from exc
    catalog = pd.read_csv(source_catalog)
    required = {
        "building_objectid",
        "publication_ready",
        source_hdf_column,
        source_bus_column,
    }
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"Source heat catalog misses columns: {sorted(missing)}")
    selected = catalog["publication_ready"].astype(bool)
    if include_approved_fallbacks:
        if source_mode != "profile":
            raise ValueError(
                "Approved fallbacks can only be packaged with --source-mode profile."
            )
        selected |= catalog.get(
            "profile_method",
            pd.Series("", index=catalog.index),
        ).astype(str).eq("nearest_floor_area_scaled_approved")
    catalog = catalog[selected].copy()
    catalog = catalog.dropna(subset=[source_hdf_column, source_bus_column])
    catalog["building_objectid"] = catalog["building_objectid"].astype(str)
    duplicates = catalog["building_objectid"].duplicated(keep=False)
    if duplicates.any():
        examples = catalog.loc[duplicates, "building_objectid"].head(10).tolist()
        raise ValueError(f"Duplicate physical heat profiles in catalog: {examples}")
    catalog = catalog.sort_values("building_objectid").reset_index(drop=True)
    if catalog.empty:
        raise ValueError("Source heat catalog contains no publication-ready profiles.")

    first = catalog.iloc[0]
    first_hdf = source_hdf_dir / str(first[source_hdf_column])
    first_demand = pd.read_hdf(first_hdf, key="urbs_in/demand")
    hours = len(first_demand)
    if hours <= 0:
        raise ValueError(f"Empty heat-profile source: {first_hdf}")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    chunks = (1, min(hours, 8760))

    with h5py.File(temporary, "w") as store:
        store.attrs["schema_version"] = SCHEMA_VERSION
        store.attrs["profile_set_id"] = str(profile_set_id)
        store.attrs["hours"] = hours
        store.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        store.attrs["source_catalog"] = str(source_catalog.resolve())
        store.attrs["source_mode"] = source_mode
        fallback_mask = catalog.get(
            "profile_method",
            pd.Series("", index=catalog.index),
        ).astype(str).eq("nearest_floor_area_scaled_approved")
        store.attrs["approved_fallback_buildings"] = int(fallback_mask.sum())
        store.create_dataset(
            "building_objectid",
            data=catalog["building_objectid"].to_numpy(dtype=object),
            dtype=string_dtype,
        )
        store.create_dataset(
            "building_use",
            data=catalog.get(
                "building_use",
                pd.Series("", index=catalog.index),
            ).map(_text).to_numpy(dtype=object),
            dtype=string_dtype,
        )
        store.create_dataset(
            "source_hdf",
            data=catalog[source_hdf_column].map(_text).to_numpy(dtype=object),
            dtype=string_dtype,
        )
        store.create_dataset(
            "source_bus",
            data=pd.to_numeric(
                catalog[source_bus_column],
                errors="raise",
            ).astype(int),
        )
        store.create_dataset(
            "profile_method",
            data=catalog.get(
                "profile_method",
                pd.Series("exact_physical_building", index=catalog.index),
            ).map(_text).to_numpy(dtype=object),
            dtype=string_dtype,
        )
        store.create_dataset(
            "profile_source_building_objectid",
            data=catalog.get(
                "profile_source_building_objectid",
                catalog["building_objectid"],
            ).map(_text).to_numpy(dtype=object),
            dtype=string_dtype,
        )
        store.create_dataset(
            "profile_scale",
            data=pd.to_numeric(
                catalog.get("profile_scale", pd.Series(1.0, index=catalog.index)),
                errors="raise",
            ).astype(float),
        )
        store.create_dataset(
            "fallback_match_scope",
            data=catalog.get(
                "fallback_match_scope",
                pd.Series("", index=catalog.index),
            ).map(_text).to_numpy(dtype=object),
            dtype=string_dtype,
        )
        datasets = {
            name: store.create_dataset(
                name,
                shape=(len(catalog), hours),
                dtype="f4",
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            for name in PROFILE_DATASETS
        }

        for source_name, rows in catalog.groupby(
            source_hdf_column,
            sort=True,
            observed=True,
        ):
            source_path = source_hdf_dir / str(source_name)
            demand = pd.read_hdf(source_path, key="urbs_in/demand")
            efficiency = pd.read_hdf(source_path, key="urbs_in/eff_factor")
            if len(demand) != hours or len(efficiency) != hours:
                raise ValueError(
                    f"Inconsistent profile length in {source_path}: "
                    f"demand={len(demand)}, efficiency={len(efficiency)}, "
                    f"expected={hours}."
                )
            for index, row in rows.iterrows():
                bus = int(row[source_bus_column])
                profile_scale = float(row.get("profile_scale", 1.0))
                if not np.isfinite(profile_scale) or profile_scale <= 0.0:
                    raise ValueError(
                        f"Invalid heat-profile scale for {row['building_objectid']}: "
                        f"{profile_scale}"
                    )
                columns = {
                    "space_heat": (bus, "space_heat"),
                    "water_heat": (bus, "water_heat"),
                    "heatpump_air_cop": (bus, "heatpump_air"),
                }
                for name, column in columns.items():
                    source = demand if name != "heatpump_air_cop" else efficiency
                    if column not in source:
                        raise ValueError(
                            f"Missing {column} for building "
                            f"{row['building_objectid']} in {source_path}."
                        )
                    values = pd.to_numeric(source[column], errors="coerce")
                    fill_value = 1.0 if name == "heatpump_air_cop" else 0.0
                    values = values.fillna(fill_value).to_numpy(dtype=np.float32)
                    if name == "heatpump_air_cop":
                        values = np.maximum(values, np.float32(0.1))
                    else:
                        values *= np.float32(profile_scale)
                    datasets[name][int(index), :] = values

    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument(
        "--source-hdf-dir",
        type=Path,
        default=SYNTHETIC_INPUT_DIR,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-set-id", required=True)
    parser.add_argument(
        "--source-mode",
        choices=("profile", "exact"),
        default="profile",
        help=(
            "Read legacy validated profile_source_* mappings or current "
            "exact_source_* mappings from the catalog."
        ),
    )
    parser.add_argument(
        "--include-approved-fallbacks",
        action="store_true",
        help=(
            "Package approved nearest-building profiles under their target "
            "building IDs, scaling heat demand by floor area."
        ),
    )
    args = parser.parse_args()
    output = create_physical_heat_profile_library(
        source_catalog=args.source_catalog.resolve(),
        source_hdf_dir=args.source_hdf_dir.resolve(),
        output=args.output,
        profile_set_id=args.profile_set_id,
        source_mode=args.source_mode,
        include_approved_fallbacks=args.include_approved_fallbacks,
    )
    library = PhysicalHeatProfileLibrary(output)
    print(
        f"{output}\n"
        f"profile_set_id={library.profile_set_id}\n"
        f"buildings={library.building_count}\n"
        f"hours={library.hours}"
    )


if __name__ == "__main__":
    main()
