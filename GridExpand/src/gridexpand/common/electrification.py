"""Auditable, topology-independent building electrification assignments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .reproducibility import stable_seed


TECHNOLOGIES = ("heat", "mobility", "pv_battery")
ASSIGNMENT_COLUMNS = (
    "building_objectid",
    "technology",
    "selection_scope_id",
    "adoption_mode",
    "configured_share",
    "eligible",
    "selection_score",
    "selection_rank",
    "selected",
    "exclusion_reason",
    "source_evidence",
    "profile_seed",
)
# Written only when a technology selects per building type (building_share_by_type),
# so manifests without strata keep their columns and hashes.
STRATUM_COLUMN = "selection_stratum"
DEFAULT_STRATUM = "other"


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _present(value: Any) -> bool:
    if value is None or value is pd.NA:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    try:
        return bool(pd.notna(value))
    except (TypeError, ValueError):
        return True


def _canonical_evidence(value: Any) -> str | None:
    if not _present(value):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        return str(value)


def _source_value(
    row: pd.Series,
    technology: str,
    source_evidence_columns: Mapping[str, str],
    source_evidence: Mapping[str, Mapping[str, Any]] | None,
) -> Any:
    column = source_evidence_columns.get(technology, f"{technology}_source_evidence")
    if column in row:
        return row[column]
    if source_evidence is not None:
        return source_evidence.get(technology, {}).get(str(row["building_objectid"]))
    return None


def _share_by_type(config: Any) -> dict[str, float] | None:
    by_type = _config_value(config, "building_share_by_type")
    return {str(k): float(v) for k, v in by_type.items()} if by_type else None


def _stratum_of(config: Any, building_type: Any) -> str:
    by_type = _share_by_type(config) or {}
    return str(building_type) if str(building_type) in by_type else DEFAULT_STRATUM


def _stratum_share(config: Any, stratum: str) -> float | None:
    by_type = _share_by_type(config) or {}
    if stratum in by_type:
        return by_type[stratum]
    share = _config_value(config, "building_share")
    return None if share is None else float(share)


def _normalise_adoption_configs(adoption: Any) -> dict[str, Any]:
    if hasattr(adoption, "for_technology"):
        return {name: adoption.for_technology(name) for name in TECHNOLOGIES}
    if not isinstance(adoption, Mapping):
        raise TypeError(
            "Electrification adoption settings must be a mapping or ElectrificationConfig."
        )
    missing = sorted(set(TECHNOLOGIES).difference(adoption))
    if missing:
        raise ValueError(f"Electrification adoption settings are missing: {missing}")
    return {name: adoption[name] for name in TECHNOLOGIES}


def build_electrification_assignment(
    inventory: pd.DataFrame,
    adoption: Any,
    *,
    selection_scope_id: str,
    profile_seed: int,
    source_evidence_columns: Mapping[str, str] | None = None,
    source_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build one auditable assignment row per building and technology.

    inventory must contain one row per physical building and, for each
    technology, <technology>_eligible plus
    <technology>_exclusion_reason. Source-inventory modes additionally
    require a source-evidence column or explicit source_evidence mapping.
    """
    if not isinstance(inventory, pd.DataFrame):
        raise TypeError("Electrification inventory must be a pandas DataFrame.")
    if not str(selection_scope_id).strip():
        raise ValueError("selection_scope_id must be non-empty.")
    if isinstance(profile_seed, bool) or not isinstance(profile_seed, (int, np.integer)):
        raise ValueError("profile_seed must be an integer.")
    if "building_objectid" not in inventory:
        if "objectid" not in inventory:
            raise ValueError("Electrification inventory requires building_objectid/objectid.")
        inventory = inventory.rename(columns={"objectid": "building_objectid"})
    frame = inventory.copy()
    frame["building_objectid"] = frame["building_objectid"].astype("string").str.strip()
    if frame["building_objectid"].isna().any() or frame["building_objectid"].eq("").any():
        raise ValueError("Every electrification inventory row needs a non-empty building_objectid.")
    if frame["building_objectid"].duplicated().any():
        raise ValueError("Electrification inventory requires one row per physical building.")

    configs = _normalise_adoption_configs(adoption)
    stratified = any(
        _config_value(configs[t], "adoption_mode") == "deterministic_share" and _share_by_type(configs[t])
        for t in TECHNOLOGIES
    )
    if stratified and "building_type" not in frame:
        raise ValueError("building_share_by_type needs a building_type column in the inventory.")
    evidence_columns = dict(source_evidence_columns or {})
    output: list[dict[str, Any]] = []
    for technology in TECHNOLOGIES:
        eligible_column = f"{technology}_eligible"
        reason_column = f"{technology}_exclusion_reason"
        missing = [column for column in (eligible_column, reason_column) if column not in frame]
        if missing:
            raise ValueError(
                f"Electrification inventory is missing {technology} columns: {missing}"
            )
        mode = _config_value(configs[technology], "adoption_mode")
        if mode not in {"deterministic_share", "source_inventory"}:
            raise ValueError(f"Unknown adoption mode for {technology}: {mode!r}.")
        configured_share = _config_value(configs[technology], "building_share")
        if mode == "deterministic_share":
            if configured_share is None or isinstance(configured_share, bool):
                raise ValueError(
                    f"{technology}.building_share is required for deterministic_share."
                )
            configured_share = float(configured_share)
            if not np.isfinite(configured_share) or not 0.0 <= configured_share <= 1.0:
                raise ValueError(
                    f"{technology}.building_share must be finite and in [0, 1]."
                )
        elif configured_share is not None:
            raise ValueError(
                f"{technology}.building_share is not allowed for source_inventory."
            )

        records: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            building_id = str(row["building_objectid"])
            eligible = bool(row[eligible_column])
            reason = row[reason_column]
            if not eligible and not _present(reason):
                reason = "ineligible"
            evidence = _source_value(
                row, technology, evidence_columns, source_evidence
            )
            score = (
                stable_seed(
                    int(profile_seed), "electrification", technology, building_id
                )
                / 2**32
            )
            stratum = (
                _stratum_of(configs[technology], row["building_type"])
                if mode == "deterministic_share" and _share_by_type(configs[technology])
                else DEFAULT_STRATUM
            )
            records.append(
                {
                    "building_objectid": building_id,
                    "technology": technology,
                    "selection_scope_id": str(selection_scope_id),
                    "adoption_mode": mode,
                    "configured_share": (
                        _stratum_share(configs[technology], stratum)
                        if mode == "deterministic_share" else configured_share
                    ),
                    STRATUM_COLUMN: stratum,
                    "eligible": eligible,
                    "selection_score": (
                        float(score) if mode == "deterministic_share" else np.nan
                    ),
                    "selection_rank": np.nan,
                    "selected": False,
                    "exclusion_reason": None if eligible else str(reason),
                    "source_evidence": _canonical_evidence(evidence),
                    "profile_seed": int(profile_seed),
                }
            )

        if mode == "source_inventory":
            evidence_column = evidence_columns.get(
                technology, f"{technology}_source_evidence"
            )
            has_column = evidence_column in frame.columns
            has_mapping = source_evidence is not None and technology in source_evidence
            if not has_column and not has_mapping:
                raise ValueError(
                    f"source_inventory for {technology} requires source evidence "
                    f"column {evidence_column!r} or an explicit source_evidence mapping."
                )
            for record in records:
                if record["eligible"] and _present(record["source_evidence"]):
                    record["selected"] = True
                elif record["eligible"]:
                    record["exclusion_reason"] = "missing_source_evidence"
        else:
            # Rank and select within each stratum (one stratum without per-type shares).
            for stratum in sorted({record[STRATUM_COLUMN] for record in records}):
                ranked = sorted(
                    (
                        record for record in records
                        if record["eligible"] and record[STRATUM_COLUMN] == stratum
                    ),
                    key=lambda record: (
                        record["selection_score"],
                        record["building_objectid"],
                    ),
                )
                for rank, record in enumerate(ranked, start=1):
                    record["selection_rank"] = rank
                share = _stratum_share(configs[technology], stratum)
                selected_count = int(round(float(share) * len(ranked)))
                for record in ranked[:selected_count]:
                    record["selected"] = True
        output.extend(records)

    columns = [*ASSIGNMENT_COLUMNS, STRATUM_COLUMN] if stratified else list(ASSIGNMENT_COLUMNS)
    result = pd.DataFrame(output, columns=columns)
    result["eligible"] = result["eligible"].astype(bool)
    result["selected"] = result["selected"].astype(bool)
    result = result.sort_values(
        ["building_objectid", "technology"], kind="stable"
    ).reset_index(drop=True)
    validate_electrification_assignment(result)
    return result


def validate_electrification_assignment(
    assignment: pd.DataFrame, *, exact_share: bool = True
) -> None:
    """Validate uniqueness and the exact selection contract."""
    if not isinstance(assignment, pd.DataFrame):
        raise TypeError("Electrification assignment must be a pandas DataFrame.")
    missing = sorted(set(ASSIGNMENT_COLUMNS).difference(assignment.columns))
    if missing:
        raise ValueError(f"Electrification assignment is missing columns: {missing}")
    if assignment.empty:
        raise ValueError("Electrification assignment must not be empty.")
    if assignment.duplicated(["building_objectid", "technology"]).any():
        raise ValueError(
            "Electrification assignment must be unique per building and technology."
        )
    unknown = set(assignment["technology"].astype(str)).difference(TECHNOLOGIES)
    if unknown:
        raise ValueError(f"Unknown electrification technologies: {sorted(unknown)}")
    present_technologies = set(assignment["technology"].astype(str))
    if present_technologies != set(TECHNOLOGIES):
        raise ValueError(
            "Electrification assignment must contain all technologies: "
            f"{sorted(TECHNOLOGIES)}."
        )
    counts = assignment.groupby("building_objectid")["technology"].nunique()
    if counts.ne(len(TECHNOLOGIES)).any():
        raise ValueError(
            "Electrification assignment must contain one row per physical "
            "building and technology."
        )
    if assignment["selection_scope_id"].nunique(dropna=False) != 1:
        raise ValueError(
            "Electrification assignment must use one selection_scope_id."
        )
    if assignment["profile_seed"].nunique(dropna=False) != 1:
        raise ValueError("Electrification assignment must use one profile_seed.")
    if assignment.loc[assignment["selected"], "eligible"].eq(False).any():
        raise ValueError("Ineligible buildings cannot be selected.")
    for technology, group in assignment.groupby("technology", sort=False):
        modes = set(group["adoption_mode"].astype(str))
        if len(modes) != 1 or modes.isdisjoint({"deterministic_share", "source_inventory"}):
            raise ValueError(f"{technology} has an invalid or inconsistent adoption mode.")
        mode = next(iter(modes))
        if group["selection_scope_id"].nunique(dropna=False) != 1:
            raise ValueError(f"{technology} has inconsistent selection_scope_id values.")
        if group["profile_seed"].nunique(dropna=False) != 1:
            raise ValueError(f"{technology} has inconsistent profile_seed values.")
        if mode == "source_inventory":
            if group["configured_share"].notna().any():
                raise ValueError(
                    f"{technology}.configured_share must be null for source_inventory."
                )
            expected_selected = (
                group["eligible"].astype(bool)
                & group["source_evidence"].map(_present)
            )
            if not group["selected"].astype(bool).equals(expected_selected):
                raise ValueError(
                    f"source_inventory selection for {technology} must exactly match "
                    "eligible rows with source evidence."
                )
        else:
            if group["configured_share"].isna().any():
                raise ValueError(
                    f"{technology}.configured_share is required for deterministic_share."
                )
            shares = pd.to_numeric(group["configured_share"], errors="coerce")
            if shares.isna().any() or (~shares.between(0.0, 1.0)).any():
                raise ValueError(
                    f"{technology}.configured_share must be in [0, 1]."
                )
            if group.loc[group["eligible"], "selection_score"].isna().any():
                raise ValueError(
                    f"Eligible deterministic_share rows for {technology} need a selection score."
                )
            strata = (
                group[STRATUM_COLUMN].astype(str)
                if STRATUM_COLUMN in group.columns
                else pd.Series(DEFAULT_STRATUM, index=group.index)
            )
            for stratum, part in group.groupby(strata, sort=True):
                part_shares = shares.loc[part.index]
                if not np.allclose(
                    part_shares.to_numpy(dtype=float),
                    float(part_shares.iloc[0]),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"{technology}.configured_share must be consistent across rows "
                        f"of stratum {stratum!r}."
                    )
                if not exact_share:
                    continue
                ranked = part.loc[part["eligible"]].sort_values(
                    ["selection_score", "building_objectid"], kind="stable"
                )
                ranks = pd.to_numeric(ranked["selection_rank"], errors="coerce")
                expected_ranks = np.arange(1, len(ranked) + 1, dtype=float)
                if ranks.isna().any() or not np.array_equal(
                    ranks.to_numpy(dtype=float), expected_ranks
                ):
                    raise ValueError(
                        f"{technology}.selection_rank does not match deterministic ordering."
                    )
                share = float(part_shares.iloc[0])
                expected = int(round(share * int(part["eligible"].sum())))
                actual = int(part["selected"].sum())
                if actual != expected:
                    raise ValueError(
                        f"{technology} selected {actual} buildings; expected {expected}."
                    )
                expected_selected = set(
                    ranked.iloc[:expected]["building_objectid"].astype(str)
                )
                actual_selected = set(
                    part.loc[part["selected"], "building_objectid"].astype(str)
                )
                if actual_selected != expected_selected:
                    raise ValueError(
                        f"{technology}.selected rows do not match deterministic ranking."
                    )


def validate_electrification_assignment_config(
    assignment: pd.DataFrame,
    adoption: Any,
    *,
    profile_seed: int | None = None,
    exact_share: bool = True,
) -> None:
    """Validate a manifest against the active scenario adoption contract."""
    validate_electrification_assignment(assignment, exact_share=exact_share)
    configs = _normalise_adoption_configs(adoption)
    if profile_seed is not None and (
        assignment["profile_seed"].nunique(dropna=False) != 1
        or int(assignment["profile_seed"].iloc[0]) != int(profile_seed)
    ):
        raise ValueError("Electrification assignment profile_seed differs from the active run.")
    for technology in TECHNOLOGIES:
        group = assignment.loc[assignment["technology"].eq(technology)]
        expected_mode = _config_value(configs[technology], "adoption_mode")
        actual_modes = set(group["adoption_mode"].astype(str))
        if actual_modes != {expected_mode}:
            raise ValueError(
                f"Electrification assignment mode for {technology} does not match "
                f"the active scenario: expected {expected_mode!r}."
            )
        expected_share = _config_value(configs[technology], "building_share")
        actual_share = group["configured_share"]
        if expected_share is None:
            if actual_share.notna().any():
                raise ValueError(
                    f"Electrification assignment share for {technology} must be null."
                )
        else:
            actual = pd.to_numeric(actual_share, errors="coerce")
            strata = (
                group[STRATUM_COLUMN].astype(str)
                if STRATUM_COLUMN in group.columns
                else pd.Series(DEFAULT_STRATUM, index=group.index)
            )
            if not _share_by_type(configs[technology]) and strata.ne(DEFAULT_STRATUM).any():
                raise ValueError(
                    f"Electrification assignment for {technology} has strata the scenario does not define."
                )
            expected = strata.map(lambda stratum: _stratum_share(configs[technology], stratum))
            if actual.isna().any() or not np.allclose(
                actual.to_numpy(dtype=float), expected.to_numpy(dtype=float), rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"Electrification assignment share for {technology} does not "
                    "match the active scenario."
                )



def assignment_manifest_hash(
    assignment: pd.DataFrame, *, exact_share: bool = True
) -> str:
    """Return a stable hash independent of row order and pandas dtypes."""
    validate_electrification_assignment(assignment, exact_share=exact_share)
    columns = list(ASSIGNMENT_COLUMNS) + ([STRATUM_COLUMN] if STRATUM_COLUMN in assignment.columns else [])
    normalized = assignment[columns].copy()
    normalized = normalized.sort_values(
        ["building_objectid", "technology"], kind="stable"
    )
    records = []
    for record in normalized.to_dict("records"):
        clean = {}
        for key, value in record.items():
            if (
                value is None
                or value is pd.NA
                or (isinstance(value, float) and np.isnan(value))
            ):
                clean[key] = None
            elif key in {"eligible", "selected"}:
                clean[key] = bool(value)
            elif key == "selection_rank":
                clean[key] = int(value) if pd.notna(value) else None
            elif key == "profile_seed":
                clean[key] = int(value)
            elif key == "selection_score":
                clean[key] = round(float(value), 15) if pd.notna(value) else None
            elif key == "configured_share":
                clean[key] = float(value) if pd.notna(value) else None
            else:
                clean[key] = str(value)
        records.append(clean)
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assignment_summary(
    assignment: pd.DataFrame, *, exact_share: bool = True
) -> pd.DataFrame:
    """Return configured, eligible, selected, and realized counts per technology."""
    validate_electrification_assignment(assignment, exact_share=exact_share)
    rows = []
    for technology, group in assignment.groupby("technology", sort=True):
        selected = int(group["selected"].sum())
        eligible = int(group["eligible"].sum())
        configured = group["configured_share"].dropna().unique()
        # Several strata have several shares; the realized share is the reference then.
        configured_share = float(configured[0]) if len(configured) == 1 else np.nan
        rows.append(
            {
                "technology": technology,
                "selection_scope_id": str(group["selection_scope_id"].iloc[0]),
                "adoption_mode": str(group["adoption_mode"].iloc[0]),
                "configured_share": configured_share,
                "eligible_building_count": eligible,
                "selected_building_count": selected,
                "realized_building_share": (
                    selected / eligible if eligible else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def battery_buildings(assignment: pd.DataFrame, share: float) -> set[str]:
    """Selected PV buildings that also get a battery: a seeded share of the PV selection.

    The ranking uses the assignment's profile seed and its own stream ("battery"), so
    the battery choice is reproducible and independent of the PV ranking.
    """
    selected = assignment.loc[
        assignment["technology"].eq("pv_battery") & assignment["selected"].astype(bool),
        "building_objectid",
    ].astype(str).tolist()
    share = float(share)
    if not 0.0 <= share <= 1.0:
        raise ValueError("battery share must be in [0, 1].")
    if share >= 1.0 or not selected:
        return set(selected)
    seed = int(assignment["profile_seed"].iloc[0])
    ranked = sorted(
        selected,
        key=lambda building: (stable_seed(seed, "electrification", "battery", building), building),
    )
    return set(ranked[: int(round(share * len(ranked)))])
