
"""Prepare one regional physical-building electrification assignment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from .scenario_calibration.paths import GRIDEXPAND_DIR

if str(GRIDEXPAND_DIR) not in sys.path:
    sys.path.insert(0, str(GRIDEXPAND_DIR))

from common.building_components import residential_component_mask  # noqa: E402
from common.database import SurroGridDatabase  # noqa: E402
from common.electrification import (  # noqa: E402
    assignment_manifest_hash,
    assignment_summary,
    build_electrification_assignment,
)
from config import config  # noqa: E402
from scenario_pipeline.config_loader import (  # noqa: E402
    load_scenario_config,
    scenario_identity_key,
)
from src.assets.pv.roof_catalog import (  # noqa: E402
    building_lod2_capacity,
    load_lod2_roof_catalog,
)
import src.functions.electricity as electricity  # noqa: E402
import src.functions.heat as heat  # noqa: E402
import src.functions.mobility as mobility  # noqa: E402
from scenario_pipeline.synthetic_ags_runner import get_candidates  # noqa: E402


def _candidate_identity(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: candidate.get(key)
        for key in (
            "candidate_index",
            "ags",
            "plz",
            "kcid",
            "bcid",
            "grid_result_id",
            "version_id",
            "bridge_filename",
            "n_buildings",
            "n_residential_buildings",
        )
    }


def _candidate_manifest_hash(candidates: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [_candidate_identity(candidate) for candidate in candidates],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selected_components(
    components: pd.DataFrame, demand_scope: str
) -> pd.DataFrame:
    included = components["included_in_lv"].astype(bool)
    if demand_scope == "all":
        selected = components.loc[included].copy()
    elif demand_scope == "residential":
        selected = components.loc[residential_component_mask(components)].copy()
    else:
        raise ValueError(f"Unknown demand_scope={demand_scope!r}.")
    if selected.empty:
        raise ValueError("Regional electrification preparation found no demand components.")
    return selected


def _prepare_grid_inventory(
    database: SurroGridDatabase,
    candidate: dict[str, Any],
    *,
    demand_scope: str,
    mobility_source: str,
    profile_seed: int,
) -> pd.DataFrame:
    physical, region, _ = database.read_step2_input_data(candidate)
    components = database.read_building_components(candidate, physical)
    selected_components = _selected_components(components, demand_scope)
    selected_components["objectid"] = selected_components["objectid"].astype(str)
    selected_components = electricity.sample_statistics(
        selected_components, base_seed=profile_seed
    )
    selected_components, _, _ = electricity.get_elec_demand(
        selected_components,
        base_seed=profile_seed,
        return_component_profiles=True,
    )

    if demand_scope == "residential":
        selected_ids = set(selected_components["objectid"].astype(str))
        physical = physical.loc[
            physical["objectid"].astype(str).isin(selected_ids)
        ].copy()
    else:
        physical = physical.copy()
    physical["building_objectid"] = physical["objectid"].astype(str)
    annual = selected_components.groupby("objectid")[
        "annual_electricity_kwh"
    ].sum()
    residential_area = (
        selected_components.loc[
            selected_components["component_category"].eq("Residential")
        ]
        .groupby("objectid")["effective_floor_area_m2"]
        .sum()
    )
    occupancy = (
        selected_components.loc[
            selected_components["component_category"].eq("Residential")
        ]
        .drop_duplicates("objectid")
        .set_index("objectid")["occ_list"]
    )
    physical["annual_electricity_kwh"] = (
        physical["building_objectid"].map(annual).fillna(0.0)
    )
    physical["residential_effective_floor_area_m2"] = (
        physical["building_objectid"].map(residential_area).fillna(0.0)
    )
    physical["occ_list"] = physical["building_objectid"].map(occupancy)
    physical["occ_list"] = physical["occ_list"].apply(
        lambda value: value if isinstance(value, (list, tuple)) else []
    )
    allowed_models = (
        mobility.get_pool_supported_models()
        if mobility_source == "pool"
        else None
    )
    owned = mobility.sample_statistics(
        physical,
        region,
        allowed_models=allowed_models,
        base_seed=profile_seed,
    )
    owned["candidate_index"] = int(candidate["candidate_index"])
    return owned


def _resolve_heat_profile_eligibility(
    physical: pd.DataFrame,
    database: SurroGridDatabase,
    source: str,
) -> set[str]:
    """Resolve valid source coverage before selecting heat adopters."""
    residential = pd.to_numeric(
        physical["residential_effective_floor_area_m2"], errors="coerce"
    ).fillna(0.0).gt(0.0)
    residential_ids = set(physical.loc[residential, "building_objectid"].astype(str))
    if source == "teaser":
        return residential_ids
    if source != "infdb_ro_heat":
        raise ValueError(f"Unknown space heat source {source!r}.")
    if not residential_ids:
        return set()

    heat_buildings = physical.loc[residential].copy()
    gross_area = (
        pd.to_numeric(heat_buildings["floor_area"], errors="coerce")
        * pd.to_numeric(heat_buildings["floor_number"], errors="coerce")
    )
    heat_buildings["residential_area_share"] = (
        pd.to_numeric(
            heat_buildings["residential_effective_floor_area_m2"],
            errors="coerce",
        )
        / gross_area
    )
    heat.load_space_heat(heat_buildings, engine=database.engine)
    return residential_ids


def _build_inventory(
    candidates: list[dict[str, Any]],
    *,
    scenario,
    pylovo_version_id: str,
    demand_scope: str,
    mobility_source: str,
    profile_seed: int,
    source_evidence_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    database = SurroGridDatabase()
    database.pylovo_version_id = str(pylovo_version_id)
    frames = [
        _prepare_grid_inventory(
            database,
            candidate,
            demand_scope=demand_scope,
            mobility_source=mobility_source,
            profile_seed=profile_seed,
        )
        for candidate in candidates
    ]
    physical = pd.concat(frames, ignore_index=True, sort=False)
    physical["building_objectid"] = physical["building_objectid"].astype(str)
    if physical["building_objectid"].duplicated().any():
        duplicated = sorted(
            physical.loc[
                physical["building_objectid"].duplicated(keep=False),
                "building_objectid",
            ].unique()
        )
        raise ValueError(
            "Regional preparation found physical buildings in multiple candidate "
            f"grids: {duplicated[:10]}"
        )
    residential = pd.to_numeric(
        physical["residential_effective_floor_area_m2"], errors="coerce"
    ).fillna(0.0).gt(0.0)
    heat_eligible_ids = _resolve_heat_profile_eligibility(
        physical,
        database,
        scenario.heat.space_heat_source,
    )
    physical["heat_eligible"] = physical["building_objectid"].isin(
        heat_eligible_ids
    )
    physical["heat_exclusion_reason"] = np.select(
        [~residential, ~physical["heat_eligible"]],
        ["no_residential_component", "no_valid_heat_profile_source"],
        default=None,
    )
    household = physical["occ_list"].map(
        lambda value: isinstance(value, (list, tuple)) and len(value) > 0
    )
    vehicles = pd.to_numeric(
        physical["n_cars_tot"], errors="coerce"
    ).fillna(0.0)
    physical["mobility_eligible"] = (
        residential & household & vehicles.gt(0.0)
    )
    physical["mobility_exclusion_reason"] = np.select(
        [~residential, ~household, vehicles.le(0.0)],
        ["no_residential_component", "no_household", "no_vehicle_inventory"],
        default=None,
    )

    roof_options = {
        "tilt_bin_deg": scenario.pv.tilt_bin_degrees,
        "azimuth_bin_deg": scenario.pv.azimuth_bin_degrees,
        "module_capacity_kw_per_m2": scenario.pv.module_capacity_kw_per_m2,
        "flat_roof_utilization": scenario.pv.flat_roof_utilization,
        "slanted_roof_utilization": scenario.pv.slanted_roof_utilization,
        "fallback_capacity_kw": scenario.pv.fallback_capacity_kwp,
    }
    roofs = load_lod2_roof_catalog(
        database.engine,
        physical["building_objectid"],
        **roof_options,
    )
    roof_capacity = building_lod2_capacity(roofs)
    physical["pv_battery_eligible"] = (
        physical["building_objectid"].map(roof_capacity).fillna(0.0).gt(0.0)
        & pd.to_numeric(physical["annual_electricity_kwh"], errors="coerce")
        .fillna(0.0)
        .gt(0.0)
    )
    physical["pv_battery_exclusion_reason"] = "no_usable_lod2_roof"
    physical.loc[
        physical["pv_battery_eligible"], "pv_battery_exclusion_reason"
    ] = None
    physical.loc[
        ~physical["pv_battery_eligible"]
        & pd.to_numeric(physical["annual_electricity_kwh"], errors="coerce")
        .fillna(0.0)
        .le(0.0),
        "pv_battery_exclusion_reason",
    ] = "no_base_electricity"

    if source_evidence_path is not None:
        evidence = pd.read_csv(source_evidence_path)
        if "building_objectid" not in evidence:
            raise ValueError("Source evidence must contain building_objectid.")
        evidence["building_objectid"] = evidence["building_objectid"].astype(str)
        if evidence["building_objectid"].duplicated().any():
            raise ValueError("Source evidence must have one row per building.")
        physical = physical.merge(
            evidence,
            on="building_objectid",
            how="left",
            suffixes=("", "_source"),
            validate="one_to_one",
        )

    return physical, roofs


def prepare_regional_electrification_assignment(
    *,
    candidates: list[dict[str, Any]],
    scenario_config_path: Path,
    pylovo_version_id: str,
    demand_scope: str,
    mobility_source: str,
    profile_seed: int,
    output_path: Path,
    source_evidence_path: Path | None = None,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("Regional electrification preparation requires candidates.")
    scenario, scenario_hash = load_scenario_config(scenario_config_path)
    config.apply_scenario(scenario)
    inventory, roofs = _build_inventory(
        candidates,
        scenario=scenario,
        pylovo_version_id=pylovo_version_id,
        demand_scope=demand_scope,
        mobility_source=mobility_source,
        profile_seed=profile_seed,
        source_evidence_path=source_evidence_path,
    )
    candidate_hash = _candidate_manifest_hash(candidates)
    selection_scope_id = (
        f"{scenario.scenario_id}|ags={candidates[0]['ags']}|"
        f"plz={','.join(sorted({str(candidate['plz']) for candidate in candidates}))}|"
        f"version={pylovo_version_id}|scope={demand_scope}|candidates={candidate_hash[:12]}"
    )
    assignment = build_electrification_assignment(
        inventory,
        scenario.electrification,
        selection_scope_id=selection_scope_id,
        profile_seed=profile_seed,
        source_evidence_columns={
            technology: f"{technology}_source_evidence"
            for technology in ("heat", "mobility", "pv_battery")
        },
    )
    manifest_hash = assignment_manifest_hash(assignment)
    summary = assignment_summary(assignment)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(output_path, index=False)
    summary.to_csv(
        output_path.with_name("electrification_assignment_summary.csv"),
        index=False,
    )
    candidate_path = output_path.with_name("electrification_candidate_grids.json")
    candidate_path.write_text(
        json.dumps(
            [_candidate_identity(candidate) for candidate in candidates],
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    metadata = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario_hash,
        "scenario_key": scenario_identity_key(scenario.scenario_id, scenario_hash),
        "selection_scope_id": selection_scope_id,
        "profile_seed": int(profile_seed),
        "pylovo_version_id": str(pylovo_version_id),
        "demand_scope": demand_scope,
        "mobility_source": mobility_source,
        "candidate_grid_manifest_hash": candidate_hash,
        "candidate_grid_count": len(candidates),
        "physical_building_count": int(inventory["building_objectid"].nunique()),
        "roof_section_count": int(len(roofs)),
        "assignment_hash": manifest_hash,
        "assignment_summary": summary.to_dict("records"),
        "candidate_grid_manifest": [
            _candidate_identity(candidate) for candidate in candidates
        ],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ags", required=True)
    parser.add_argument("--plz", type=int)
    parser.add_argument("--min-buildings", type=int, default=5)
    parser.add_argument("--pylovo-version-id", required=True)
    parser.add_argument("--demand-scope", choices=["all", "residential"], default="all")
    parser.add_argument("--mobility-source", choices=["emobpy", "pool"], default="pool")
    parser.add_argument("--profile-seed", type=int, default=481527)
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path)
    args = parser.parse_args()
    candidates = get_candidates(
        GRIDEXPAND_DIR.parent,
        args.ags,
        args.min_buildings,
        args.demand_scope,
        args.pylovo_version_id,
    )
    if args.plz is not None:
        candidates = [
            candidate for candidate in candidates if int(candidate["plz"]) == args.plz
        ]
    metadata = prepare_regional_electrification_assignment(
        candidates=candidates,
        scenario_config_path=args.scenario_config.resolve(),
        pylovo_version_id=str(args.pylovo_version_id),
        demand_scope=args.demand_scope,
        mobility_source=args.mobility_source,
        profile_seed=args.profile_seed,
        output_path=args.output,
        source_evidence_path=(
            args.source_evidence.resolve()
            if args.source_evidence is not None
            else None
        ),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
