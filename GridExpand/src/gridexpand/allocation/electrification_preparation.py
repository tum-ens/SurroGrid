
"""Prepare one regional physical-building electrification assignment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from gridexpand.common.building_components import residential_component_mask
from gridexpand.db.database import SurroGridDatabase
from gridexpand.common.electrification import (
    assignment_manifest_hash,
    assignment_summary,
    build_electrification_assignment,
)
from gridexpand.allocation.config import config
from gridexpand.scenario.config_loader import (
    load_scenario_config,
    scenario_identity_key,
)
from gridexpand.allocation.assets.pv.roof_catalog import (
    building_lod2_capacity,
    load_lod2_roof_catalog,
    roof_catalog_options,
)
from gridexpand.allocation.electrification import (
    INVENTORY_COLUMNS,
    check_heat_profile_source,
    electrification_inventory,
    file_sha256,
    has_household,
)
import gridexpand.allocation.functions.electricity as electricity
import gridexpand.allocation.functions.mobility as mobility


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
    selected_components, _, _ = electricity.profile_components(
        selected_components, base_seed=profile_seed
    )

    if demand_scope == "residential":
        selected_ids = set(selected_components["objectid"].astype(str))
        physical = physical.loc[
            physical["objectid"].astype(str).isin(selected_ids)
        ].copy()
    physical = electricity.aggregate_components_to_buildings(
        physical, selected_components, residential_area=True
    )
    physical["building_objectid"] = physical["objectid"].astype(str)
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
    check_heat_profile_source(
        scenario.heat.space_heat_source,
        physical.loc[residential],
        engine=database.engine,
    )

    roof_options = roof_catalog_options(scenario.pv)
    roofs = load_lod2_roof_catalog(
        database.engine,
        physical["building_objectid"],
        **roof_options,
    )
    # Exclusion reasons follow the roof-first order of the in-grid assignment
    # (B4: a building without roof and base electricity used to be labelled
    # no_base_electricity here and no_usable_lod2_roof in Grid).
    inventory = electrification_inventory(
        physical["building_objectid"],
        residential=residential,
        has_household=has_household(physical["occ_list"]),
        vehicle_count=physical["n_cars_tot"],
        roof_capacity_kw=physical["building_objectid"].map(building_lod2_capacity(roofs)),
        annual_electricity_kwh=physical["annual_electricity_kwh"],
    )
    for column in INVENTORY_COLUMNS:
        physical[column] = inventory[column]

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
    summary = assignment_summary(assignment)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(output_path, index=False)
    # Hash the CSV as consumers read it: empty strings become NaN on reload.
    manifest_hash = assignment_manifest_hash(pd.read_csv(output_path))
    # Step 2 verifies this digest instead of re-hashing every row per grid.
    file_hash = file_sha256(output_path)
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
        "assignment_file_sha256": file_hash,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ags", required=True)
    parser.add_argument("--plz", type=int, help="Only the candidate grids of this PLZ.")
    parser.add_argument("--kcid", type=int, help="With --plz and --bcid: one candidate grid only.")
    parser.add_argument("--bcid", type=int, help="With --plz and --kcid: one candidate grid only.")
    parser.add_argument("--min-buildings", type=int, default=5)
    parser.add_argument("--pylovo-version-id", required=True)
    parser.add_argument("--demand-scope", choices=["all", "residential"], default="all")
    parser.add_argument("--mobility-source", choices=["emobpy", "pool"], default="pool")
    parser.add_argument("--profile-seed", type=int, default=481527)
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-evidence", type=Path)
    args = parser.parse_args(argv)
    if (args.kcid is None) != (args.bcid is None) or (args.kcid is not None and args.plz is None):
        parser.error("--kcid and --bcid go together and need --plz.")
    db = SurroGridDatabase()
    db.pylovo_version_id = str(args.pylovo_version_id)
    candidates = db.list_grid_candidates(
        args.ags, min_buildings=args.min_buildings, demand_scope=args.demand_scope
    )
    if args.plz is not None:
        candidates = [
            candidate for candidate in candidates if int(candidate["plz"]) == args.plz
        ]
    if args.kcid is not None:
        candidates = [
            candidate for candidate in candidates
            if (int(candidate["kcid"]), int(candidate["bcid"])) == (args.kcid, args.bcid)
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
