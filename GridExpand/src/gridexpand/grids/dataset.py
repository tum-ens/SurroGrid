"""Scenario dataset of given grids: the demand, assets and electrification of every supplied building.

Input: the building supply (``building_objectid, grid_id, bus``) of own grids (the matcher,
:mod:`gridexpand.grids.matching`, plus the grid file of every grid; ``network`` ``real``) or of pylovo grids
(:mod:`gridexpand.grids.pylovo`; ``network`` ``synthetic``, ``grid_id`` = grid case id, no grid files). Every building takes its attributes from InfDB, derived with pylovo's rules
(:mod:`gridexpand.grids.attributes`); no pylovo result is read. Electricity components, GHD rules,
household sampling, LoD2 roofs, vehicles and the electrification assignment follow the method of the
real-vs-synthetic validation, building by building with stable seeds, so a building gets the same scenario in a
given grid as in a pylovo grid. The outputs are the dataset contract of the per-grid runner, the URBS input and the
power flow of one network side:

``paired_real_bus_allocation_plan.csv`` or ``paired_synthetic_bus_allocation_plan.csv`` (one scenario unit per
building, its grid and bus, the grid file or the pylovo grid),
``paired_component_scenario_plan.csv``, ``paired_building_scenario_plan.csv``,
``paired_electrification_assignment.csv`` (+ ``_summary``), ``paired_pv_roof_assignments.csv``,
``paired_roof_sections.csv``, ``paired_ghd_policy_audit.csv``, ``paired_scope_audit.csv``,
``paired_physical_buildings.csv`` (the building attributes) and ``paired_scenario_metadata.json``. The RegioStaR class of the mobility statistics comes from the municipal
register ``pylovo.municipal_register``.

``uv run python -m gridexpand.grids.dataset --supply <building_supply.csv> --grids <dir> --scenario-config <yaml>
--output-dir <dir>`` (read-only database access); pylovo grids: ``--network synthetic --pylovo-version-id <id>``
instead of ``--grids``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import gridexpand.allocation.functions.electricity as electricity
import gridexpand.allocation.functions.mobility as mobility
from gridexpand.allocation.assets.pv.roof_catalog import (
    LOD2_QUALITY_FLAG,
    building_lod2_capacity,
    load_lod2_roof_catalog,
    roof_catalog_options,
)
from gridexpand.allocation.config import config as grid_config
from gridexpand.allocation.scenario_calibration.profiles.paired_profiles import session_pool_supported_models
from gridexpand.common import ghd
from gridexpand.common.building_components import build_building_components
from gridexpand.common.electrification import (
    assignment_manifest_hash,
    assignment_summary,
    build_electrification_assignment,
)
from gridexpand.common.reproducibility import DEFAULT_PROFILE_SEED, frame_fingerprint, stable_seed
from gridexpand.grids import attributes
from gridexpand.scenario.config_loader import load_scenario_config, scenario_identity_key

# Columns of the component plan (paired_component_scenario_plan.csv), the contract read by the URBS-input
# materialization and the equivalence checks of the validation.
PAIRED_COMPONENT_COLUMNS = [
    "scenario_unit_id", "building_objectid", "component_id", "component_kind",
    "component_category", "source_component_category", "effective_floor_area_m2",
    "gross_floor_area_m2", "households", "occupants", "installed_peak_kw",
    "source_pylovo_grid_result_id", "source_pylovo_version_id", "source_building_use",
    "source_building_use_id", "source_building_type", "mix_score", "mix_rule",
    "mix_confidence", "source_lv_id", "source_allocation_bus", "real_target_grid_id",
    "real_target_bus", "synthetic_target_grid_case_id", "synthetic_target_bus",
    "synthetic_bridge_filename", "included_in_lv", "mv_direct", "annual_energy_kwh",
    "profile_method", "profile_hash", "stable_seed", "profile_seed",
    "source_asset_count", "matched_dso_asset_count", "source_asset_ids",
    "source_use_conflict", "suppression_reason",
]
# Columns of the bus allocation plans (paired_<network>_bus_allocation_plan.csv).
PLAN_COLUMNS = [
    "building_objectid", "provider", "postcode", "ags", "regio7", "scenario_unit_id",
    "target_network", "target_grid_id", "allocation_bus", "lv_id", "source_lv_id",
    "source_allocation_bus", "real_grid_id", "real_bus", "real_grid_file",
    "synthetic_grid_case_id", "synthetic_bus", "synthetic_bridge_filename",
    "synthetic_kcid", "synthetic_bcid", "pylovo_grid_result_id", "building_use",
    "building_type", "building_households", "building_floor_area",
    "residential_effective_floor_area_m2", "residential_equivalent_hh_rows",
    "residential_equivalent_hh_annual_kwh", "calibrated_annual_ghd_kwh",
    "pv_roof_capacity_kw", "pv_roof_eligible", "pv_roof_assignment_method",
    "deterministic_vehicle_count", "include_full_local_demand_scenario", "scenario_scope",
]
BUILDING_PLAN_COLUMNS = [
    "building_objectid", "provider", "postcode", "building_use", "building_type",
    "building_households", "building_floor_area", "residential_effective_floor_area_m2",
    "pv_roof_capacity_kw", "pv_roof_eligible", "deterministic_vehicle_count",
]


def paired_profile_hash(
    component_id: str,
    category: str,
    annual_energy_kwh: float,
    profile_method: str,
    stable_profile_seed: int,
    asset_ids: list[str],
) -> str:
    payload = "|".join([
        str(component_id), str(category), f"{float(annual_energy_kwh):.12f}",
        str(profile_method), str(int(stable_profile_seed)), ",".join(sorted(asset_ids)),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pv_scenario_unit_assignments(real_plan: pd.DataFrame) -> pd.DataFrame:
    """Choose exactly one scenario unit for each eligible physical roof: the connection with the largest base demand."""
    candidates = real_plan.copy()
    candidates["building_objectid"] = candidates["building_objectid"].astype(str)
    candidates["_base_annual_kwh"] = pd.to_numeric(
        candidates["residential_equivalent_hh_annual_kwh"], errors="coerce"
    ).fillna(0.0) + pd.to_numeric(
        candidates["calibrated_annual_ghd_kwh"], errors="coerce"
    ).fillna(0.0)
    candidates = candidates.sort_values(
        ["building_objectid", "_base_annual_kwh", "scenario_unit_id"],
        ascending=[True, False, True],
    )
    chosen = candidates.drop_duplicates("building_objectid", keep="first").copy()
    chosen["pv_roof_assignment_method"] = "all_buildings_primary_demand_connection"
    return chosen[
        ["building_objectid", "scenario_unit_id", "pv_roof_assignment_method"]
    ].reset_index(drop=True)


SYNTHETIC_SUPPLY_COLUMNS = ("synthetic_bridge_filename", "synthetic_kcid", "synthetic_bcid", "pylovo_grid_result_id")


def _supply(supply: pd.DataFrame, grid_files: Mapping[str, Path] | None) -> pd.DataFrame:
    """The supply with its grid number (and grid file); ``grid_files`` None: pylovo grids (no files)."""
    missing = [name for name in ("building_objectid", "grid_id", "bus") if name not in supply.columns]
    if missing:
        raise ValueError(f"The building supply lacks the columns {missing}.")
    result = supply[["building_objectid", "grid_id", "bus"]].copy()
    result["building_objectid"] = result["building_objectid"].astype(str)
    result["grid_id"] = result["grid_id"].astype(str)
    result["bus"] = pd.to_numeric(result["bus"], errors="raise").astype(int)
    if result["building_objectid"].duplicated().any():
        raise ValueError("A building has two supplies; the matcher gives every building one bus.")
    if grid_files is None:
        missing = [name for name in SYNTHETIC_SUPPLY_COLUMNS if name not in supply.columns]
        if missing:
            raise ValueError(f"The supply of pylovo grids lacks the columns {missing}.")
        result["grid_file"] = pd.NA
    else:
        unknown = sorted(set(result["grid_id"]) - set(grid_files))
        if unknown:
            raise ValueError(f"Grids without a grid file: {unknown[:5]}")
        result["grid_file"] = result["grid_id"].map(lambda grid_id: str(Path(grid_files[grid_id]).resolve()))
    result["grid_number"] = result["grid_id"].map(grid_numbers(result["grid_id"]))
    return result


def grid_numbers(grid_ids) -> dict[str, int]:
    """The number of every grid: its id when every id is an integer, else 1, 2, ... in the order of the ids.

    The per-grid runner, the power flow and the result tables key real grids by number; the file stem stays in
    ``real_grid_name`` and the database keeps the grid file.
    """
    names = sorted({str(value) for value in grid_ids})
    if all(name.isdigit() for name in names):
        return {name: int(name) for name in names}
    return {name: index for index, name in enumerate(names, start=1)}


def _regio7(engine, buildings: pd.DataFrame) -> pd.Series:
    from sqlalchemy import text

    postcodes = sorted({int(value) for value in buildings["postcode"]})
    with engine.connect() as conn:
        rows = pd.read_sql_query(
            text("SELECT DISTINCT ags, plz, regio7 FROM pylovo.municipal_register WHERE plz = ANY(:plz)"),
            conn, params={"plz": postcodes},
        )
    regio7 = {(int(row["ags"]), int(row["plz"])): int(row["regio7"]) for row in rows.to_dict("records")}
    keys = list(zip(buildings["ags"].astype(int), buildings["postcode"].astype(int)))
    unknown = sorted({key for key in keys if key not in regio7})
    if unknown:
        raise ValueError(f"No pylovo.municipal_register row for (AGS, postcode) {unknown[:5]}.")
    return pd.Series([regio7[key] for key in keys], index=buildings.index)


def _grid_components(physical: pd.DataFrame, *, profile_seed: int, ghd_config, ghd_evidence
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Modelled electricity components and GHD audit of the buildings of one grid."""
    components = build_building_components(physical)
    components, audit = ghd.apply_ghd_policy(physical, components, ghd_config, ghd_evidence)
    components["objectid"] = components["objectid"].astype(str)
    if not components.empty:
        components = electricity.sample_statistics(components, base_seed=profile_seed)
    if components["included_in_lv"].astype(bool).any():
        components, _, _ = electricity.get_elec_demand(components, base_seed=profile_seed,
                                                       return_component_profiles=True)
    else:
        # A given grid can hold only buildings without LV load (MV-direct or non-demand): no profiles.
        components["annual_electricity_kwh"] = 0.0
        for column in ("demand_tot_list", "occ_list"):
            if column not in components:
                components[column] = [[] for _ in range(len(components))]
    return components, audit


def build_dataset(
    supply: pd.DataFrame,
    grid_files: Mapping[str, Path] | None,
    *,
    engine,
    scenario_config_path: Path,
    profile_seed: int,
    output_dir: Path,
    network: str = "real",
    selection_scope_id: str | None = None,
    pylovo_version_id: str = "none",
) -> dict[str, Any]:
    """Write the scenario dataset of the supplied buildings and return its metadata.

    ``network`` ``synthetic``: pylovo grids (no grid files; ``pylovo_version_id`` of the grids). Other labels are
    real grids (``real``; the validation's ``real_<provider>``). ``selection_scope_id`` names the population of the
    electrification draw (default: scenario, network and grids); the same id and seed select the same buildings.
    """
    synthetic = network == "synthetic"
    if synthetic == (grid_files is not None):
        raise ValueError("Grid files belong to real grids; pylovo grids (network synthetic) have none.")
    if synthetic and pylovo_version_id == "none":
        raise ValueError("A dataset of pylovo grids needs their pylovo version.")
    scenario, scenario_hash = load_scenario_config(scenario_config_path)
    grid_config.apply_scenario(scenario)
    pylovo_keys = (supply.assign(building_objectid=supply["building_objectid"].astype(str))
                   .set_index("building_objectid")[list(SYNTHETIC_SUPPLY_COLUMNS)] if synthetic else None)
    supply = _supply(supply, grid_files)
    ids = sorted(supply["building_objectid"])
    physical = attributes.building_attributes(engine, ids)
    physical = physical.merge(
        supply.rename(columns={"building_objectid": "objectid"}), on="objectid", how="inner", validate="one_to_one")
    physical["objectid"] = physical["objectid"].astype(str)
    for column in ("postcode", "gemeindeschluessel"):
        if pd.to_numeric(physical[column], errors="coerce").isna().any():
            raise ValueError(f"Buildings without {column}: "
                             f"{physical.loc[pd.to_numeric(physical[column], errors='coerce').isna(), 'objectid'].tolist()[:5]}")

    ghd_evidence = ghd.load_evidence(scenario.ghd, ids, engine=engine)
    component_frames, ghd_audits = [], []
    for _, group in physical.groupby("grid_id", sort=True):
        components, audit = _grid_components(group.drop(columns=["grid_id", "grid_file", "grid_number"]).reset_index(drop=True),
                                             profile_seed=profile_seed, ghd_config=scenario.ghd,
                                             ghd_evidence=ghd_evidence)
        component_frames.append(components)
        ghd_audits.append(audit)
    components = pd.concat(component_frames, ignore_index=True)
    ghd_audit = pd.concat([audit for audit in ghd_audits if not audit.empty] or ghd_audits[:1], ignore_index=True)

    buildings = physical.rename(columns={"objectid": "building_objectid"})
    buildings["postcode"] = pd.to_numeric(buildings["postcode"]).astype(int)
    buildings["ags"] = pd.to_numeric(buildings["gemeindeschluessel"]).astype(int)
    buildings["regio7"] = _regio7(engine, buildings)

    # One scenario unit per building, ordered by its connection.
    buildings = buildings.sort_values(["grid_number", "bus", "building_objectid"]).reset_index(drop=True)
    buildings["scenario_unit_id"] = np.arange(len(buildings), dtype=int)
    unit = buildings.set_index("building_objectid")

    # Annual demand per component and building.
    components["building_objectid"] = components["objectid"].astype(str)
    components["annual_energy_kwh"] = pd.to_numeric(
        components["annual_electricity_kwh"], errors="coerce"
    ).fillna(0.0).where(components["included_in_lv"].astype(bool), 0.0)
    residential = components["component_category"].eq("Residential")
    components["source_asset_count"] = np.where(
        residential, components["demand_tot_list"].map(lambda value: len(value) if isinstance(value, list) else 0), 1
    )
    by_building = components.assign(
        hh_kwh=components["annual_energy_kwh"].where(residential, 0.0),
        ghd_kwh=components["annual_energy_kwh"].where(~residential, 0.0),
        hh_rows=components["source_asset_count"].where(residential, 0),
        res_area=pd.to_numeric(components["effective_floor_area_m2"], errors="coerce").where(residential, 0.0),
    ).groupby("building_objectid")[["hh_kwh", "ghd_kwh", "hh_rows", "res_area"]].sum()
    buildings["residential_equivalent_hh_annual_kwh"] = buildings["building_objectid"].map(by_building["hh_kwh"]).fillna(0.0)
    buildings["calibrated_annual_ghd_kwh"] = buildings["building_objectid"].map(by_building["ghd_kwh"]).fillna(0.0)
    buildings["residential_equivalent_hh_rows"] = buildings["building_objectid"].map(by_building["hh_rows"]).fillna(0).astype(int)
    buildings["residential_effective_floor_area_m2"] = buildings["building_objectid"].map(by_building["res_area"]).fillna(0.0)
    buildings["building_households"] = buildings["households"]
    buildings["building_floor_area"] = (
        pd.to_numeric(buildings["floor_area"], errors="coerce")
        * pd.to_numeric(buildings["floor_number"], errors="coerce")
    )

    # LoD2 roofs.
    roof_options = roof_catalog_options(scenario.pv)
    roof_catalog = load_lod2_roof_catalog(engine, ids, **roof_options)
    roof_capacity = building_lod2_capacity(roof_catalog)
    buildings["pv_roof_capacity_kw"] = buildings["building_objectid"].map(roof_capacity).fillna(0.0)

    # Vehicles: household occupancy from the electricity sampling, cars from the region's ownership statistics.
    occupancy = (
        components.loc[residential].drop_duplicates("building_objectid")
        .set_index("building_objectid")["occ_list"]
    )
    owned_frames = []
    for region, group in buildings.groupby("regio7"):
        frame = pd.DataFrame({
            "objectid": group["building_objectid"].to_numpy(),
            "building_objectid": group["building_objectid"].to_numpy(),
            "bus": group["bus"].astype(int).to_numpy(),
            "occ_list": [
                occupancy.get(object_id, []) if isinstance(occupancy.get(object_id, []), list) else []
                for object_id in group["building_objectid"]
            ],
        })
        owned_frames.append(mobility.sample_statistics(
            frame, pd.DataFrame([{"regio7": int(region)}]),
            allowed_models=session_pool_supported_models(), base_seed=int(profile_seed),
        ))
    owned = pd.concat(owned_frames, ignore_index=True).set_index("building_objectid")
    buildings["deterministic_vehicle_count"] = (
        buildings["building_objectid"].map(pd.to_numeric(owned["n_cars_tot"], errors="coerce")).fillna(0).astype(int)
    )
    has_household = buildings["building_objectid"].map(lambda object_id: len(occupancy.get(object_id, []) or []) > 0)

    # Electrification assignment, eligibility as in the synthetic route.
    annual = buildings["residential_equivalent_hh_annual_kwh"] + buildings["calibrated_annual_ghd_kwh"]
    is_residential = buildings["residential_effective_floor_area_m2"].gt(0.0)
    inventory = pd.DataFrame({
        "building_objectid": buildings["building_objectid"],
        "building_type": buildings["building_type"] if "building_type" in buildings else pd.NA,
        "heat_eligible": is_residential,
        "heat_exclusion_reason": np.where(is_residential, None, "no_residential_component"),
        "mobility_eligible": is_residential & has_household & buildings["deterministic_vehicle_count"].gt(0),
        "mobility_exclusion_reason": np.select(
            [~is_residential, ~has_household, buildings["deterministic_vehicle_count"].le(0)],
            ["no_residential_component", "no_household", "no_vehicle_inventory"], default=None,
        ),
        "pv_battery_eligible": buildings["pv_roof_capacity_kw"].gt(0.0) & annual.gt(0.0),
        "pv_battery_exclusion_reason": np.select(
            [buildings["pv_roof_capacity_kw"].le(0.0), annual.le(0.0)],
            ["no_usable_lod2_roof", "no_base_electricity"], default=None,
        ),
    })
    grids = sorted(supply["grid_id"].unique())
    if selection_scope_id is None:
        selection_scope_id = f"{scenario.scenario_id}|{network}|grids={','.join(grids)}"
    assignment = build_electrification_assignment(inventory, scenario.electrification,
                                                  selection_scope_id=selection_scope_id,
                                                  profile_seed=int(profile_seed))

    # PV roof assignment: every eligible roof belongs to its only scenario unit.
    plan_base = buildings.assign(
        lv_id=buildings["grid_number"], source_lv_id=buildings["grid_number"], source_allocation_bus=buildings["bus"],
        include_full_local_demand_scenario=True, scenario_scope="paired_full_local_demand", provider=network,
    )
    if synthetic:
        keys = pylovo_keys.loc[plan_base["building_objectid"]].reset_index(drop=True)
        plan_base = plan_base.assign(
            real_grid_id=pd.NA, real_bus=pd.NA, real_grid_file=pd.NA, real_grid_name=pd.NA,
            synthetic_grid_case_id=plan_base["grid_number"], synthetic_bus=plan_base["bus"],
            synthetic_bridge_filename=keys["synthetic_bridge_filename"].to_numpy(),
            synthetic_kcid=keys["synthetic_kcid"].to_numpy(), synthetic_bcid=keys["synthetic_bcid"].to_numpy(),
            pylovo_grid_result_id=keys["pylovo_grid_result_id"].to_numpy(),
        )
    else:
        plan_base = plan_base.assign(
            real_grid_id=plan_base["grid_number"], real_bus=plan_base["bus"], real_grid_file=plan_base["grid_file"],
            real_grid_name=plan_base["grid_id"], synthetic_grid_case_id=pd.NA, synthetic_bus=pd.NA,
            synthetic_bridge_filename=pd.NA, synthetic_kcid=pd.NA, synthetic_bcid=pd.NA,
        )
    pv_assignments = pv_scenario_unit_assignments(plan_base[plan_base["pv_roof_capacity_kw"].gt(0.0)])
    plan_base = plan_base.merge(pv_assignments, on=["building_objectid", "scenario_unit_id"], how="left")
    plan_base["pv_roof_eligible"] = plan_base["pv_roof_assignment_method"].notna()
    real_plan = plan_base.assign(target_network=network, target_grid_id=plan_base["grid_number"],
                                 allocation_bus=plan_base["bus"])[[*PLAN_COLUMNS, "real_grid_name"]]

    # Component plan in the dataset contract.
    rows = []
    for component in components.to_dict("records"):
        object_id = str(component["building_objectid"])
        target = unit.loc[object_id]
        grid_number, bus = int(target["grid_number"]), int(target["bus"])
        category = str(component["component_category"])
        annual_kwh = float(component["annual_energy_kwh"])
        if bool(component["mv_direct"]):
            included, reason, method = False, "outside_paired_lv_scope", "suppressed_mv_direct"
        elif annual_kwh <= 0.0:
            included, reason, method = False, "no_modelled_demand", "suppressed_no_modelled_demand"
        else:
            included, reason = True, None
            method = ("aligned_modelled_hh_profile_v1" if category == "Residential"
                      else f"aligned_modelled_ghd_{category.lower()}_shape_v1")
        seed = stable_seed(int(profile_seed), object_id, category, "electricity", "paired")
        rows.append({
            **component,
            "scenario_unit_id": int(target["scenario_unit_id"]),
            "component_kind": "pylovo",
            "source_component_category": category,
            "source_pylovo_grid_result_id": component.get("pylovo_grid_result_id"),
            "source_pylovo_version_id": component.get("pylovo_version_id"),
            "source_lv_id": grid_number,
            "source_allocation_bus": bus,
            "real_target_grid_id": None if synthetic else grid_number,
            "real_target_bus": None if synthetic else bus,
            "synthetic_target_grid_case_id": grid_number if synthetic else None,
            "synthetic_target_bus": bus if synthetic else None,
            "synthetic_bridge_filename": (pylovo_keys.at[object_id, "synthetic_bridge_filename"]
                                          if synthetic else None),
            "included_in_lv": included,
            "annual_energy_kwh": annual_kwh if included else 0.0,
            "profile_method": method,
            "profile_hash": paired_profile_hash(component["component_id"], category, annual_kwh, method, seed, []),
            "stable_seed": int(seed),
            "profile_seed": int(profile_seed),
            "matched_dso_asset_count": 0,
            "source_asset_ids": "",
            "source_use_conflict": None,
            "suppression_reason": reason,
        })
    component_plan = pd.DataFrame(rows)[PAIRED_COMPONENT_COLUMNS].sort_values(
        ["scenario_unit_id", "building_objectid", "component_id"]
    ).reset_index(drop=True)

    scope_audit = pd.DataFrame([{
        "target_network": network,
        "target_grids": int(real_plan["target_grid_id"].nunique()),
        "physical_buildings": int(real_plan["building_objectid"].nunique()),
        "hh_rows": float(real_plan["residential_equivalent_hh_rows"].sum()),
        "hh_annual_kwh": float(real_plan["residential_equivalent_hh_annual_kwh"].sum()),
        "ghd_annual_kwh": float(real_plan["calibrated_annual_ghd_kwh"].sum()),
    }])
    summary = assignment_summary(assignment)
    postcodes = sorted(int(value) for value in buildings["postcode"].unique())
    metadata = {
        "provider": network,
        "ags": int(buildings["ags"].mode().iloc[0]),
        "ags_list": sorted(int(value) for value in buildings["ags"].unique()),
        "plz": int(buildings["postcode"].mode().iloc[0]),
        "postcodes": postcodes,
        "grids": {name: int(number) for name, number in grid_numbers(grids).items()},
        "attribute_source": "infdb_basedata_pylovo_rules",
        # Own grids read no pylovo result ("none"); pylovo grids record their version (power flow, expansion).
        "pylovo_version_id": str(pylovo_version_id),
        "scenario_scope": "paired_full_local_demand",
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario_hash,
        "scenario_key": scenario_identity_key(scenario.scenario_id, scenario_hash),
        "profile_seed": int(profile_seed),
        "selection_scope_id": selection_scope_id,
        "pv_adoption_mode": "deterministic_share",
        "pv_roof_parameters": roof_options,
        "pv_fallback_buildings": int(
            roof_catalog.loc[roof_catalog["quality_flag"].ne(LOD2_QUALITY_FLAG), "building_objectid"].nunique()
        ),
        "electrification_assignment_summary": summary.to_dict("records"),
        "component_contract": "physical_building_component_v1",
        "paired_contract": "physical_building_component_paired_v2",
        "source_connection_policy": ("pylovo_bus_one_bus_per_building" if synthetic
                                     else "matched_connection_points_one_bus_per_building"),
        "demand_source": "modelled_pylovo_components",
        "component_rows": int(len(component_plan)),
        "included_component_rows": int(component_plan["included_in_lv"].astype(bool).sum()),
        f"{'synthetic' if synthetic else 'real'}_grids": int(real_plan["target_grid_id"].nunique()),
        "ghd_policy": {
            "activity_gating": scenario.ghd.activity_gating,
            "single_volume_one_storey": scenario.ghd.single_volume_one_storey,
            "osm_levels": scenario.ghd.osm_levels,
            "osm_source": None if scenario.ghd.osm is None else scenario.ghd.osm.source,
            "evidence_fingerprint": None if ghd_evidence is None else frame_fingerprint(ghd_evidence.sort_index()),
            **ghd.summarize_ghd_audit(ghd_audit),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        f"paired_{'synthetic' if synthetic else 'real'}_bus_allocation_plan": real_plan,
        "paired_component_scenario_plan": component_plan,
        "paired_building_scenario_plan": plan_base[BUILDING_PLAN_COLUMNS],
        "paired_scope_audit": scope_audit,
        "paired_roof_sections": roof_catalog,
        "paired_pv_roof_assignments": pv_assignments,
        "paired_electrification_assignment": assignment,
        "paired_electrification_assignment_summary": summary,
        "paired_ghd_policy_audit": ghd_audit,
        # The building attributes of the dataset, read again by the per-building heat sources.
        "paired_physical_buildings": buildings.drop(
            columns=[column for column in buildings.columns if buildings[column].map(type).eq(list).any()]
        ),
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    # Consumers read the CSV, whose dtypes differ from the in-memory frame, so the identity hash is taken from the
    # persisted assignment.
    metadata["electrification_assignment_hash"] = assignment_manifest_hash(
        pd.read_csv(output_dir / "paired_electrification_assignment.csv")
    )
    (output_dir / "paired_scenario_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True),
                                                              encoding="utf-8")
    return metadata


def read_only_engine():
    """The GridExpand database engine with read-only sessions (psycopg2; also behind PgBouncer)."""
    from gridexpand.db.engine import get_engine, load_env

    load_env()
    return get_engine().execution_options(postgresql_readonly=True)


def main(argv: list[str] | None = None) -> int:
    from gridexpand.grids import sources

    parser = argparse.ArgumentParser(prog="python -m gridexpand.grids.dataset", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--supply", type=Path, required=True, help="building_supply.csv of gridexpand match")
    parser.add_argument("--grids", type=Path, nargs="+", help="the grid files or their directories (real grids)")
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-seed", type=int, default=DEFAULT_PROFILE_SEED)
    parser.add_argument("--network", default="real",
                        help="label of the grids in the dataset: real (default) or synthetic (pylovo grids)")
    parser.add_argument("--pylovo-version-id", default="none", help="pylovo version of the grids (synthetic)")
    parser.add_argument("--selection-scope-id", default=None,
                        help="population id of the electrification draw (default: scenario, network, grids, version)")
    args = parser.parse_args(argv)
    if (args.network == "synthetic") == bool(args.grids):
        parser.error("--grids for real grids; pylovo grids (--network synthetic) have no files")
    engine = read_only_engine()
    metadata = build_dataset(
        pd.read_csv(args.supply, dtype={"building_objectid": str, "grid_id": str}),
        None if args.network == "synthetic" else sources.grid_files(args.grids),
        engine=engine,
        scenario_config_path=args.scenario_config.resolve(),
        profile_seed=args.profile_seed,
        output_dir=args.output_dir.resolve(),
        network=args.network,
        selection_scope_id=args.selection_scope_id,
        pylovo_version_id=args.pylovo_version_id,
    )
    print(json.dumps({key: value for key, value in metadata.items() if key in (
        "real_grids", "synthetic_grids", "component_rows", "attribute_source", "scenario_key")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
