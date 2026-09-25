"""Build one provider's paired scenario from its pylovo alignment bundle.

The real-grid side comes from the finalized alignment bundle
(``<alignment_dir>/<provider>/building-mapping.json``): one real grid and bus
per canonical building. The compared population is pylovo's metric cohort of
the same version (``comparison.json``), so the expansion comparison uses exactly
the buildings of the structural comparison. Demand is modelled identically for
both providers from the pylovo building components; no DSO load inventory is
read. Outputs follow the paired dataset contract consumed by
``pipeline/paired_urbs_input.py`` and the paired runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from ..paths import ENV_PATH

from gridexpand.db.database import SurroGridDatabase
from gridexpand.common.electrification import (
    assignment_manifest_hash,
    assignment_summary,
    build_electrification_assignment,
)
from gridexpand.common.reproducibility import stable_seed
from gridexpand.allocation.config import config as grid_config
import gridexpand.allocation.functions.electricity as electricity
import gridexpand.allocation.functions.mobility as mobility
from gridexpand.scenario.config_loader import (
    load_scenario_config,
    scenario_identity_key,
)

from ..profiles.paired_profiles import session_pool_supported_models
from ..profiles.profile_contract import (
    assert_paired_component_plan_equivalence,
    assert_paired_plan_equivalence,
)
from .paired_allocation import (
    PAIRED_COMPONENT_COLUMNS,
    _paired_profile_hash,
    _pv_scenario_unit_assignments,
)
from .pv_roof_potential import building_lod2_capacity, load_lod2_roof_catalog

PROVIDERS = ("swf", "uzw")
UZW_INDEX_FILE = "index.json"
UZW_LEDGER_FILE = "source-ledger.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def uzw_delivery_fingerprint(root: Path) -> str:
    """Same hash as pylovo ``validations_new.buildings.fingerprint``."""
    root = Path(root).resolve()
    index = json.loads((root / UZW_INDEX_FILE).read_text(encoding="utf-8"))
    paths = {UZW_INDEX_FILE, UZW_LEDGER_FILE}
    for row in index:
        if row.get("file"):
            paths.add(f"{row['cohort']}/{row['file']}")
    hashes = {name: _sha256(root / name) for name in sorted(paths)}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def real_building_mapping(
    provider: str,
    alignment_dir: Path,
    uzw_grids_dir: Path | None,
) -> pd.DataFrame:
    """Return one real grid, bus and grid file per bundle building."""
    bundle = alignment_dir / provider
    document = json.loads((bundle / "building-mapping.json").read_text(encoding="utf-8"))
    postcode = {}
    if provider == "swf":
        workbooks = {}
        for station in document["stations"]:
            path = Path(station["workbook"])
            if _sha256(path) != station["sha256"]:
                raise ValueError(f"SWF workbook differs from its bundle hash: {path}")
            workbooks[station["station_id"]] = str(path)
        rows = [
            {
                "building_objectid": str(row["building_id"]),
                "real_grid_id": int(str(row["station_id"]).removeprefix("LV_")),
                "real_bus": int(row["source_bus_id"]),
                "real_grid_file": workbooks[row["station_id"]],
                "postcode": int(row["postcode"]),
            }
            for row in document["buildings"]
            if row.get("inclusion_status") == "included"
        ]
        return pd.DataFrame(rows)

    if provider != "uzw":
        raise ValueError(f"Unknown provider {provider!r}.")
    if uzw_grids_dir is None:
        raise ValueError("ÜZW needs the frozen delivery directory (--uzw-grids-dir).")
    manifest = json.loads((bundle / "generation-manifest.json").read_text(encoding="utf-8"))
    actual = uzw_delivery_fingerprint(uzw_grids_dir)
    if actual != manifest["delivery_fingerprint"]:
        raise ValueError(
            f"ÜZW delivery {uzw_grids_dir} has fingerprint {actual}, the bundle "
            f"expects {manifest['delivery_fingerprint']}."
        )
    index = json.loads((uzw_grids_dir / UZW_INDEX_FILE).read_text(encoding="utf-8"))
    area_files = {
        int(row["area"]): str(uzw_grids_dir / row["cohort"] / row["file"])
        for row in index
        if row.get("file") and row.get("cohort") == "resolved"
    }
    for row in document["buildings"]:
        postcode[str(row["building_id"])] = int(row["postcode"])
    links = pd.DataFrame(
        [
            {
                "building_objectid": str(link["building_id"]),
                "real_grid_id": int(link["area_id"]),
                "real_bus": int(link["source_bus_id"]),
            }
            for link in document["links"]
            if link.get("mapping_status") == "matched"
        ]
    ).drop_duplicates()
    split = links.groupby("building_objectid").size()
    if split.gt(1).any():
        raise ValueError(
            "ÜZW buildings with several supplies: "
            f"{split[split.gt(1)].index.tolist()[:10]}"
        )
    links["real_grid_file"] = links["real_grid_id"].map(area_files)
    if links["real_grid_file"].isna().any():
        raise ValueError("ÜZW bundle areas without a resolved grid file.")
    links["postcode"] = links["building_objectid"].map(postcode)
    return links[links["postcode"].notna()].astype({"postcode": int})


def metric_population(population_json: Path, provider: str) -> set[str]:
    document = json.loads(population_json.read_text(encoding="utf-8"))
    return {str(value) for value in document["metric_cohort"][provider]["metric_building_ids"]}


def _read_sql(database: SurroGridDatabase, query: str, **params) -> pd.DataFrame:
    with database.engine.connect() as conn:
        return pd.read_sql_query(text(query), conn, params=params)


def register_synthetic_grids(
    database: SurroGridDatabase,
    *,
    pylovo_version_id: str,
    postcodes: list[int],
) -> pd.DataFrame:
    """Register every pylovo grid of the postcodes; AGS from building keys."""
    grids = _read_sql(
        database,
        """
        SELECT g.grid_result_id, g.version_id, g.plz, g.kcid, g.bcid,
               MIN(b.gemeindeschluessel) AS gemeindeschluessel,
               COUNT(DISTINCT b.gemeindeschluessel) AS n_keys,
               COUNT(*) AS n_buildings
        FROM pylovo.grid_result g
        JOIN pylovo.buildings_result b
          ON b.grid_result_id = g.grid_result_id AND b.version_id = g.version_id
        WHERE g.version_id::text = :version AND g.plz = ANY(:plz)
        GROUP BY g.grid_result_id, g.version_id, g.plz, g.kcid, g.bcid
        ORDER BY g.plz, g.kcid, g.bcid
        """,
        version=str(pylovo_version_id),
        plz=[int(value) for value in postcodes],
    )
    if grids.empty:
        raise ValueError(f"No pylovo v{pylovo_version_id} grids for {postcodes}.")
    if grids["n_keys"].gt(1).any():
        raise ValueError("A pylovo grid spans several municipalities.")
    grids["ags"] = grids["gemeindeschluessel"].astype(int)
    grids["candidate_index"] = grids.groupby("ags").cumcount()
    grids["cell_id"] = grids["ags"].astype(str) + "-" + grids["candidate_index"].map("{:02d}".format)
    grids["bridge_filename"] = (
        grids["cell_id"] + "_" + grids["plz"].astype(str) + "_"
        + grids["kcid"].astype(str) + "_" + grids["bcid"].astype(str) + ".h5"
    )
    grids["version_id"] = grids["version_id"].astype(str)
    grids["grid_case_id"] = [
        database.get_or_create_grid_case(row) for row in grids.to_dict("records")
    ]
    return grids


def _grid_components(
    database: SurroGridDatabase,
    grid: dict[str, Any],
    *,
    profile_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Physical buildings and modelled electricity components of one grid."""
    physical, _, _ = database.read_step2_input_data(grid)
    components = database.read_building_components(grid, physical)
    components["objectid"] = components["objectid"].astype(str)
    components = electricity.sample_statistics(components, base_seed=profile_seed)
    components, _, _ = electricity.get_elec_demand(
        components, base_seed=profile_seed, return_component_profiles=True
    )
    physical["objectid"] = physical["objectid"].astype(str)
    return physical, components


def build_aligned_allocation(
    *,
    provider: str,
    alignment_dir: Path,
    population_json: Path,
    pylovo_version_id: str,
    scenario_config_path: Path,
    profile_seed: int,
    output_dir: Path,
    uzw_grids_dir: Path | None = None,
) -> dict[str, Any]:
    load_dotenv(ENV_PATH, override=True)
    os.environ["PYLOVO_VERSION_ID"] = str(pylovo_version_id)
    scenario, scenario_hash = load_scenario_config(scenario_config_path)
    grid_config.apply_scenario(scenario)
    for technology in ("heat", "mobility", "pv_battery"):
        if scenario.electrification.for_technology(technology).adoption_mode != "deterministic_share":
            raise ValueError("Aligned allocation has no source inventory; use deterministic_share.")

    real = real_building_mapping(provider, alignment_dir, uzw_grids_dir)
    population = metric_population(population_json, provider)
    missing = sorted(population - set(real["building_objectid"]))
    if missing:
        raise ValueError(f"Metric population buildings missing from the bundle: {missing[:10]}")
    real = real[real["building_objectid"].isin(population)].copy()

    database = SurroGridDatabase()
    database.pylovo_version_id = str(pylovo_version_id)
    database.ensure_schema()
    postcodes = sorted(real["postcode"].unique().tolist())
    grids = register_synthetic_grids(
        database, pylovo_version_id=str(pylovo_version_id), postcodes=postcodes
    )
    located = _read_sql(
        database,
        """
        SELECT objectid AS building_objectid, grid_result_id
        FROM pylovo.buildings_result
        WHERE version_id::text = :version AND objectid = ANY(:ids)
        """,
        version=str(pylovo_version_id),
        ids=sorted(population),
    )
    if set(located["building_objectid"]) != population:
        raise ValueError("Metric population buildings missing from pylovo buildings_result.")
    cohort_grids = grids[grids["grid_result_id"].isin(located["grid_result_id"])]

    physical_frames, component_frames = [], []
    for grid in cohort_grids.to_dict("records"):
        physical, components = _grid_components(database, grid, profile_seed=profile_seed)
        outside = sorted(set(physical["objectid"]) - population)
        if outside:
            raise ValueError(
                f"Synthetic grid {grid['bridge_filename']} holds buildings outside "
                f"the metric population: {outside[:10]}"
            )
        for frame in (physical, components):
            frame["synthetic_grid_case_id"] = int(grid["grid_case_id"])
            frame["synthetic_bridge_filename"] = grid["bridge_filename"]
            frame["synthetic_kcid"] = int(grid["kcid"])
            frame["synthetic_bcid"] = int(grid["bcid"])
            frame["ags"] = int(grid["ags"])
        physical_frames.append(physical)
        component_frames.append(components)
    physical = pd.concat(physical_frames, ignore_index=True)
    components = pd.concat(component_frames, ignore_index=True)
    if physical["objectid"].duplicated().any() or set(physical["objectid"]) != population:
        raise ValueError("Synthetic grids do not partition the metric population.")

    buildings = physical.rename(columns={"objectid": "building_objectid", "bus": "synthetic_bus"})
    buildings = buildings.merge(real, on="building_objectid", how="left", validate="one_to_one")
    if not buildings["postcode_x"].astype(int).eq(buildings["postcode_y"]).all():
        raise ValueError("Bundle and pylovo postcodes differ for some buildings.")
    buildings["postcode"] = buildings.pop("postcode_x").astype(int)
    buildings = buildings.drop(columns="postcode_y")
    regio7 = {
        (int(row["ags"]), int(row["plz"])): int(row["regio7"])
        for row in _read_sql(
            database,
            "SELECT DISTINCT ags, plz, regio7 FROM pylovo.municipal_register WHERE plz = ANY(:plz)",
            plz=postcodes,
        ).to_dict("records")
    }
    buildings["regio7"] = [
        regio7[(int(ags), int(plz))] for ags, plz in zip(buildings["ags"], buildings["postcode"])
    ]

    # One scenario unit per building, ordered by its real connection.
    buildings = buildings.sort_values(["real_grid_id", "real_bus", "building_objectid"]).reset_index(drop=True)
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
    roof_options = {
        "tilt_bin_deg": scenario.pv.tilt_bin_degrees,
        "azimuth_bin_deg": scenario.pv.azimuth_bin_degrees,
        "module_capacity_kw_per_m2": scenario.pv.module_capacity_kw_per_m2,
        "flat_roof_utilization": scenario.pv.flat_roof_utilization,
        "slanted_roof_utilization": scenario.pv.slanted_roof_utilization,
        "fallback_capacity_kw": scenario.pv.fallback_capacity_kwp,
    }
    roof_catalog = load_lod2_roof_catalog(
        database.engine, sorted(population), **roof_options
    )
    roof_capacity = building_lod2_capacity(roof_catalog)
    buildings["pv_roof_capacity_kw"] = buildings["building_objectid"].map(roof_capacity).fillna(0.0)

    # Vehicles: household occupancy from the same electricity sampling, cars
    # from the region's ownership statistics.
    occupancy = (
        components.loc[residential].drop_duplicates("building_objectid")
        .set_index("building_objectid")["occ_list"]
    )
    owned_frames = []
    for region, group in buildings.groupby("regio7"):
        frame = pd.DataFrame(
            {
                "objectid": group["building_objectid"].to_numpy(),
                "building_objectid": group["building_objectid"].to_numpy(),
                "bus": group["real_bus"].astype(int).to_numpy(),
                "occ_list": [
                    occupancy.get(object_id, []) if isinstance(occupancy.get(object_id, []), list) else []
                    for object_id in group["building_objectid"]
                ],
            }
        )
        owned_frames.append(
            mobility.sample_statistics(
                frame,
                pd.DataFrame([{"regio7": int(region)}]),
                allowed_models=session_pool_supported_models(),
                base_seed=int(profile_seed),
            )
        )
    owned = pd.concat(owned_frames, ignore_index=True).set_index("building_objectid")
    buildings["deterministic_vehicle_count"] = (
        buildings["building_objectid"].map(pd.to_numeric(owned["n_cars_tot"], errors="coerce"))
        .fillna(0).astype(int)
    )
    has_household = buildings["building_objectid"].map(
        lambda object_id: len(occupancy.get(object_id, []) or []) > 0
    )

    # Electrification assignment, eligibility as in the synthetic route.
    annual = buildings["residential_equivalent_hh_annual_kwh"] + buildings["calibrated_annual_ghd_kwh"]
    is_residential = buildings["residential_effective_floor_area_m2"].gt(0.0)
    inventory = pd.DataFrame(
        {
            "building_objectid": buildings["building_objectid"],
            "heat_eligible": is_residential,
            "heat_exclusion_reason": np.where(is_residential, None, "no_residential_component"),
            "mobility_eligible": is_residential & has_household & buildings["deterministic_vehicle_count"].gt(0),
            "mobility_exclusion_reason": np.select(
                [~is_residential, ~has_household, buildings["deterministic_vehicle_count"].le(0)],
                ["no_residential_component", "no_household", "no_vehicle_inventory"],
                default=None,
            ),
            "pv_battery_eligible": buildings["pv_roof_capacity_kw"].gt(0.0) & annual.gt(0.0),
            "pv_battery_exclusion_reason": np.select(
                [buildings["pv_roof_capacity_kw"].le(0.0), annual.le(0.0)],
                ["no_usable_lod2_roof", "no_base_electricity"],
                default=None,
            ),
        }
    )
    selection_scope_id = (
        f"{scenario.scenario_id}|aligned|{provider}|"
        f"plz={','.join(map(str, postcodes))}|version={pylovo_version_id}"
    )
    assignment = build_electrification_assignment(
        inventory,
        scenario.electrification,
        selection_scope_id=selection_scope_id,
        profile_seed=int(profile_seed),
    )
    assignment_hash = assignment_manifest_hash(assignment)

    # PV roof assignment: every eligible roof belongs to its only scenario unit.
    plan_base = buildings.assign(
        lv_id=buildings["real_grid_id"],
        source_lv_id=buildings["real_grid_id"],
        source_allocation_bus=buildings["real_bus"],
        include_full_local_demand_scenario=True,
        scenario_scope="paired_full_local_demand",
        provider=provider,
    )
    pv_assignments = _pv_scenario_unit_assignments(
        plan_base[plan_base["pv_roof_capacity_kw"].gt(0.0)], None, location_mode="all_buildings"
    )
    plan_base = plan_base.merge(pv_assignments, on=["building_objectid", "scenario_unit_id"], how="left")
    plan_base["pv_roof_eligible"] = plan_base["pv_roof_assignment_method"].notna()

    plan_columns = [
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
    real_plan = plan_base.assign(
        target_network=f"real_{provider}",
        target_grid_id=plan_base["real_grid_id"],
        allocation_bus=plan_base["real_bus"],
    )[plan_columns]
    synthetic_plan = plan_base.assign(
        target_network="synthetic",
        target_grid_id=plan_base["synthetic_grid_case_id"],
        allocation_bus=plan_base["synthetic_bus"],
    )[plan_columns]
    assert_paired_plan_equivalence(real_plan, synthetic_plan)

    # Component plan in the paired contract.
    rows = []
    for component in components.to_dict("records"):
        object_id = str(component["building_objectid"])
        target = unit.loc[object_id]
        category = str(component["component_category"])
        annual_kwh = float(component["annual_energy_kwh"])
        if bool(component["mv_direct"]):
            included, reason, method = False, "outside_paired_lv_scope", "suppressed_mv_direct"
        elif annual_kwh <= 0.0:
            included, reason, method = False, "no_modelled_demand", "suppressed_no_modelled_demand"
        else:
            included, reason = True, None
            method = (
                "aligned_modelled_hh_profile_v1" if category == "Residential"
                else f"aligned_modelled_ghd_{category.lower()}_shape_v1"
            )
        seed = stable_seed(int(profile_seed), object_id, category, "electricity", "paired")
        rows.append({
            **component,
            "scenario_unit_id": int(target["scenario_unit_id"]),
            "component_kind": "pylovo",
            "source_component_category": category,
            "source_pylovo_grid_result_id": component.get("pylovo_grid_result_id"),
            "source_pylovo_version_id": component.get("pylovo_version_id"),
            "source_lv_id": int(target["real_grid_id"]),
            "source_allocation_bus": int(target["real_bus"]),
            "real_target_grid_id": int(target["real_grid_id"]),
            "real_target_bus": int(target["real_bus"]),
            "synthetic_target_grid_case_id": int(target["synthetic_grid_case_id"]),
            "synthetic_target_bus": int(target["synthetic_bus"]),
            "included_in_lv": included,
            "annual_energy_kwh": annual_kwh if included else 0.0,
            "profile_method": method,
            "profile_hash": _paired_profile_hash(
                component["component_id"], category, annual_kwh, method, seed, []
            ),
            "stable_seed": int(seed),
            "profile_seed": int(profile_seed),
            "matched_swf_asset_count": 0,
            "source_asset_ids": "",
            "source_use_conflict": None,
            "suppression_reason": reason,
        })
    component_plan = pd.DataFrame(rows)[PAIRED_COMPONENT_COLUMNS].sort_values(
        ["scenario_unit_id", "building_objectid", "component_id"]
    ).reset_index(drop=True)
    assert_paired_component_plan_equivalence(component_plan)

    building_plan = plan_base[[
        "building_objectid", "provider", "postcode", "building_use", "building_type",
        "building_households", "building_floor_area", "residential_effective_floor_area_m2",
        "pv_roof_capacity_kw", "pv_roof_eligible", "deterministic_vehicle_count",
    ]]
    scope_audit = pd.DataFrame([
        {
            "target_network": frame["target_network"].iloc[0],
            "target_grids": int(frame["target_grid_id"].nunique()),
            "physical_buildings": int(frame["building_objectid"].nunique()),
            "hh_rows": float(frame["residential_equivalent_hh_rows"].sum()),
            "hh_annual_kwh": float(frame["residential_equivalent_hh_annual_kwh"].sum()),
            "ghd_annual_kwh": float(frame["calibrated_annual_ghd_kwh"].sum()),
        }
        for frame in (real_plan, synthetic_plan)
    ])
    summary = assignment_summary(assignment)
    bundle_manifest = json.loads(
        (alignment_dir / provider / "generation-manifest.json").read_text(encoding="utf-8")
    )
    metadata = {
        "provider": provider,
        "ags": int(buildings["ags"].mode().iloc[0]),
        "ags_list": sorted(int(value) for value in buildings["ags"].unique()),
        "plz": int(postcodes[0]),
        "postcodes": [int(value) for value in postcodes],
        "pylovo_version_id": str(pylovo_version_id),
        "alignment_dir": str(alignment_dir),
        "alignment_fingerprint": bundle_manifest["alignment_fingerprint"],
        "uzw_grids_dir": None if uzw_grids_dir is None else str(uzw_grids_dir),
        "population_json": str(population_json),
        "population_sha256": _sha256(population_json),
        "min_physical_buildings_per_target_grid": 5,
        "scenario_scope": "paired_full_local_demand",
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario_hash,
        "scenario_key": scenario_identity_key(scenario.scenario_id, scenario_hash),
        "profile_seed": int(profile_seed),
        "pv_adoption_mode": "deterministic_share",
        "pv_roof_parameters": roof_options,
        "pv_fallback_buildings": int(
            roof_catalog.loc[roof_catalog["quality_flag"].ne("lod2"), "building_objectid"].nunique()
        ),
        "electrification_assignment_hash": assignment_hash,
        "electrification_assignment_summary": summary.to_dict("records"),
        "component_contract": "physical_building_component_v1",
        "paired_contract": "physical_building_component_paired_v2",
        "source_connection_policy": "alignment_bundle_one_bus_per_building",
        "demand_source": "modelled_pylovo_components",
        "component_rows": int(len(component_plan)),
        "included_component_rows": int(component_plan["included_in_lv"].astype(bool).sum()),
        "real_grids": int(real_plan["target_grid_id"].nunique()),
        "synthetic_grids": int(synthetic_plan["target_grid_id"].nunique()),
        "registered_pylovo_grid_cases": int(len(grids)),
        "registered_pylovo_buildings": int(grids["n_buildings"].sum()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "paired_real_bus_allocation_plan": real_plan,
        "paired_synthetic_bus_allocation_plan": synthetic_plan,
        "paired_component_scenario_plan": component_plan,
        "paired_building_scenario_plan": building_plan,
        "paired_scope_audit": scope_audit,
        "paired_roof_sections": roof_catalog,
        "paired_pv_roof_assignments": pv_assignments,
        "paired_electrification_assignment": assignment,
        "paired_electrification_assignment_summary": summary,
        "paired_registered_synthetic_grids": grids,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    # Consumers read the CSV, whose dtypes differ from the in-memory frame, so
    # the identity hash is taken from the persisted assignment.
    metadata["electrification_assignment_hash"] = assignment_manifest_hash(
        pd.read_csv(output_dir / "paired_electrification_assignment.csv")
    )
    (output_dir / "paired_scenario_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--uzw-grids-dir", type=Path, default=None)
    parser.add_argument("--pylovo-version-id", required=True)
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument("--profile-seed", type=int, default=481527)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = build_aligned_allocation(
        provider=args.provider,
        alignment_dir=args.alignment_dir.expanduser().resolve(),
        population_json=args.population.expanduser().resolve(),
        pylovo_version_id=str(args.pylovo_version_id),
        scenario_config_path=args.scenario_config.resolve(),
        profile_seed=args.profile_seed,
        output_dir=args.output_dir.resolve(),
        uzw_grids_dir=None if args.uzw_grids_dir is None else args.uzw_grids_dir.expanduser().resolve(),
    )
    print(json.dumps({key: metadata[key] for key in (
        "provider", "real_grids", "synthetic_grids", "component_rows",
        "electrification_assignment_summary",
    )}, indent=2, default=str))


if __name__ == "__main__":
    main()
