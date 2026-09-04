from config import config

import json
from pathlib import Path

import pandas as pd
import numpy as np
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_EXCEPTION
import heapq

import src.classes.save_grid as svgrd
import src.functions.weather as wth
import src.functions.electricity as elc
import src.functions.heat as heat
import src.functions.mobility as mbl
from src.assets.battery.materialization import materialize_battery_urbs_inputs
from src.assets.battery.sizing import build_battery_asset_plan
from src.assets.heat.materialization import materialize_heat_urbs_inputs
from src.assets.heat.sizing import build_heat_asset_plan
from src.assets.pv.materialization import materialize_pv_urbs_inputs
from src.assets.pv.profiles import generate_profile_library
from src.assets.pv.roof_catalog import (
    assert_fallback_share,
    building_lod2_capacity,
    load_lod2_roof_catalog,
    read_lod2_roof_catalog_hdf,
)
from src.assets.pv.sizing import build_pv_asset_plan
from common.reproducibility import (
    frame_fingerprint,
    physical_building_id,
    realization_id,
)
from common.building_components import residential_component_mask
from common.electrification import (
    assignment_manifest_hash,
    assignment_summary,
    build_electrification_assignment,
    validate_electrification_assignment_config,
)
from common.timeframe import (
    TIMESLICE_HOURS,
    build_full_year_metadata,
    select_timeframe_from_electricity,
    select_timeframe_from_weather,
)


class Grid:
    def __init__(self, settings):
        ### Setup savefile instance which manages data retrieval and saving
        self.settings = settings
        self.SF = svgrd.SaveFile(
            settings["grid_filename"],
            storage=settings.get("storage", "h5"),
            grid_ref=settings.get("grid_ref"),
            allocation_settings=settings,
        )

        ### Basic grid data
        self.df_buildings, self.df_region, self.df_weather_raw = self.SF.get_input_data()
        self.df_building_components = self.SF.get_building_components()
        self.df_demand_components = self.df_building_components.copy()
        self._apply_demand_scope()
        self.profile_seed = int(self.settings.get("profile_seed", 0))
        building_ids = [physical_building_id(row) for _, row in self.df_buildings.iterrows()]
        self.profile_realization_id = realization_id(self.profile_seed, building_ids)
        self.settings["scenario_assumptions"].update({
            "profile_seed": self.profile_seed,
            "profile_realization_id": self.profile_realization_id,
            "profile_realization_contract": "physical-building-component-v1",
        })
        if self.df_weather_raw is None:
            self.df_weather_raw = pd.DataFrame()
        if self.df_region is None or self.df_region.empty:
            raise ValueError("Missing region metadata in input file (/raw_data/region).")

        region_row = self.df_region.iloc[0]
        self.region = int(region_row["regio7"])            # regiostar region used for mobility statistics
        self.plz = str(region_row["plz"]).zfill(5)         # plz of assumed grid position (not of pylovo grid used as representation)
        self.location = {"lat": float(region_row["lat"]), # latitude of transformer position used for weather data
                         "lon": float(region_row["lon"])} # longitude of transformer position used for weather data
        self.altitude = float(region_row.get("altitude", 0.0))  # altitude of location in meters

        ### Data to be generated
        if not self.settings["weather_data_exists"]: self.df_weather_raw = pd.DataFrame()
        self.df_supim_solar = pd.DataFrame()
        self.df_demand_elec = pd.DataFrame()
        self.df_electricity_component_profiles = pd.DataFrame()
        self.df_demand_component_audit = pd.DataFrame()
        # self.df_demand_elec_react = pd.DataFrame()
        self.df_demand_heat_space = pd.DataFrame()
        self.df_demand_heat_water = pd.DataFrame()
        self._space_heat_source_audit = {}
        self.df_demand_mobility = pd.DataFrame()
        self.df_tve_hpcop = pd.DataFrame()
        self.df_tve_mobility = pd.DataFrame()
        self.df_pv_roof_catalog = pd.DataFrame()
        self.df_pv_asset_plan = pd.DataFrame()
        self.df_pv_selected_sections = pd.DataFrame()
        self.df_pv_process = pd.DataFrame()
        self.df_pv_commodity = pd.DataFrame()
        self.df_battery_asset_plan = pd.DataFrame()
        self.df_battery_storage = pd.DataFrame()
        self.df_battery_audit = pd.DataFrame()
        self.df_heat_asset_plan = pd.DataFrame()
        self.df_heat_audit = pd.DataFrame()
        self.df_heat_process = pd.DataFrame()
        self.df_heat_commodity = pd.DataFrame()
        self.df_heat_process_commodity = pd.DataFrame()
        self.df_heat_storage = pd.DataFrame()
        self.df_pv_process_commodity = pd.DataFrame()
        self.df_pv_audit = pd.DataFrame()
        self.df_electrification_assignment = pd.DataFrame()
        self.df_electrification_summary = pd.DataFrame()
        self._mobility_ownership_ready = False
        self.battery_dict = {} # Holds mobility battery capacities for every vehicle generated by emobpy {(bus, veh_id): cap, ...}
        self.timeframe_mode = self.settings.get("timeframe_mode", "full_year")
        self.timeframe_metadata = dict(self.settings.get("timeframe_metadata") or build_full_year_metadata())
        self._timeframe_slice_applied = False
        self.settings["scenario_assumptions"] = self._scenario_assumptions()
        self.SF.update_timeframe_metadata(self.settings["scenario_assumptions"])

        ### Urbs output sheets
        self.df_weather_urbs = pd.DataFrame()
        self.df_supim = pd.DataFrame()
        self.df_demand = pd.DataFrame()
        self.df_pro = pd.DataFrame()
        self.df_com = pd.DataFrame()
        self.df_pro_com = pd.DataFrame()
        self.df_sto = pd.DataFrame()
        self.df_tve = pd.DataFrame()
        self.df_bsp = pd.DataFrame()


    ############################################
    ############## Demand Scope ################
    ############################################
    def _apply_demand_scope(self):
        demand_scope = self.settings.get("demand_scope", "all")
        if demand_scope not in {"all", "residential"}:
            raise ValueError(f"Unknown demand scope: {demand_scope}")

        before_buildings = len(self.df_buildings)
        before_components = len(self.df_building_components)
        before_buses = self.df_building_components["bus"].dropna().nunique()
        if demand_scope == "all":
            selected_components = self.df_building_components.loc[
                self.df_building_components["included_in_lv"].astype(bool)
            ].copy()
            selected_objectids = self.df_buildings["objectid"].astype(str)
        else:
            selected_components = self.df_building_components.loc[
                residential_component_mask(self.df_building_components)
            ].copy()
            selected_objectids = selected_components["objectid"].astype(str).unique()
            self.df_buildings = self.df_buildings.loc[
                self.df_buildings["objectid"].astype(str).isin(selected_objectids)
            ].copy().reset_index(drop=True)

        if selected_components.empty or self.df_buildings.empty:
            raise ValueError(f"Demand scope {demand_scope} removed all demand components/buildings.")
        self.df_demand_components = selected_components.reset_index(drop=True)
        after_buses = self.df_demand_components["bus"].dropna().nunique()
        stats = {
            "input_buildings": int(before_buildings),
            "selected_buildings": int(len(self.df_buildings)),
            "input_components": int(before_components),
            "selected_components": int(len(self.df_demand_components)),
            "input_buses": int(before_buses),
            "selected_buses": int(after_buses),
        }
        self.settings["demand_scope_stats"] = stats
        if isinstance(self.settings.get("scenario_assumptions"), dict):
            self.settings["scenario_assumptions"].update(stats)
        print(
            f"{demand_scope} demand scope: kept {len(self.df_demand_components)} component(s) "
            f"from {before_components} on {after_buses}/{before_buses} buses."
        )

    @staticmethod
    def _residential_building_mask(df_buildings):
        if "residential_floor_area" not in df_buildings:
            raise ValueError("Residential selection requires residential_floor_area from the mixed-use PyLoVo contract.")
        return pd.to_numeric(df_buildings["residential_floor_area"], errors="coerce").gt(0)

    def _scenario_assumptions(self):
        assumptions = dict(self.settings.get("scenario_assumptions") or {})
        assumptions.update(self.timeframe_metadata)
        assumptions["scenario_key"] = self.settings.get("scenario_key")
        assumptions["component_contract"] = "physical_building_component_v1"
        assumptions["demand_scope"] = self.settings.get("demand_scope", "all")
        if assumptions["demand_scope"] == "residential":
            assumptions["demand_scope_filter"] = "included Residential component (residential_floor_area > 0)"
        assumptions.update(self.settings.get("demand_scope_stats") or {})
        return assumptions


    def _record_profile_fingerprints(self, **frames):
        values = {
            f"profile_hash_{name}": frame_fingerprint(frame)
            for name, frame in frames.items()
            if frame is not None and not frame.empty
        }
        self.settings["scenario_assumptions"].update(values)
        self.SF.update_timeframe_metadata(self.settings["scenario_assumptions"])

    def _build_demand_component_audit(self):
        """Create compact component evidence for the current allocation run."""
        base = self.df_building_components.copy()
        profiled = self.df_demand_components.set_index("component_id")
        profile_ids = [
            str(column[0]) for column in self.df_electricity_component_profiles.columns
        ]
        profile_max = self.df_electricity_component_profiles.max(axis=0)
        profile_max.index = profile_ids
        audit = base.rename(columns={"component_category": "category"})
        audit["scenario_unit_id"] = audit["objectid"].astype(str)
        audit["commodity"] = "electricity"
        audit["annual_energy_kwh"] = audit["component_id"].map(
            profiled["annual_electricity_kwh"]
        ).fillna(0.0)
        audit["max_profile_value"] = audit["component_id"].map(profile_max).fillna(0.0)
        audit["profile_hash"] = audit["component_id"].map(profiled["profile_hash"])
        audit["profile_method"] = audit["component_id"].map(
            profiled["profile_method"]
        ).fillna("not_allocated")
        audit["stable_seed"] = audit["component_id"].map(profiled["stable_seed"])
        selected_ids = set(self.df_demand_components["component_id"].astype(str))
        audit["suppression_reason"] = audit.apply(
            lambda row: (
                "outside_lv_scope" if not bool(row["included_in_lv"])
                else None if str(row["component_id"]) in selected_ids
                else "outside_demand_scope"
            ),
            axis=1,
        )
        audit["source_asset_count"] = pd.NA
        audit["matched_swf_asset_count"] = pd.NA
        audit["mv_direct"] = audit["mv_direct"].astype(bool)
        return audit[
            [
                "component_id", "objectid", "scenario_unit_id", "bus", "category",
                "commodity", "annual_energy_kwh", "max_profile_value", "profile_hash",
                "profile_method", "stable_seed", "source_asset_count",
                "matched_swf_asset_count", "included_in_lv", "suppression_reason",
                "pylovo_version_id", "mix_score", "mix_rule", "mix_confidence", "mv_direct",
            ]
        ]


    def _electrification_scope_id(self) -> str:
        """Return the stable population identity used by the assignment manifest."""
        explicit = self.settings.get("selection_scope_id")
        if explicit:
            return str(explicit)
        grid_ref = self.settings.get("grid_ref") or {}
        identity = grid_ref.get("cell_id") or grid_ref.get("bridge_filename")
        if not identity:
            identity = self.settings.get("grid_filename", "unknown_grid")
        topology = (
            grid_ref.get("version_id")
            or grid_ref.get("pylovo_version_id")
            or self.settings.get("pylovo_version_id")
            or "unknown_topology"
        )
        grid_result = grid_ref.get("grid_result_id", "unknown_grid_result")
        scenario = self.settings["scenario_config"]
        return (
            f"{scenario.scenario_id}|topology={topology}|"
            f"grid_result={grid_result}|grid={identity}|"
            f"scope={self.settings.get('demand_scope', 'all')}"
        )

    def _prepare_mobility_ownership(self) -> None:
        """Sample household ownership once so assignment and profiles share it."""
        if self._mobility_ownership_ready:
            return
        source = self.settings.get("mobility_source", "emobpy")
        allowed_models = mbl.get_pool_supported_models() if source == "pool" else None
        residential_ids = set(
            self.df_demand_components.loc[
                self.df_demand_components["component_category"].eq("Residential")
                & self.df_demand_components["included_in_lv"].astype(bool),
                "objectid",
            ].astype(str)
        )
        mobility_buildings = self.df_buildings.loc[
            self.df_buildings["objectid"].astype(str).isin(residential_ids)
        ].copy()
        if not mobility_buildings.empty:
            mobility_buildings["objectid"] = mobility_buildings["objectid"].astype(str)
            mobility_buildings = mbl.sample_statistics(
                mobility_buildings,
                self.df_region,
                allowed_models=allowed_models,
                base_seed=self.profile_seed,
            )
        sampled_by_id = (
            mobility_buildings.set_index("objectid")
            if not mobility_buildings.empty
            else pd.DataFrame()
        )
        physical_ids = self.df_buildings["objectid"].astype(str)
        for column, default in (("cars_by_flat", []), ("n_cars_tot", 0), ("car_dict", {})):
            if isinstance(sampled_by_id, pd.DataFrame) and column in sampled_by_id:
                values = physical_ids.map(sampled_by_id[column])
            else:
                values = pd.Series(index=self.df_buildings.index, dtype=object)
            self.df_buildings[column] = values.apply(
                lambda value, fallback=default: fallback
                if value is None
                or (
                    not isinstance(value, (list, tuple, dict, np.ndarray))
                    and pd.isna(value)
                )
                else value
            )
        self._mobility_ownership_ready = True

    def _resolve_heat_profile_eligibility(
        self,
        physical: pd.DataFrame,
        residential_components: pd.DataFrame,
        residential_ids: set[str],
    ) -> set[str]:
        """Resolve the active heat source before selecting heat adopters."""
        source = getattr(heat.config, "SPACE_HEAT_SOURCE", "teaser")
        if not residential_ids:
            return set()
        if source == "teaser":
            return set(residential_ids)
        if source != "infdb_ro_heat":
            raise ValueError(f"Unknown space heat source {source!r}.")

        areas = (
            residential_components.groupby("objectid")["effective_floor_area_m2"]
            .sum()
            .rename("residential_effective_floor_area_m2")
        )
        heat_buildings = physical.loc[
            physical["building_objectid"].isin(residential_ids)
        ].copy()
        heat_buildings["residential_effective_floor_area_m2"] = (
            heat_buildings["building_objectid"].map(areas).fillna(0.0)
        )
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
        heat.load_space_heat(
            heat_buildings,
            engine=self.SF.db.engine if self.SF.db is not None else None,
        )
        return set(residential_ids)

    def prepare_electrification_assignment(
        self, roof_catalog: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Create or load the exact physical-building technology manifest."""
        if not self.df_electrification_assignment.empty:
            return self.df_electrification_assignment

        assignment_path = self.settings.get("electrification_assignment_path")
        assignment_source_hash = None
        assignment_source_summary = None
        if assignment_path:
            path = Path(assignment_path)
            if path.suffix.lower() in {".csv", ".txt"}:
                existing = pd.read_csv(path)
            else:
                existing = pd.read_hdf(
                    path, key="raw_data/electrification_assignment"
                )
            full_assignment_hash = assignment_manifest_hash(existing)
            ids = set(self.df_buildings["objectid"].astype(str))
            existing["building_objectid"] = existing["building_objectid"].astype(str)
            existing = existing.loc[
                existing["building_objectid"].isin(ids)
            ].copy()
            expected = len(ids) * 3
            if len(existing) != expected or existing.duplicated(
                ["building_objectid", "technology"]
            ).any():
                raise ValueError(
                    "The supplied electrification assignment is not one row per "
                    "current physical building and technology."
                )
            assignment = existing.reset_index(drop=True)
            metadata_path = path.with_suffix(".json")
            if not metadata_path.exists():
                raise ValueError(
                    f"Prepared electrification assignment is missing sidecar: {metadata_path}"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            assignment_source_hash = metadata.get("assignment_hash")
            if assignment_source_hash != full_assignment_hash:
                raise ValueError(
                    "Prepared electrification assignment sidecar hash does not "
                    "match the assignment rows."
                )
            if metadata.get("scenario_hash") != self.settings.get("scenario_hash"):
                raise ValueError(
                    "Prepared electrification assignment scenario_hash differs "
                    "from the active Step-2 scenario."
                )
            if int(metadata.get("profile_seed", -1)) != int(self.profile_seed):
                raise ValueError(
                    "Prepared electrification assignment profile_seed differs "
                    "from the active Step-2 run."
                )
            assignment_source_summary = metadata.get("assignment_summary")
        else:
            self._prepare_mobility_ownership()
            physical = self.df_buildings.copy()
            physical["building_objectid"] = physical["objectid"].astype(str)
            residential_components = self.df_demand_components.loc[
                residential_component_mask(self.df_demand_components)
            ].copy()
            residential_components["objectid"] = (
                residential_components["objectid"].astype(str)
            )
            residential_ids = set(residential_components["objectid"])
            heat_eligible_ids = self._resolve_heat_profile_eligibility(
                physical,
                residential_components,
                residential_ids,
            )
            physical["heat_eligible"] = physical["building_objectid"].isin(
                heat_eligible_ids
            )
            residential_buildings = physical["building_objectid"].isin(residential_ids)
            physical["heat_exclusion_reason"] = np.select(
                [~residential_buildings, ~physical["heat_eligible"]],
                ["no_residential_component", "no_valid_heat_profile_source"],
                default=None,
            )
            household = physical["occ_list"].apply(
                lambda value: isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0
            )
            vehicles = pd.to_numeric(
                physical["n_cars_tot"], errors="coerce"
            ).fillna(0.0)
            physical["mobility_eligible"] = (
                residential_buildings & household & vehicles.gt(0.0)
            )
            physical["mobility_exclusion_reason"] = np.select(
                [~residential_buildings, ~household, vehicles.le(0.0)],
                ["no_residential_component", "no_household", "no_vehicle_inventory"],
                default=None,
            )
            if roof_catalog is None:
                roof_capacity = pd.Series(0.0, index=physical.index)
            else:
                roof = roof_catalog.copy()
                roof["building_objectid"] = roof["building_objectid"].astype(str)
                roof_capacity = (
                    building_lod2_capacity(roof)
                    .reindex(physical["building_objectid"])
                    .fillna(0.0)
                    .set_axis(physical.index)
                )
            annual = pd.to_numeric(
                physical["annual_electricity_kwh"], errors="coerce"
            ).fillna(0.0)
            physical["pv_roof_eligible"] = roof_capacity.gt(0.0) & annual.gt(0.0)
            physical["pv_battery_eligible"] = physical["pv_roof_eligible"]
            physical["pv_battery_exclusion_reason"] = np.select(
                [
                    roof_capacity.le(0.0),
                    annual.le(0.0),
                ],
                ["no_usable_lod2_roof", "no_base_electricity"],
                default=None,
            )
            source_columns = {
                technology: column
                for technology in ("heat", "mobility", "pv_battery")
                if (column := self.settings.get(
                    "electrification_source_evidence_columns", {}
                ).get(technology))
                and column in physical.columns
            }
            assignment = build_electrification_assignment(
                physical,
                self.settings["scenario_config"].electrification,
                selection_scope_id=self._electrification_scope_id(),
                profile_seed=self.profile_seed,
                source_evidence_columns=source_columns,
            )

        validate_electrification_assignment_config(
            assignment,
            self.settings["scenario_config"].electrification,
            profile_seed=self.profile_seed,
            exact_share=assignment_path is None,
        )
        self.df_electrification_assignment = assignment
        self.df_electrification_summary = assignment_summary(
            assignment, exact_share=assignment_path is None
        )
        local_manifest_hash = assignment_manifest_hash(
            assignment, exact_share=assignment_path is None
        )
        manifest_hash = assignment_source_hash or local_manifest_hash
        self.settings["electrification_assignment_hash"] = manifest_hash
        assumptions = self.settings.setdefault("scenario_assumptions", {})
        assumptions.update(
            {
                "electrification_assignment_hash": manifest_hash,
                "electrification_assignment_local_hash": local_manifest_hash,
                "electrification_assignment_summary": (
                    self.df_electrification_summary.to_dict("records")
                ),
                "electrification_assignment_prepared_summary": (
                    assignment_source_summary
                    if assignment_source_summary is not None
                    else self.df_electrification_summary.to_dict("records")
                ),
            }
        )
        self.SF.update_timeframe_metadata(self._scenario_assumptions())
        return assignment

    def _selected_buildings(self, technology: str) -> set[str]:
        if self.df_electrification_assignment.empty:
            raise ValueError("Electrification assignment has not been prepared.")
        return set(
            self.df_electrification_assignment.loc[
                self.df_electrification_assignment["technology"].eq(technology)
                & self.df_electrification_assignment["selected"].astype(bool),
                "building_objectid",
            ].astype(str)
        )

    @staticmethod
    def _positive_plan_count(
        plan: pd.DataFrame,
        building_column: str,
        capacity_column: str,
    ) -> int:
        if plan.empty or building_column not in plan or capacity_column not in plan:
            return 0
        capacity = pd.to_numeric(plan[capacity_column], errors="coerce").fillna(0.0)
        return int(plan.loc[capacity.gt(0.0), building_column].astype(str).nunique())

    @staticmethod
    def _plan_sum(plan: pd.DataFrame, column: str) -> float:
        if plan.empty or column not in plan:
            return 0.0
        return float(pd.to_numeric(plan[column], errors="coerce").fillna(0.0).sum())

    def _electrification_asset_plan_summary(self) -> dict[str, dict[str, object]]:
        """Expose selected cohorts separately from positive materialized capacity."""
        if self.df_electrification_assignment.empty:
            return {}

        selected_by_technology = {
            technology: self._selected_buildings(technology)
            for technology in ("heat", "mobility", "pv_battery")
        }
        summary: dict[str, dict[str, object]] = {
            technology: {
                "reporting_stage": "step2_urbs_input",
                "selected_candidate_building_count": int(
                    len(selected_by_technology[technology])
                ),
            }
            for technology in selected_by_technology
        }

        pv_capacity = self._plan_sum(self.df_pv_asset_plan, "pv_max_kwp")
        pv_installed = self._plan_sum(
            self.df_pv_asset_plan, "pv_installed_kwp"
        )
        battery_capacity = self._plan_sum(
            self.df_battery_asset_plan, "battery_capacity_upper_kwh"
        )
        battery_installed = self._plan_sum(
            self.df_battery_asset_plan, "battery_installed_kwh"
        )
        summary["pv_battery"].update(
            {
                "step2_materialized_asset_count": self._positive_plan_count(
                    self.df_pv_asset_plan,
                    "building_objectid",
                    "pv_max_kwp",
                ),
                "positive_pv_capacity_upper_bound_building_count": self._positive_plan_count(
                    self.df_pv_asset_plan,
                    "building_objectid",
                    "pv_max_kwp",
                ),
                "positive_pv_input_installed_building_count": self._positive_plan_count(
                    self.df_pv_asset_plan,
                    "building_objectid",
                    "pv_installed_kwp",
                ),
                "step2_input_capacity_kw": pv_capacity,
                "step2_input_installed_capacity_kw": pv_installed,
                "positive_battery_capacity_upper_bound_building_count": self._positive_plan_count(
                    self.df_battery_asset_plan,
                    "building_objectid",
                    "battery_capacity_upper_kwh",
                ),
                "positive_battery_input_installed_building_count": self._positive_plan_count(
                    self.df_battery_asset_plan,
                    "building_objectid",
                    "battery_installed_kwh",
                ),
                "step2_materialized_battery_candidate_count": self._positive_plan_count(
                    self.df_battery_asset_plan,
                    "building_objectid",
                    "battery_capacity_upper_kwh",
                ),
                "battery_capacity_upper_bound_kwh": battery_capacity,
                "battery_input_installed_capacity_kwh": battery_installed,
                "pv_supply_profile_sum_hours": float(
                    self.df_supim_solar.apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .to_numpy()
                    .sum()
                ),
                "pv_supply_profile_basis": "per_unit_available_pv_output",
            }
        )

        heat_capacity = self._plan_sum(
            self.df_heat_asset_plan, "heat_pump_capacity_upper_kw_el"
        )
        heat_installed = self._plan_sum(
            self.df_heat_asset_plan, "heat_pump_installed_kw_el"
        )
        summary["heat"].update(
            {
                "step2_materialized_asset_count": self._positive_plan_count(
                    self.df_heat_asset_plan,
                    "building_objectid",
                    "heat_pump_capacity_upper_kw_el",
                ),
                "positive_heat_pump_capacity_upper_bound_building_count": self._positive_plan_count(
                    self.df_heat_asset_plan,
                    "building_objectid",
                    "heat_pump_capacity_upper_kw_el",
                ),
                "positive_heat_pump_input_installed_building_count": self._positive_plan_count(
                    self.df_heat_asset_plan,
                    "building_objectid",
                    "heat_pump_installed_kw_el",
                ),
                "step2_capacity_upper_kw_el": heat_capacity,
                "step2_input_installed_capacity_kw_el": heat_installed,
                "selected_building_base_electricity_kwh": float(
                    self.df_buildings.loc[
                        self.df_buildings["objectid"].astype(str).isin(
                            selected_by_technology["heat"]
                        ),
                        "annual_electricity_kwh",
                    ]
                    .pipe(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
                if "annual_electricity_kwh" in self.df_buildings
                else 0.0,
                "heat_electricity_outcome_basis": "solver_output_not_step2_input",
                "annual_heat_demand_kwh_th": self._plan_sum(
                    self.df_heat_asset_plan, "annual_space_heat_kwh"
                )
                + self._plan_sum(
                    self.df_heat_asset_plan, "annual_water_heat_kwh"
                ),
            }
        )

        if not self.df_demand_mobility.empty and "n_cars_tot" in self.df_buildings:
            selected_mobility = self.df_buildings["objectid"].astype(str).isin(
                selected_by_technology["mobility"]
            )
            cars = pd.to_numeric(
                self.df_buildings["n_cars_tot"], errors="coerce"
            ).fillna(0.0).where(selected_mobility, 0.0)
            ev_count = int(cars.sum())
            summary["mobility"].update(
                {
                    "step2_materialized_asset_count": ev_count,
                    "positive_ev_building_count": int(cars.gt(0.0).sum()),
                    "positive_ev_vehicle_count": ev_count,
                    "ev_profile_count": int(len(self.battery_dict)),
                    "step2_input_capacity_kw": float(
                        getattr(config, "CS_INST_CAP", 0.0) * ev_count
                    ),
                    "annual_ev_charging_demand_kwh": float(
                        self.df_demand_mobility.apply(pd.to_numeric, errors="coerce")
                        .fillna(0.0)
                        .to_numpy()
                        .sum()
                    ),
                    "ev_energy_basis": "charging_demand_input",
                }
            )
        else:
            summary["mobility"].update(
                {
                    "step2_materialized_asset_count": 0,
                    "positive_ev_building_count": 0,
                    "positive_ev_vehicle_count": 0,
                    "ev_profile_count": 0,
                    "step2_input_capacity_kw": 0.0,
                    "annual_ev_charging_demand_kwh": 0.0,
                    "ev_energy_basis": "charging_demand_input",
                }
            )
        return summary

    ############################################
    ########### Timeseries Generators ########## 
    ############################################
    # The order of these operations has to be followed (e.g. heat depends on electric results)
    def retrieve_weather(self):
        if self.settings["weather_data_exists"]: pass
        else:
            # Get TMY data from SARAH3 dataset as DataFrame
            self.df_weather_raw, self.altitude, selected_months = wth.get_pvgis_tmy_sarah3_dataframe(self.location["lat"], self.location["lon"])
            # Add dew point temperature necessary for vehicle simulation
            self.df_weather_raw["dew_point"] = wth.get_dew_point(self.df_weather_raw["temp_air"], self.df_weather_raw["relative_humidity"])
            # Add soil temperature (1.00-2.55m) necessary for ground source heat pumps
            self.df_weather_raw["soil_temp"] = wth.get_open_meteo_soil_temperature(self.location["lat"], self.location["lon"], selected_months)

    def generate_solar(self):
        """Compile and materialize LoD2 PV after base electricity generation."""
        scenario = self.settings["scenario_config"]
        pv_config = scenario.pv
        id_column = next(
            (
                column
                for column in ("building_objectid", "objectid")
                if column in self.df_buildings
            ),
            None,
        )
        if id_column is None:
            raise ValueError("LoD2 PV requires a building_objectid/objectid column.")
        building_ids = self.df_buildings[id_column].astype(str)
        if building_ids.duplicated().any():
            raise ValueError("Ordinary scenario PV requires one row per physical building object ID.")
        roof_options = {
            "tilt_bin_deg": pv_config.tilt_bin_degrees,
            "azimuth_bin_deg": pv_config.azimuth_bin_degrees,
            "module_capacity_kw_per_m2": pv_config.module_capacity_kw_per_m2,
            "flat_roof_utilization": pv_config.flat_roof_utilization,
            "slanted_roof_utilization": pv_config.slanted_roof_utilization,
            "fallback_capacity_kw": pv_config.fallback_capacity_kwp,
        }
        if self.settings["storage"] == "db":
            self.df_pv_roof_catalog = load_lod2_roof_catalog(
                self.SF.db.engine, building_ids, **roof_options
            )
        else:
            self.df_pv_roof_catalog = read_lod2_roof_catalog_hdf(
                self.SF.input_path, building_ids, **roof_options
            )
        self.prepare_electrification_assignment(self.df_pv_roof_catalog)
        selected_pv_buildings = self._selected_buildings("pv_battery")
        fallback_share = assert_fallback_share(
            self.df_pv_roof_catalog,
            selected_pv_buildings,
            pv_config.maximum_fallback_share,
        )
        profile_library = generate_profile_library(
            self.df_pv_roof_catalog,
            self.df_weather_raw,
            self.location,
            self.altitude,
        )
        annual_electricity = pd.to_numeric(self.df_buildings["annual_electricity_kwh"], errors="coerce").fillna(0.0)
        buildings = pd.DataFrame({
            "building_objectid": building_ids,
            "Site": self.df_buildings["bus"].astype(int),
            "annual_electricity_kwh": annual_electricity,
            "pv_roof_eligible": building_ids.isin(selected_pv_buildings),
        })
        for column in ("building_use", "building_type", "floor_area"):
            if column in self.df_buildings:
                buildings[column] = self.df_buildings[column]
        if "households" in self.df_buildings:
            buildings["number_of_households"] = self.df_buildings["households"]
        sizing_method = scenario.pv_sizing_method(self.settings["model_case"])
        self.df_pv_asset_plan, self.df_pv_selected_sections = build_pv_asset_plan(
            buildings, self.df_pv_roof_catalog, profile_library,
            sizing_method=sizing_method,
            demand_multiplier=pv_config.demand_multiplier,
            eligibility_column="pv_roof_eligible",
        )
        pv = materialize_pv_urbs_inputs(
            self.df_pv_asset_plan, self.df_pv_selected_sections,
            profile_library, sizing_method=sizing_method,
            technical_parameters=scenario.technologies.processes["rooftop_pv"],
        )
        self.df_supim_solar = pv.supim
        self.df_pv_process = pv.process
        self.df_pv_commodity = pv.commodity
        self.df_pv_process_commodity = pv.process_commodity
        self.df_pv_audit = pv.audit
        self.settings["pv_fallback_share"] = fallback_share

    def generate_battery(self):
        """Compile stationary batteries from PV and base electricity only."""
        scenario = self.settings["scenario_config"]
        battery = scenario.battery
        model_case = self.settings["model_case"]
        sizing_method = scenario.battery_sizing_method(model_case)
        pv_coefficient, demand_coefficient = scenario.battery_capacity_coefficients(model_case)
        self.df_battery_asset_plan = build_battery_asset_plan(
            self.df_pv_asset_plan,
            sizing_method=sizing_method,
            minimum_pv_kwp_per_annual_mwh=battery.minimum_pv_kwp_per_annual_mwh,
            usable_kwh_per_pv_kwp=pv_coefficient,
            usable_kwh_per_annual_mwh=demand_coefficient,
            eligible_buildings=self._selected_buildings("pv_battery"),
            location_source="electrification_assignment",
        )
        materialized = materialize_battery_urbs_inputs(
            self.df_battery_asset_plan,
            sizing_method=sizing_method,
            energy_to_power_hours=battery.energy_to_power_hours,
            technical_parameters=scenario.technologies.storages["stationary_battery"],
        )
        self.df_battery_storage = materialized.storage
        plan_audit = self.df_battery_asset_plan.copy()
        plan_audit["sector"] = "stationary_battery"
        plan_audit["audit_record_type"] = "battery_asset_plan"
        self.df_battery_audit = pd.concat(
            [plan_audit, materialized.audit], ignore_index=True, sort=False
        )

    def generate_electricity(self):
        # Profile the explicit electricity components. Physical rows receive
        # only residential occupancy and aggregate annual demand for assets.
        self.df_demand_components = elc.sample_statistics(
            self.df_demand_components, self.profile_seed
        )
        (
            self.df_demand_components,
            self.df_demand_elec,
            self.df_electricity_component_profiles,
        ) = elc.get_elec_demand(
            self.df_demand_components,
            base_seed=self.profile_seed,
            return_component_profiles=True,
        )
        residential = self.df_demand_components.loc[
            self.df_demand_components["component_category"].eq("Residential")
        ]
        occupancy_by_building = dict(
            zip(
                residential["objectid"].astype(str),
                residential["occ_list"],
            )
        )
        component_ids = self.df_demand_components["objectid"].astype(str)
        annual_by_building = (
            self.df_demand_components.assign(_building_objectid=component_ids)
            .groupby("_building_objectid")["annual_electricity_kwh"]
            .sum()
        )
        physical_ids = self.df_buildings["objectid"].astype(str)
        self.df_buildings = self.df_buildings.copy()
        self.df_buildings["occ_list"] = physical_ids.map(
            occupancy_by_building
        ).apply(
            lambda value: value if isinstance(value, (list, tuple, np.ndarray)) else []
        )
        self.df_buildings["annual_electricity_kwh"] = physical_ids.map(
            annual_by_building
        ).fillna(0.0)
        self.df_demand_component_audit = self._build_demand_component_audit()
        self._record_profile_fingerprints(base_electricity=self.df_demand_elec)
        # self.df_demand_elec_react = elc.get_elec_react_demand(self.df_demand_elec)

        # Include daylight saving time effect (electricity timeseries are all UTC+1 only, thus include summer time demand shift):
        # For normal elec demand only after heat, as still needed in this form!  
        # self.df_demand_elec_react = self._add_output_data_daylight_saving_shift(self.df_demand_elec_react)

    def align_electricity_output_time(self):
        self.df_demand_elec = self._add_output_data_daylight_saving_shift(self.df_demand_elec)

    def generate_heat(self):
        # This first sizing method intentionally electrifies residential heat only.
        selected_heat_buildings = (
            self._selected_buildings("heat")
            if not self.df_electrification_assignment.empty
            else None
        )
        residential_components = self.df_demand_components.loc[
            self.df_demand_components["component_category"].eq("Residential")
            & self.df_demand_components["included_in_lv"].astype(bool)
        ]
        residential_mask = self._residential_building_mask(self.df_buildings)
        if selected_heat_buildings is not None:
            residential_mask &= self.df_buildings["objectid"].astype(str).isin(
                selected_heat_buildings
            )
        residential = self.df_buildings.loc[residential_mask].copy().merge(
            residential_components[["objectid", "effective_floor_area_m2"]],
            on="objectid",
            how="inner",
            validate="one_to_one",
        )
        residential = residential.rename(
            columns={"effective_floor_area_m2": "residential_effective_floor_area_m2"}
        )
        gross_area = pd.to_numeric(residential["floor_area"], errors="coerce") * pd.to_numeric(
            residential["floor_number"], errors="coerce"
        )
        residential["residential_area_share"] = (
            pd.to_numeric(residential["residential_effective_floor_area_m2"], errors="coerce")
            / gross_area
        )

        # Electricity is shifted here even when a grid has no residential heat.
        df_wth_input = self._add_input_data_daylight_saving_shift(self.df_weather_raw)
        df_elec_input = self._add_input_data_daylight_saving_shift(self.df_demand_elec)
        self.align_electricity_output_time()
        if residential.empty:
            empty = pd.DataFrame(index=self.df_demand_elec.index)
            self.df_demand_heat_space = empty.copy()
            self.df_demand_heat_water = empty.copy()
            self.df_tve_hpcop = empty.copy()
            print("No residential buildings; skipped heat demand and asset generation.")
            return

        residential = heat.sample_statistics(residential, self.profile_seed)
        space_heat_source_audit = {}
        physical_by_id = self.df_buildings["objectid"].astype(str)
        sampled_by_id = residential.set_index("objectid")
        for column in ("construction_year", "heating_type"):
            if column in residential:
                sampled_values = physical_by_id.map(sampled_by_id[column])
                if column in self.df_buildings.columns:
                    sampled_values = sampled_values.where(
                        sampled_values.notna(), self.df_buildings[column]
                    )
                self.df_buildings[column] = sampled_values

        if getattr(heat.config, "SPACE_HEAT_SOURCE", "teaser") == "infdb_ro_heat":
            # The INFDB loader aggregates by bus. Re-load with the selected
            # physical buildings so an unselected building sharing a bus cannot
            # contribute heat to the materialized profile.
            self.df_demand_heat_space, space_heat_source_audit = (
                heat.load_space_heat(residential)
            )
            self.df_demand_heat_water = heat.generate_opendhw(
                residential, base_seed=self.profile_seed
            )
        elif self.settings["parallel"]:
            print(
                f"Generating heat demands for {len(residential)} building(s) "
                f"with {sum(residential["households"])} flat(s)..."
            )
            building_subsets = self.partition_df_by_cpu(
                residential, self.settings["n_cpu"], "households"
            )
            column_subsets = [
                [(bus, "electricity") for bus in subset["bus"].values]
                for subset in building_subsets
            ]
            job_args = [
                (
                    subset.reset_index(drop=True),
                    df_elec_input[column_subsets[index]],
                    [np.array(df_wth_input[column]) for column in ("dni", "dhi", "temp_air")],
                    self.plz,
                    self.profile_seed,
                )
                for index, subset in enumerate(building_subsets)
            ]
            with Pool() as pool:
                results = pool.starmap(heat.generate_heat_demands, job_args)
            self.df_demand_heat_space = pd.concat(
                [result[0] for result in results], axis=1
            ).sort_index(axis=1, level=[0])
            self.df_demand_heat_water = pd.concat(
                [result[1] for result in results], axis=1
            ).sort_index(axis=1, level=[0])
        else:
            columns = [(bus, "electricity") for bus in residential["bus"].values]
            self.df_demand_heat_space, self.df_demand_heat_water = heat.generate_heat_demands(
                residential,
                df_elec_input[columns],
                [np.array(df_wth_input[column]) for column in ("dni", "dhi", "temp_air")],
                self.plz,
                self.profile_seed,
            )
        self._space_heat_source_audit = dict(space_heat_source_audit)
        if space_heat_source_audit:
            self.settings["scenario_assumptions"].update(space_heat_source_audit)

        self.df_demand_heat_space = self._add_output_data_daylight_saving_shift(
            self.df_demand_heat_space
        )
        self.df_demand_heat_water = self._add_output_data_daylight_saving_shift(
            self.df_demand_heat_water
        )

        self._record_profile_fingerprints(
            space_heat=self.df_demand_heat_space,
            hot_water=self.df_demand_heat_water,
        )

        self.df_tve_hpcop = heat.generate_hp_cop(
            residential,
            self.df_demand_heat_space,
            self.df_demand_heat_water,
            self.df_weather_raw,
        )

        self._record_profile_fingerprints(heat_pump_cop=self.df_tve_hpcop)

        id_column = next((name for name in ("building_objectid", "objectid") if name in residential), None)
        if id_column is None:
            raise ValueError("Heat sizing requires building_objectid/objectid.")
        heat_buildings = pd.DataFrame({
            "building_objectid": residential[id_column].astype(str),
            "Site": residential["bus"].astype(int),
            "floor_area": residential["residential_effective_floor_area_m2"].astype(float),
            "floor_number": 1,
        })
        for column in ("building_type", "building_use"):
            if column in residential:
                heat_buildings[column] = residential[column].values
        if "households" in residential:
            heat_buildings["number_of_households"] = residential["households"].values
        scenario = self.settings["scenario_config"]
        sizing_method = scenario.heat_sizing_method(self.settings["model_case"])
        norm_outside = heat.get_norm_outside_temperature(self.plz)
        self.df_heat_asset_plan, climate = build_heat_asset_plan(
            heat_buildings,
            self.df_demand_heat_space,
            self.df_demand_heat_water,
            self.df_tve_hpcop,
            self.df_weather_raw["temp_air"],
            sizing_method=sizing_method,
            norm_outside_temperature_c=norm_outside,
            indoor_design_temperature_c=scenario.heat.indoor_design_temperature_c,
            heating_limit_temperature_c=scenario.heat.heating_limit_temperature_c,
            heat_pump_design_share=scenario.heat.heat_pump_design_share,
            buffer_volume_l_per_kw_th=scenario.heat.buffer_volume_l_per_kw_th,
            buffer_usable_temperature_spread_k=scenario.heat.buffer_usable_temperature_spread_k,
        )
        materialized = materialize_heat_urbs_inputs(
            self.df_heat_asset_plan,
            sizing_method=sizing_method,
            process_parameters=scenario.technologies.processes,
            storage_parameters=scenario.technologies.storages["thermal_storage"],
        )
        self.df_heat_process = materialized.process
        self.df_heat_commodity = materialized.commodity
        self.df_heat_process_commodity = materialized.process_commodity
        self.df_heat_storage = materialized.storage
        self.df_heat_audit = materialized.audit
        if self._space_heat_source_audit:
            source_audit = {
                "sector": "heat",
                "audit_record_type": "space_heat_source",
                **self._space_heat_source_audit,
            }
            self.df_heat_audit = pd.concat(
                [pd.DataFrame([source_audit]), self.df_heat_audit],
                ignore_index=True,
                sort=False,
            )
        self.settings["scenario_assumptions"].update(climate)
        self.settings["scenario_assumptions"]["heat_sizing_method"] = sizing_method
        self.settings["scenario_assumptions"]["heat_scope"] = "residential_buildings"
        self.SF.update_timeframe_metadata(self.settings["scenario_assumptions"])
        print("Finished generating heat demands and building heat asset plan!")

    def generate_mobility(self):
        mobility_source = self.settings.get("mobility_source", "emobpy")
        # Ownership is sampled once before the manifest is built. The selected
        # vehicle dictionaries below are then the sole source for profiles.
        self._prepare_mobility_ownership()
        selected_mobility_buildings = self._selected_buildings("mobility")
        selected = self.df_buildings["objectid"].astype(str).isin(
            selected_mobility_buildings
        )
        self.df_buildings["car_dict"] = [
            value if keep and isinstance(value, dict) else {}
            for value, keep in zip(self.df_buildings["car_dict"], selected)
        ]
        self.df_buildings["cars_by_flat"] = [
            value if keep else []
            for value, keep in zip(self.df_buildings["cars_by_flat"], selected)
        ]
        self.df_buildings["n_cars_tot"] = self.df_buildings["car_dict"].map(len)

        n_cars = self.df_buildings["n_cars_tot"].sum()
        if mobility_source == "pool":
            print(f"Assigning {n_cars} vehicle(s) from pregenerated mobility profile pool.")
        else:
            cars_per_cpu = -(-n_cars//self.settings["n_cpu"])
            print(f"Simulating {n_cars} vehicle(s). Expected time {40*cars_per_cpu:.1f} minutes!")

        if n_cars > 0:
            if mobility_source == "pool":
                vehicles = {}
                self.df_buildings["car_dict"].apply(lambda x: vehicles.update(x))
                self.df_demand_mobility, self.df_tve_mobility, self.battery_dict = mbl.get_mobility_demand_from_pool(
                    vehicles,
                    self.region,
                )
            elif mobility_source == "emobpy":
                # Add daylight saving dummy shift to input data
                wth_input = self._add_input_data_daylight_saving_shift(self.df_weather_raw)
                wth_input = mbl.prepare_weather_input(wth_input)
                vehicles = [{key: value} for build_dict in self.df_buildings["car_dict"].values for key,value in build_dict.items()]

                with ProcessPoolExecutor(max_workers=self.settings["n_cpu"]) as exe:
                    futures = [exe.submit(mbl.get_mobility_demand, v, wth_input) for v in vehicles]
                    done, pending = wait(futures, return_when=FIRST_EXCEPTION)
                    for fut in done:
                        if fut.exception() is not None:
                            exe.shutdown(cancel_futures=True)
                            raise fut.exception()
                    results = [f.result() for f in futures]

                self.df_demand_mobility = pd.concat([results[i][0] for i in range(len(results))], axis=1)
                self.df_tve_mobility = pd.concat([results[i][1] for i in range(len(results))], axis=1)
                for d in [results[i][2] for i in range(len(results))]: self.battery_dict.update(d)
            else:
                raise ValueError(f"Unknown mobility source: {mobility_source}")
        
        self.df_demand_mobility = self._add_output_data_daylight_saving_shift(self.df_demand_mobility, mobility_dmd=True)
        self.df_tve_mobility = self._add_output_data_daylight_saving_shift(self.df_tve_mobility)
        self._record_profile_fingerprints(
            mobility_demand=self.df_demand_mobility,
            mobility_availability=self.df_tve_mobility,
        )

    ############################################
    ############# Timeframe Helpers ############
    ############################################
    def select_timeframe_from_weather(self):
        if self.timeframe_mode in {"full_year", "max_base_electricity_demand_week"}:
            return
        self._set_timeframe_metadata(
            select_timeframe_from_weather(self.df_weather_raw, self.timeframe_mode)
        )

    def select_timeframe_after_electricity(self):
        if self.timeframe_mode != "max_base_electricity_demand_week":
            return
        self._set_timeframe_metadata(
            select_timeframe_from_electricity(self.df_demand_elec, self.timeframe_mode)
        )

    def _set_timeframe_metadata(self, metadata):
        self.timeframe_metadata = dict(metadata)
        self.settings["timeframe_metadata"] = self.timeframe_metadata
        self.settings["scenario_assumptions"] = self._scenario_assumptions()
        self.SF.update_timeframe_metadata(self.settings["scenario_assumptions"])
        print(
            "Selected timeframe "
            f"{self.timeframe_metadata['timeframe_mode']} "
            f"from {self.timeframe_metadata['timeframe_start']} "
            f"to {self.timeframe_metadata['timeframe_end']} "
            f"({self.timeframe_metadata['horizon_hours']} h)."
        )

    def apply_timeframe_slice(self):
        if self.timeframe_mode == "full_year" or self._timeframe_slice_applied:
            return
        start = self.timeframe_metadata.get("selected_start_hour")
        if start is None:
            raise ValueError(f"No selected week available for {self.timeframe_mode}.")
        end = int(start) + TIMESLICE_HOURS
        for attr in (
            "df_weather_raw",
            "df_supim_solar",
            "df_demand_elec",
            "df_demand_heat_space",
            "df_demand_heat_water",
            "df_demand_mobility",
            "df_tve_hpcop",
            "df_tve_mobility",
        ):
            df = getattr(self, attr)
            if isinstance(df, pd.DataFrame) and not df.empty:
                if len(df) < end:
                    raise ValueError(
                        f"Cannot slice {attr}: need row {end}, found {len(df)} rows."
                    )
                sliced = df.iloc[int(start):end].reset_index(drop=True)
                setattr(self, attr, sliced)
        self._timeframe_slice_applied = True
        print(f"Applied {TIMESLICE_HOURS}-hour timeframe slice starting at hour {start}.")

    ############################################
    ############ Urbs Input Sheets ############# 
    ############################################
    def create_weather_urbs(self):
        df_weather_urbs = self.df_weather_raw[["temp_air", "ghi"]].copy()
        df_weather_urbs.rename(columns={"temp_air":"Tamb", "ghi":"Irradiation"}, inplace=True)
        df_weather_urbs.index.name = "t"
        df_weather_urbs.columns = pd.MultiIndex.from_product([['ambient'], df_weather_urbs.columns])
        self.df_weather_urbs = df_weather_urbs

    def create_supim(self):
        self.df_supim = self.df_supim_solar
        self.df_supim.index.name = "t"
    
    def create_demand(self):
        # self.df_demand = pd.concat([self.df_demand_elec, self.df_demand_elec_react, 
        #                             self.df_demand_heat_space, self.df_demand_heat_water, 
        #                             self.df_demand_mobility], axis=1).reset_index(drop=True)
        self.df_demand = pd.concat([self.df_demand_elec, self.df_demand_heat_space, 
                                    self.df_demand_heat_water, self.df_demand_mobility], axis=1).reset_index(drop=True)
        self.df_demand.index.name = "t"
        
    def create_tve(self):
        self.df_tve = pd.concat([self.df_tve_hpcop, self.df_tve_mobility], axis=1).reset_index(drop=True)
        self.df_tve.index.name = "t"

    def create_bsp(self):
        economics = self.settings["scenario_config"].economics
        df_bsp = pd.DataFrame(
            [[economics.import_price_eur_per_kwh, economics.pv_feed_in_tariff_eur_per_kwh]]
            * len(self.df_supim_solar.index),
            index=self.df_supim_solar.index,
            columns=["electricity_import", "electricity_feed_in"],
        )
        self.df_bsp = df_bsp

    def create_processes(self):
        consumer_buses = list(self.df_demand_elec.columns.get_level_values(0).unique())
        dfs = [
            elc.create_pro_elec(consumer_buses),
            self.df_pv_process,
        ]
        if self.settings["include_heat"]:
            dfs.append(self.df_heat_process)
        if self.settings["include_mobility"]:
            dfs.append(mbl.create_pro_mob(self.battery_dict))

        self.df_pro = pd.concat(dfs, axis=0).reset_index(drop=True)

    def create_commodities(self):
        consumer_buses = list(self.df_demand_elec.columns.get_level_values(0).unique())
        dfs = [
            elc.create_com_elec(consumer_buses),
            self.df_pv_commodity,
        ]
        if self.settings["include_heat"]:
            dfs.append(self.df_heat_commodity)
        if self.settings["include_mobility"]:
            dfs.append(mbl.create_com_mob(self.battery_dict))

        self.df_com = pd.concat(dfs, axis=0).reset_index(drop=True)

    def create_process_commodity(self):
        dfs = [
            elc.create_pro_com_elec(),
            self.df_pv_process_commodity,
        ]
        if self.settings["include_heat"]:
            dfs.append(self.df_heat_process_commodity)
        if self.settings["include_mobility"]:
            dfs.append(mbl.create_pro_com_mob(self.battery_dict))

        self.df_pro_com = pd.concat(dfs, axis=0).reset_index(drop=True)

    def create_storages(self):
        dfs = [self.df_battery_storage]
        if self.settings["include_heat"]:
            dfs.append(self.df_heat_storage)
        if self.settings["include_mobility"]:
            dfs.append(mbl.create_sto_mob(self.battery_dict))

        self.df_sto = pd.concat(dfs, axis=0).reset_index(drop=True)
    


    ############################################
    ########## Saving all grid data ############ 
    ############################################
    def save_grid_data(self):
        if not self.df_electrification_assignment.empty:
            self.settings["scenario_assumptions"] = self._scenario_assumptions()
            self.settings["scenario_assumptions"]["electrification_asset_plan_summary"] = (
                self._electrification_asset_plan_summary()
            )
            self.SF.update_timeframe_metadata(self.settings["scenario_assumptions"])

        ### Copy input file into results to write results to it:
        self.SF.copy_save_file()
        self.SF.save_df(self.df_building_components, "raw_data/building_components")

        ### Save other data:
        self.SF.save_timeframe_metadata()
        self.SF.save_df(self.df_weather_raw, "raw_data/weather")
        self.SF.save_df(self.df_buildings,   "raw_data/buildings")
        if not self.df_electrification_assignment.empty:
            self.SF.save_df(
                self.df_electrification_assignment,
                "raw_data/electrification_assignment",
            )
            self.SF.save_df(
                self.df_electrification_summary,
                "raw_data/electrification_assignment_summary",
            )
        self.SF.save_df(self.df_pv_roof_catalog, "raw_data/pv_roof_sections")
        self.SF.save_df(self.df_pv_asset_plan, "raw_data/asset_plan")
        self.SF.save_df(self.df_pv_selected_sections, "raw_data/pv_selected_sections")
        self.SF.save_df(self.df_pv_audit, "raw_data/pv_asset_audit")
        self.SF.save_df(self.df_battery_asset_plan, "raw_data/battery_asset_plan")
        self.SF.save_df(self.df_battery_audit, "raw_data/battery_asset_audit")
        self.SF.save_df(self.df_heat_asset_plan, "raw_data/heat_asset_plan")
        self.SF.save_df(self.df_heat_audit, "raw_data/heat_asset_audit")
        self.SF.save_df(self.df_demand_component_audit, "raw_data/demand_component_audit")
        self.SF.save_allocated_vehicles(self.df_buildings, self.battery_dict)

        ### Saving urbs input sheets:
        self.SF.save_df(self.df_demand,      "urbs_in/demand")
        self.SF.save_df(self.df_supim,       "urbs_in/supim")
        self.SF.save_df(self.df_tve,         "urbs_in/eff_factor")
        self.SF.save_df(self.df_bsp,         "urbs_in/buy_sell_price")
        self.SF.save_df(self.df_weather_urbs,"urbs_in/weather")
        self.SF.save_df(self.df_pro,         "urbs_in/process")
        self.SF.save_df(self.df_com,         "urbs_in/commodity")
        self.SF.save_df(self.df_pro_com,     "urbs_in/process_commodity")
        self.SF.save_df(self.df_sto,         "urbs_in/storage")

    ############################################
    ################# Helpers ################## 
    ############################################
    @staticmethod
    def _add_input_data_daylight_saving_shift(df_ts):
        """ To align human behaviour with daylight savings time:
            - Insert a dummy row (later deleted) at 02:00-03:00AM on ts_hour1 (the hour of the year which is skipped), in order to get the human activity of one hour later with the unshifted weather
            - Remove a row (later replaced) at 02:00-03:00AM on ts_hour2 (the hour of the year which is reapeated), in order to realign human activity with weather data
        """
        if len(df_ts)==0: return df_ts.copy()
        else:
            ts_hour1 = 2090 #(= 02:00AM-03:00AM, 29th March 2009), at this position insert previous hour (already accounted for zero indexing)
            ts_hour2 = 7130 #(= 02:00AM-03:00AM, 10th October 2009), at this position delete hour (already accounted for zero indexing)
            df_ts = df_ts.copy()

            ### Delete alignement row
            df_ts = df_ts.drop(index=ts_hour2).reset_index(drop=True)

            ### Insert dummy row:
            new_row = df_ts.iloc[ts_hour1-1].copy()
            new_row_df = pd.DataFrame([new_row], columns=df_ts.columns)
            df_ts = pd.concat([df_ts.iloc[:ts_hour1], new_row_df, df_ts.iloc[ts_hour1:]]).reset_index(drop=True)

            return df_ts

    @staticmethod
    def _add_output_data_daylight_saving_shift(df_ts, mobility_dmd=False):
        """ To align human behaviour with daylight savings time:
            - Now delete the dummy row at 02:00-03:00AM of ts_hour1 (the hour of the year which is skipped), in order to delete the human activity which never actually occured
            - Add a row (simply copy previous timestep) at 02:00-03:00AM of ts_hour2 (the hour of the year which is skipped), in order to get two hours with same human acitivty
            """
        if len(df_ts)==0: return df_ts.copy()
        else:
            ts_hour1 = 2090 #(= 02:00AM-03:00AM, 29th March 2009), at this position delete the dummy row (already accounted for zero indexing)
            ts_hour2 = 7130 #(= 02:00AM-03:00AM, 10th October 2009), at this position copy the previous row and insert below to realign weather with behaviour (already accounted for zero indexing)
            df_ts = df_ts.copy()

            ### Insert copy row:
            new_row = df_ts.iloc[ts_hour2].copy()
            new_row_df = pd.DataFrame([new_row], columns=df_ts.columns)
            df_ts = pd.concat([df_ts.iloc[:ts_hour2+1], new_row_df, df_ts.iloc[ts_hour2+1:]]).reset_index(drop=True)

            # If mobility dataframe, we don't want to copy an existing accumulated demand (would lead to huge demand spike) -> simply set previous copied timestep to 0
            if mobility_dmd: df_ts.iloc[ts_hour2] = 0

            ### Delete dummy row
            df_ts = df_ts.drop(index=ts_hour1).reset_index(drop=True)

        return df_ts
    
    @staticmethod
    def partition_df_by_cpu(df: pd.DataFrame, n_cpus: int, count_column: str ) -> list[pd.DataFrame]:
        """
        Partition the DataFrame `df` (one row per building, with count_column proportional to computational load e.g "n_cars_tot" or "n_flats_tot")
        into up to `n_cpus` subsets, balancing total car counts as evenly as possible
        without splitting any building. If any bin ends up empty, it is dropped from the result.

        Uses the Longest‐Processing‐Time (LPT) greedy heuristic:
        1. Sort buildings by descending `count_column`.
        2. Maintain a min‐heap of (current_load, bin_id) for each of the `n_cpus` bins.
        3. Assign each building to the bin with the smallest load, updating that bin’s load.
        4. After assignment, discard any empty bins.

        Returns:
            A list of pandas DataFrames; each DataFrame is a subset of `df` (same columns/index),
            and no returned DataFrame is empty.
        """
        # 1. Create a list of (index, cars) and sort descending by cars
        building_list = list(df[count_column].items())  # [(idx_0, cars_0), (idx_1, cars_1), ...]
        building_list.sort(key=lambda x: x[1], reverse=True)

        # 2. Initialize a min‐heap [(current_load, bin_id), ...] for bin_id in [0 .. n_cpus-1]
        heap: list[tuple[int, int]] = [(0, bin_id) for bin_id in range(n_cpus)]
        heapq.heapify(heap)

        # 3. Prepare a list of lists to collect row‐indices for each bin
        bins_indices: list[list[pd.Index]] = [[] for _ in range(n_cpus)]

        # 4. Greedily assign each building to the bin with the smallest current load
        for idx, n_count in building_list:
            if n_count == 0: pass
            else:
                current_load, bin_id = heapq.heappop(heap)
                bins_indices[bin_id].append(idx)
                new_load = current_load + n_count
                heapq.heappush(heap, (new_load, bin_id))

        # 5. Convert each non‐empty list of indices into a DataFrame slice
        bins_dfs: list[pd.DataFrame] = []
        for indices_list in bins_indices:
            if not indices_list:
                # Skip any bin that has no assigned buildings
                continue
            subset_df = df.loc[indices_list].copy()
            bins_dfs.append(subset_df)

        return bins_dfs
