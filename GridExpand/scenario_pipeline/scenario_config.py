"""Typed, deliberately small scientific scenario configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from .model_cases import MODEL_CASES
PV_LOCATION_MODES = ("predefined", "all_buildings")
PV_SIZING_METHODS = ("annual_electricity_rule", "optimization")


def _only(mapping: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(mapping).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown {label} option(s): {sorted(unknown)}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping.")
    return value


def _positive(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if result < 0 or (result == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {qualifier}.")
    return result


def _number_or_none(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric or null.")
    return float(value)


@dataclass(frozen=True)
class EconomicsConfig:
    import_price_eur_per_kwh: float
    pv_feed_in_tariff_eur_per_kwh: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EconomicsConfig":
        raw = _mapping(raw, "economics.electricity")
        _only(
            raw,
            {"import_price_eur_per_kwh", "pv_feed_in_tariff_eur_per_kwh"},
            "economics.electricity",
        )
        return cls(
            import_price_eur_per_kwh=_positive(
                raw["import_price_eur_per_kwh"],
                "economics.electricity.import_price_eur_per_kwh",
                allow_zero=True,
            ),
            pv_feed_in_tariff_eur_per_kwh=_positive(
                raw["pv_feed_in_tariff_eur_per_kwh"],
                "economics.electricity.pv_feed_in_tariff_eur_per_kwh",
                allow_zero=True,
            ),
        )


@dataclass(frozen=True)
class PvSizingConfig:
    heuristic_method: str
    optimized_method: str
    location_mode: str
    demand_multiplier: float
    fallback_capacity_kwp: float
    maximum_fallback_share: float
    module_capacity_kw_per_m2: float
    flat_roof_utilization: float
    slanted_roof_utilization: float
    tilt_bin_degrees: float
    azimuth_bin_degrees: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PvSizingConfig":
        raw = _mapping(raw, "asset_sizing.pv")
        allowed = {
            "heuristic_method", "optimized_method", "location_mode",
            "demand_multiplier", "fallback_capacity_kwp",
            "maximum_fallback_share", "module_capacity_kw_per_m2",
            "flat_roof_utilization", "slanted_roof_utilization",
            "tilt_bin_degrees", "azimuth_bin_degrees",
        }
        _only(raw, allowed, "asset_sizing.pv")
        heuristic = str(raw["heuristic_method"])
        optimized = str(raw["optimized_method"])
        if heuristic not in PV_SIZING_METHODS or optimized not in PV_SIZING_METHODS:
            raise ValueError(f"PV sizing methods must be one of {PV_SIZING_METHODS}.")
        location_mode = str(raw["location_mode"])
        if location_mode not in PV_LOCATION_MODES:
            raise ValueError(f"PV location mode must be one of {PV_LOCATION_MODES}.")
        fallback_share = _positive(
            raw["maximum_fallback_share"],
            "asset_sizing.pv.maximum_fallback_share",
            allow_zero=True,
        )
        if fallback_share > 1:
            raise ValueError("asset_sizing.pv.maximum_fallback_share must be <= 1.")
        utilization = {}
        for name in ("flat_roof_utilization", "slanted_roof_utilization"):
            utilization[name] = _positive(raw[name], f"asset_sizing.pv.{name}")
            if utilization[name] > 1:
                raise ValueError(f"asset_sizing.pv.{name} must be <= 1.")
        return cls(
            heuristic_method=heuristic,
            optimized_method=optimized,
            location_mode=location_mode,
            demand_multiplier=_positive(raw["demand_multiplier"], "asset_sizing.pv.demand_multiplier"),
            fallback_capacity_kwp=_positive(raw["fallback_capacity_kwp"], "asset_sizing.pv.fallback_capacity_kwp"),
            maximum_fallback_share=fallback_share,
            module_capacity_kw_per_m2=_positive(raw["module_capacity_kw_per_m2"], "asset_sizing.pv.module_capacity_kw_per_m2"),
            flat_roof_utilization=utilization["flat_roof_utilization"],
            slanted_roof_utilization=utilization["slanted_roof_utilization"],
            tilt_bin_degrees=_positive(raw["tilt_bin_degrees"], "asset_sizing.pv.tilt_bin_degrees"),
            azimuth_bin_degrees=_positive(raw["azimuth_bin_degrees"], "asset_sizing.pv.azimuth_bin_degrees"),
        )


@dataclass(frozen=True)
class HeatSizingConfig:
    space_heat_source: str
    indoor_design_temperature_c: float
    heating_limit_temperature_c: float
    heat_pump_design_share: float
    buffer_volume_l_per_kw_th: float
    buffer_usable_temperature_spread_k: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HeatSizingConfig":
        raw = _mapping(raw, "asset_sizing.heat")
        allowed = {
            "space_heat_source",
            "indoor_design_temperature_c", "heating_limit_temperature_c",
            "heat_pump_design_share", "buffer_volume_l_per_kw_th",
            "buffer_usable_temperature_spread_k",
        }
        _only(raw, allowed, "asset_sizing.heat")
        if set(raw) != allowed:
            raise ValueError("asset_sizing.heat is incomplete.")
        source = str(raw["space_heat_source"])
        if source not in {"teaser", "infdb_ro_heat"}:
            raise ValueError(
                "asset_sizing.heat.space_heat_source must be teaser or infdb_ro_heat."
            )
        inside = _positive(raw["indoor_design_temperature_c"], "asset_sizing.heat.indoor_design_temperature_c")
        limit = _positive(raw["heating_limit_temperature_c"], "asset_sizing.heat.heating_limit_temperature_c")
        if limit >= inside:
            raise ValueError("asset_sizing.heat.heating_limit_temperature_c must be below indoor_design_temperature_c.")
        share = _positive(raw["heat_pump_design_share"], "asset_sizing.heat.heat_pump_design_share")
        if share > 1.0:
            raise ValueError("asset_sizing.heat.heat_pump_design_share must be <= 1.")
        return cls(
            space_heat_source=source,
            indoor_design_temperature_c=inside,
            heating_limit_temperature_c=limit,
            heat_pump_design_share=share,
            buffer_volume_l_per_kw_th=_positive(raw["buffer_volume_l_per_kw_th"], "asset_sizing.heat.buffer_volume_l_per_kw_th"),
            buffer_usable_temperature_spread_k=_positive(raw["buffer_usable_temperature_spread_k"], "asset_sizing.heat.buffer_usable_temperature_spread_k"),
        )


@dataclass(frozen=True)
class BatterySizingConfig:
    heuristic_method: str
    optimized_method: str
    location_mode: str
    predefined_locations_when_available: bool
    minimum_pv_kwp_per_annual_mwh: float
    heuristic_usable_kwh_per_pv_kwp: float
    heuristic_usable_kwh_per_annual_mwh: float
    optimized_upper_kwh_per_pv_kwp: float
    optimized_upper_kwh_per_annual_mwh: float
    energy_to_power_hours: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BatterySizingConfig":
        raw = _mapping(raw, "asset_sizing.battery")
        allowed = {
            "heuristic_method", "optimized_method", "location_mode",
            "predefined_locations_when_available",
            "minimum_pv_kwp_per_annual_mwh",
            "heuristic_usable_kwh_per_pv_kwp",
            "heuristic_usable_kwh_per_annual_mwh",
            "optimized_upper_kwh_per_pv_kwp",
            "optimized_upper_kwh_per_annual_mwh",
            "energy_to_power_hours",
        }
        _only(raw, allowed, "asset_sizing.battery")
        if raw["heuristic_method"] != "htw_2025_scaled_rule":
            raise ValueError("asset_sizing.battery.heuristic_method must be htw_2025_scaled_rule.")
        if raw["optimized_method"] != "optimization":
            raise ValueError("asset_sizing.battery.optimized_method must be optimization.")
        if raw["location_mode"] != "all_pv_buildings":
            raise ValueError("asset_sizing.battery.location_mode must be all_pv_buildings.")
        if not isinstance(raw["predefined_locations_when_available"], bool):
            raise ValueError("asset_sizing.battery.predefined_locations_when_available must be true or false.")
        coefficients = {
            name: _positive(raw[name], f"asset_sizing.battery.{name}")
            for name in (
                "heuristic_usable_kwh_per_pv_kwp",
                "heuristic_usable_kwh_per_annual_mwh",
                "optimized_upper_kwh_per_pv_kwp",
                "optimized_upper_kwh_per_annual_mwh",
            )
        }
        above_htw = {
            name: value for name, value in coefficients.items() if value > 1.5
        }
        if above_htw:
            raise ValueError(
                "Battery sizing coefficients must not exceed the HTW 2025 "
                f"recommended upper limit of 1.5: {above_htw}"
            )
        return cls(
            heuristic_method="htw_2025_scaled_rule",
            optimized_method="optimization",
            location_mode="all_pv_buildings",
            predefined_locations_when_available=raw["predefined_locations_when_available"],
            minimum_pv_kwp_per_annual_mwh=_positive(raw["minimum_pv_kwp_per_annual_mwh"], "asset_sizing.battery.minimum_pv_kwp_per_annual_mwh"),
            heuristic_usable_kwh_per_pv_kwp=coefficients["heuristic_usable_kwh_per_pv_kwp"],
            heuristic_usable_kwh_per_annual_mwh=coefficients["heuristic_usable_kwh_per_annual_mwh"],
            optimized_upper_kwh_per_pv_kwp=coefficients["optimized_upper_kwh_per_pv_kwp"],
            optimized_upper_kwh_per_annual_mwh=coefficients["optimized_upper_kwh_per_annual_mwh"],
            energy_to_power_hours=_positive(raw["energy_to_power_hours"], "asset_sizing.battery.energy_to_power_hours"),
        )


@dataclass(frozen=True)
class MobilityConfig:
    commuting_probability: float
    emobpy_timestep_hours: float
    reference_year: int
    passenger_mass_kg: float
    passenger_sensible_heat_w: float
    passengers_per_vehicle: float
    cabin_heat_transfer_coefficient: float
    cabin_air_flow_m3_per_s: float
    driving_cycle_type: str
    road_type: int
    road_slope: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MobilityConfig":
        raw = _mapping(raw, "mobility")
        allowed = {
            "commuting_probability", "emobpy_timestep_hours", "reference_year",
            "passenger_mass_kg", "passenger_sensible_heat_w",
            "passengers_per_vehicle", "cabin_heat_transfer_coefficient",
            "cabin_air_flow_m3_per_s", "driving_cycle_type", "road_type",
            "road_slope",
        }
        _only(raw, allowed, "mobility")
        probability = _positive(raw["commuting_probability"], "mobility.commuting_probability", allow_zero=True)
        if probability > 1:
            raise ValueError("mobility.commuting_probability must be <= 1.")
        cycle = str(raw["driving_cycle_type"])
        if cycle not in {"WLTC", "EPA"}:
            raise ValueError("mobility.driving_cycle_type must be WLTC or EPA.")
        return cls(
            commuting_probability=probability,
            emobpy_timestep_hours=_positive(raw["emobpy_timestep_hours"], "mobility.emobpy_timestep_hours"),
            reference_year=int(_positive(raw["reference_year"], "mobility.reference_year")),
            passenger_mass_kg=_positive(raw["passenger_mass_kg"], "mobility.passenger_mass_kg"),
            passenger_sensible_heat_w=_positive(raw["passenger_sensible_heat_w"], "mobility.passenger_sensible_heat_w", allow_zero=True),
            passengers_per_vehicle=_positive(raw["passengers_per_vehicle"], "mobility.passengers_per_vehicle"),
            cabin_heat_transfer_coefficient=_positive(raw["cabin_heat_transfer_coefficient"], "mobility.cabin_heat_transfer_coefficient"),
            cabin_air_flow_m3_per_s=_positive(raw["cabin_air_flow_m3_per_s"], "mobility.cabin_air_flow_m3_per_s", allow_zero=True),
            driving_cycle_type=cycle,
            road_type=int(_positive(raw["road_type"], "mobility.road_type", allow_zero=True)),
            road_slope=float(_number_or_none(raw["road_slope"], "mobility.road_slope")),
        )


PROCESS_PARAMETER_NAMES = {
    "installed_capacity_kw", "capacity_upper_kw", "fixed_investment_cost_eur",
    "investment_cost_eur_per_kw", "fixed_cost_eur_per_hour",
    "variable_cost_eur_per_kwh", "wacc", "depreciation_years",
    "minimum_power_factor",
}
STORAGE_PARAMETER_NAMES = {
    "installed_energy_kwh", "capacity_upper_kwh", "installed_power_kw",
    "power_upper_kw", "energy_to_power_hours",
    "charge_efficiency", "discharge_efficiency", "self_discharge_per_timestep",
    "investment_cost_eur_per_kw", "investment_cost_eur_per_kwh",
    "fixed_investment_cost_power_eur", "fixed_investment_cost_energy_eur",
    "variable_cost_eur_per_kwh", "wacc", "depreciation_years",
}


@dataclass(frozen=True)
class TechnologyParameters:
    processes: dict[str, dict[str, float | None]]
    storages: dict[str, dict[str, float | None]]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TechnologyParameters":
        raw = _mapping(raw, "technologies")
        _only(raw, {"processes", "storages"}, "technologies")
        processes = _mapping(raw["processes"], "technologies.processes")
        storages = _mapping(raw["storages"], "technologies.storages")
        required_processes = {
            "rooftop_pv", "heatpump_air", "heatpump_booster", "heat_dummy",
            "home_charger", "grid_connection",
        }
        required_storages = {
            "stationary_battery", "thermal_storage", "mobility_storage",
        }
        if set(processes) != required_processes:
            raise ValueError(f"technologies.processes must define {sorted(required_processes)}.")
        if set(storages) != required_storages:
            raise ValueError(f"technologies.storages must define {sorted(required_storages)}.")
        parsed_processes = {}
        for name, values in processes.items():
            values = _mapping(values, f"technologies.processes.{name}")
            _only(values, PROCESS_PARAMETER_NAMES, f"technologies.processes.{name}")
            if set(values) != PROCESS_PARAMETER_NAMES:
                raise ValueError(f"technologies.processes.{name} is incomplete.")
            parsed_processes[name] = {
                key: _number_or_none(value, f"technologies.processes.{name}.{key}")
                for key, value in values.items()
            }
        parsed_storages = {}
        for name, values in storages.items():
            values = _mapping(values, f"technologies.storages.{name}")
            _only(values, STORAGE_PARAMETER_NAMES, f"technologies.storages.{name}")
            if set(values) != STORAGE_PARAMETER_NAMES:
                raise ValueError(f"technologies.storages.{name} is incomplete.")
            parsed_storages[name] = {
                key: _number_or_none(value, f"technologies.storages.{name}.{key}")
                for key, value in values.items()
            }
        return cls(processes=parsed_processes, storages=parsed_storages)


@dataclass(frozen=True)
class TimeAggregationConfig:
    enabled: bool
    number_of_typical_periods: int
    hours_per_period: int
    extreme_period_method: str
    clustering_method: str
    cluster_representation: str
    segmentation: bool
    rescale_cluster_periods: bool
    feature_weights: dict[str, float]
    extreme_features: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TimeAggregationConfig":
        raw = _mapping(raw, "time_aggregation")
        allowed = {
            "enabled", "number_of_typical_periods", "hours_per_period",
            "extreme_period_method", "clustering_method", "cluster_representation",
            "segmentation", "rescale_cluster_periods", "feature_weights",
            "extreme_features",
        }
        _only(raw, allowed, "time_aggregation")
        for name in ("enabled", "segmentation", "rescale_cluster_periods"):
            if not isinstance(raw[name], bool):
                raise ValueError(f"time_aggregation.{name} must be true or false.")
        periods = int(_positive(raw["number_of_typical_periods"], "time_aggregation.number_of_typical_periods"))
        hours = int(_positive(raw["hours_per_period"], "time_aggregation.hours_per_period"))
        extreme_method = str(raw["extreme_period_method"])
        if extreme_method not in {"append", "new_cluster_center", "replace_cluster_center"}:
            raise ValueError("Unsupported time_aggregation.extreme_period_method.")
        weights = _mapping(raw["feature_weights"], "time_aggregation.feature_weights")
        if not weights:
            raise ValueError("time_aggregation.feature_weights cannot be empty.")
        return cls(
            enabled=raw["enabled"],
            number_of_typical_periods=periods,
            hours_per_period=hours,
            extreme_period_method=extreme_method,
            clustering_method=str(raw["clustering_method"]),
            cluster_representation=str(raw["cluster_representation"]),
            segmentation=raw["segmentation"],
            rescale_cluster_periods=raw["rescale_cluster_periods"],
            feature_weights={str(k): _positive(v, f"time_aggregation.feature_weights.{k}") for k, v in weights.items()},
            extreme_features=tuple(str(value) for value in raw["extreme_features"]),
        )


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    milestone_year: int
    model_cases: tuple[str, ...]
    economics: EconomicsConfig
    pv: PvSizingConfig
    battery: BatterySizingConfig
    heat: HeatSizingConfig
    mobility: MobilityConfig
    technologies: TechnologyParameters
    time_aggregation: TimeAggregationConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ScenarioConfig":
        raw = _mapping(raw, "scenario configuration")
        _only(raw, {"scenario", "economics", "asset_sizing", "mobility", "technologies", "time_aggregation"}, "top-level scenario")
        scenario = _mapping(raw["scenario"], "scenario")
        _only(scenario, {"id", "milestone_year", "model_cases"}, "scenario")
        cases = tuple(str(case) for case in scenario["model_cases"])
        unknown_cases = set(cases).difference(MODEL_CASES)
        if unknown_cases:
            raise ValueError(f"Unknown model cases: {sorted(unknown_cases)}")
        if len(cases) != len(set(cases)):
            raise ValueError("scenario.model_cases contains duplicates.")
        economics = _mapping(raw["economics"], "economics")
        _only(economics, {"electricity"}, "economics")
        assets = _mapping(raw["asset_sizing"], "asset_sizing")
        _only(assets, {"pv", "battery", "heat"}, "asset_sizing")
        return cls(
            scenario_id=str(scenario["id"]),
            milestone_year=int(_positive(scenario["milestone_year"], "scenario.milestone_year")),
            model_cases=cases,
            economics=EconomicsConfig.from_dict(economics["electricity"]),
            pv=PvSizingConfig.from_dict(assets["pv"]),
            battery=BatterySizingConfig.from_dict(assets["battery"]),
            heat=HeatSizingConfig.from_dict(assets["heat"]),
            mobility=MobilityConfig.from_dict(raw["mobility"]),
            technologies=TechnologyParameters.from_dict(raw["technologies"]),
            time_aggregation=TimeAggregationConfig.from_dict(raw["time_aggregation"]),
        )

    def pv_sizing_method(self, model_case: str) -> str:
        if model_case not in MODEL_CASES:
            raise ValueError(f"Unknown model case {model_case!r}.")
        if model_case == "post-hems-optimized":
            return self.pv.optimized_method
        if model_case == "pre":
            return "none"
        return self.pv.heuristic_method

    def battery_sizing_method(self, model_case: str) -> str:
        if model_case not in MODEL_CASES:
            raise ValueError(f"Unknown model case {model_case!r}.")
        if model_case == "post-hems-optimized":
            return self.battery.optimized_method
        if model_case == "pre":
            return "none"
        return self.battery.heuristic_method

    def battery_capacity_coefficients(self, model_case: str) -> tuple[float, float]:
        """Return PV- and demand-based battery coefficients for one model case."""
        if model_case == "post-hems-optimized":
            return (
                self.battery.optimized_upper_kwh_per_pv_kwp,
                self.battery.optimized_upper_kwh_per_annual_mwh,
            )
        if model_case == "pre":
            return (0.0, 0.0)
        if model_case not in MODEL_CASES:
            raise ValueError(f"Unknown model case {model_case!r}.")
        return (
            self.battery.heuristic_usable_kwh_per_pv_kwp,
            self.battery.heuristic_usable_kwh_per_annual_mwh,
        )

    def heat_sizing_method(self, model_case: str) -> str:
        if model_case not in MODEL_CASES:
            raise ValueError(f"Unknown model case {model_case!r}.")
        if model_case == "post-hems-optimized":
            return "optimization"
        if model_case == "pre":
            return "none"
        return "full_load_hours_rule"
