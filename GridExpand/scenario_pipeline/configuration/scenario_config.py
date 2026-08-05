"""Typed, deliberately small scientific scenario configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MODEL_CASES = (
    "pre",
    "post-inflex-heuristic",
    "post-hems-optimized",
    "post-hems-heuristic",
)
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
class PlaceholderSizingConfig:
    method: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any], label: str) -> "PlaceholderSizingConfig":
        raw = _mapping(raw, label)
        _only(raw, {"method"}, label)
        if raw.get("method") != "temporary_placeholder":
            raise ValueError(f"{label}.method must currently be temporary_placeholder.")
        return cls(method="temporary_placeholder")


@dataclass(frozen=True)
class BatterySizingConfig:
    heuristic_method: str
    optimized_method: str
    location_mode: str
    predefined_locations_when_available: bool
    minimum_pv_kwp_per_annual_mwh: float
    maximum_usable_kwh_per_pv_kwp: float
    maximum_usable_kwh_per_annual_mwh: float
    energy_to_power_hours: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BatterySizingConfig":
        raw = _mapping(raw, "asset_sizing.battery")
        allowed = {
            "heuristic_method", "optimized_method", "location_mode",
            "predefined_locations_when_available",
            "minimum_pv_kwp_per_annual_mwh",
            "maximum_usable_kwh_per_pv_kwp",
            "maximum_usable_kwh_per_annual_mwh",
            "energy_to_power_hours",
        }
        _only(raw, allowed, "asset_sizing.battery")
        if raw["heuristic_method"] != "htw_2025_upper_bound":
            raise ValueError("asset_sizing.battery.heuristic_method must be htw_2025_upper_bound.")
        if raw["optimized_method"] != "optimization":
            raise ValueError("asset_sizing.battery.optimized_method must be optimization.")
        if raw["location_mode"] != "all_pv_buildings":
            raise ValueError("asset_sizing.battery.location_mode must be all_pv_buildings.")
        if not isinstance(raw["predefined_locations_when_available"], bool):
            raise ValueError("asset_sizing.battery.predefined_locations_when_available must be true or false.")
        return cls(
            heuristic_method="htw_2025_upper_bound",
            optimized_method="optimization",
            location_mode="all_pv_buildings",
            predefined_locations_when_available=raw["predefined_locations_when_available"],
            minimum_pv_kwp_per_annual_mwh=_positive(raw["minimum_pv_kwp_per_annual_mwh"], "asset_sizing.battery.minimum_pv_kwp_per_annual_mwh"),
            maximum_usable_kwh_per_pv_kwp=_positive(raw["maximum_usable_kwh_per_pv_kwp"], "asset_sizing.battery.maximum_usable_kwh_per_pv_kwp"),
            maximum_usable_kwh_per_annual_mwh=_positive(raw["maximum_usable_kwh_per_annual_mwh"], "asset_sizing.battery.maximum_usable_kwh_per_annual_mwh"),
            energy_to_power_hours=_positive(raw["energy_to_power_hours"], "asset_sizing.battery.energy_to_power_hours"),
        )


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
    heat_pump: PlaceholderSizingConfig
    time_aggregation: TimeAggregationConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ScenarioConfig":
        raw = _mapping(raw, "scenario configuration")
        _only(raw, {"scenario", "economics", "asset_sizing", "time_aggregation"}, "top-level scenario")
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
        _only(assets, {"pv", "battery", "heat_pump"}, "asset_sizing")
        return cls(
            scenario_id=str(scenario["id"]),
            milestone_year=int(_positive(scenario["milestone_year"], "scenario.milestone_year")),
            model_cases=cases,
            economics=EconomicsConfig.from_dict(economics["electricity"]),
            pv=PvSizingConfig.from_dict(assets["pv"]),
            battery=BatterySizingConfig.from_dict(assets["battery"]),
            heat_pump=PlaceholderSizingConfig.from_dict(assets["heat_pump"], "asset_sizing.heat_pump"),
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
