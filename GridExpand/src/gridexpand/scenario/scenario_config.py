"""Typed, deliberately small scientific scenario configuration."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import math
from typing import Any

from gridexpand.common.ghd import GhdConfig
from gridexpand.common.timeframe import REFERENCE_YEAR

from .model_cases import get_model_case

PV_SIZING_METHODS = ("annual_electricity_rule", "optimization")
ADOPTION_MODES = ("deterministic_share", "source_inventory")
ELECTRIFICATION_TECHNOLOGIES = ("heat", "mobility", "pv_battery")


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


def _share(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be a finite number in [0, 1].")
    return result


BUILDING_TYPES = ("SFH", "TH", "MFH", "AB")


@dataclass(frozen=True)
class TechnologyAdoptionConfig:
    adoption_mode: str
    building_share: float | None = None
    # deterministic_share only: shares per residential building type (SFH, TH, MFH,
    # AB); building_share applies to buildings whose type is not listed.
    building_share_by_type: dict[str, float] | None = None
    # pv_battery only: share of the selected PV buildings that also get a battery.
    battery_share_of_selected: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any], label: str) -> "TechnologyAdoptionConfig":
        raw = _mapping(raw, label)
        allowed = {"adoption_mode", "building_share", "building_share_by_type"}
        if label.endswith("pv_battery"):
            allowed.add("battery_share_of_selected")
        _only(raw, allowed, label)
        if "adoption_mode" not in raw:
            raise ValueError(f"{label}.adoption_mode is required.")
        mode = str(raw["adoption_mode"])
        if mode not in ADOPTION_MODES:
            raise ValueError(
                f"{label}.adoption_mode must be one of {ADOPTION_MODES}."
            )
        has_share = "building_share" in raw
        by_type = None
        if mode == "deterministic_share":
            if not has_share:
                raise ValueError(f"{label}.building_share is required for deterministic_share.")
            share = _share(raw["building_share"], f"{label}.building_share")
            if "building_share_by_type" in raw:
                mapping = _mapping(raw["building_share_by_type"], f"{label}.building_share_by_type")
                unknown = sorted(set(map(str, mapping)).difference(BUILDING_TYPES))
                if unknown or not mapping:
                    raise ValueError(
                        f"{label}.building_share_by_type needs keys from {BUILDING_TYPES}; got {unknown or 'none'}."
                    )
                by_type = {
                    str(key): _share(value, f"{label}.building_share_by_type.{key}")
                    for key, value in mapping.items()
                }
        else:
            if has_share or "building_share_by_type" in raw:
                raise ValueError(f"{label}.building_share is not allowed for source_inventory.")
            share = None
        battery = _share(raw.get("battery_share_of_selected", 1.0), f"{label}.battery_share_of_selected")
        return cls(
            adoption_mode=mode, building_share=share, building_share_by_type=by_type,
            battery_share_of_selected=battery,
        )

    def share_for_type(self, building_type: Any) -> float | None:
        """Configured share of a building of ``building_type`` (deterministic_share)."""
        if self.building_share_by_type and str(building_type) in self.building_share_by_type:
            return self.building_share_by_type[str(building_type)]
        return self.building_share


@dataclass(frozen=True)
class ElectrificationConfig:
    heat: TechnologyAdoptionConfig
    mobility: TechnologyAdoptionConfig
    pv_battery: TechnologyAdoptionConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ElectrificationConfig":
        raw = _mapping(raw, "electrification")
        _only(raw, {"heat", "mobility", "pv_battery"}, "electrification")
        missing = [name for name in ELECTRIFICATION_TECHNOLOGIES if name not in raw]
        if missing:
            raise ValueError(f"electrification is missing: {missing}")
        return cls(
            heat=TechnologyAdoptionConfig.from_dict(raw["heat"], "electrification.heat"),
            mobility=TechnologyAdoptionConfig.from_dict(
                raw["mobility"], "electrification.mobility"
            ),
            pv_battery=TechnologyAdoptionConfig.from_dict(
                raw["pv_battery"], "electrification.pv_battery"
            ),
        )

    def for_technology(self, technology: str) -> TechnologyAdoptionConfig:
        if technology not in ELECTRIFICATION_TECHNOLOGIES:
            raise ValueError(f"Unknown electrification technology {technology!r}.")
        return getattr(self, technology)


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
            "heuristic_method", "optimized_method",
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
class InternalHeatConfig:
    """Explicit 1R1C assumptions; coefficients use the stored ro-heat envelope."""
    parameter_source: str = "infdb_rc"
    refurbishment_state: str = "stored_ro_heat"
    minimum_temperature_c: float = 20.0
    hems_preheat_uplift_k: float = 2.0
    ventilation_air_changes_per_hour: float = 0.5
    zone_height_m: float = 3.125
    person_gain_w: float = 70.0
    electricity_gain_fraction: float = 0.362
    glazing_solar_transmittance: float = 0.6
    blind_irradiance_threshold_w_per_m2: float = 100.0
    closed_blind_transmittance: float = 0.15

    @classmethod
    def from_dict(cls, raw):
        raw = _mapping(raw, "asset_sizing.heat.internal")
        fields = cls.__dataclass_fields__
        _only(raw, set(fields), "asset_sizing.heat.internal")
        values = {name: raw.get(name, field.default) for name, field in fields.items()}
        if values["parameter_source"] != "infdb_rc" or values["refurbishment_state"] != "stored_ro_heat":
            raise ValueError("Internal heat requires infdb_rc and the traceable stored_ro_heat refurbishment state.")
        for name in set(fields) - {"parameter_source", "refurbishment_state"}:
            values[name] = _positive(values[name], f"asset_sizing.heat.internal.{name}", allow_zero=name in {"hems_preheat_uplift_k", "person_gain_w", "electricity_gain_fraction", "glazing_solar_transmittance", "closed_blind_transmittance"})
        for name in ("electricity_gain_fraction", "glazing_solar_transmittance", "closed_blind_transmittance"):
            if values[name] > 1:
                raise ValueError(f"Internal heat {name} must be <= 1.")
        return cls(**values)


_TEASER_LEGACY = {
    "heated_area_ratio": 1.0,
    "envelope_source": "tabula",
    "subtract_shared_walls": False,
    "attic": "heated",
    "attic_ceiling_b": 0.8,
    "pitched_roof_area_ratio": 1.05,
    "lod2_window_ratio": None,
    "sfh_roof_fix": False,
    "solar_beam": "normal",
    "window_shading_factor": 1.0,
    "window_frame_fraction": 0.0,
    "blind_threshold_w_per_m2": 100.0,
    "ground_floor_b": 1.0,
    "age_mapping": "band_end",
    "apartment_block_after_1978": "cap_1978",
    "air_change_per_h": 0.5,
    "ventilation_height_m": None,
    "heating_limit_c": None,
    "fixed_heating_off_days": (135, 259),
    "retrofit_share_before_1979": 0.0,
    "retrofit_share_1979_1994": 0.0,
    "consumption_factor": "none",
}


@dataclass(frozen=True)
class TeaserHeatConfig:
    """Assumptions of the TEASER/DistrictGenerator space-heat path (``space_heat_source: teaser``).

    ``preset: corrected`` (the default) fixes the defects found in the 2026-11-10 review. ``preset: legacy``
    reproduces the TEASER profiles of SurroGrid 2.0.0 (dev 1c671ac) bit for bit. Single options can be set on top
    of either preset. The internal 1R1C model (``space_heat_source: internal``) does not read this block.

    S1 ``heated_area_ratio`` (legacy 1.0): TEASER's ``net_leased_area`` = ratio x the residential gross floor
    area of InfDB (footprint x ``floor_number``). TEASER builds every TABULA element area, the air volume and the
    internal-mass areas from ``net_leased_area``, and TABULA's reference area A_C,ref is the conditioned net floor
    area, about 1.0-1.1 x the living area (IWU 2015, App. B), so the ratio maps the InfDB gross area to the
    heated living area. Default 0.55 = heated (occupied) living area / InfDB gross area, Bavaria-wide:
    - InfDB gross residential area (pylovo.buildings_result v1, all of Bavaria): 1,152 Mio m2 for 6.39 M
      households;
    - living area 31.12.2022: 650.5 Mio m2 in residential buildings, 665.6 Mio m2 in all buildings
      (Bayerisches Landesamt fuer Statistik 2023, Statistischer Bericht F2400C 202200, Tab. 3 and 4; StMWi
      Bayern, Daten zur Raumbeobachtung, Tab. 5.5: 97.8 m2 per dwelling), i.e. 0.56-0.58 of the InfDB gross area;
    - minus just over 4 % vacant dwellings (Zensus 2022) gives about 0.55.
    A check on the 43,193 Bavarian buildings inside the Zensus 2022 100 m cells loaded in infdb-pylovo
    (cell mean living area per dwelling x households) gives 0.58 (SFH 0.55, TH 0.55, MFH 0.62, AB 0.57).
    Caveats: a Bavarian average, not per type; the 2022 stock is a Zensus 2011-based update; the two Bavarian
    publications differ by about 2 % (97.8 vs 99.9 m2 per dwelling). The ratio is below the usual living /
    gross floor area ratio of 0.7-0.8 (DIN 277) because InfDB counts floors from the full LoD2 height up to the
    ridge at about 4 m per floor, so attics count as full storeys. An InfDB version with eaves-based floors
    (``floor_source: eaves``) gives 0.75 x this gross area on 275 common buildings; for such a version the
    ratio is about 0.73 and must be set explicitly.

    LoD2 envelope (``envelope_source: lod2``, legacy ``tabula``): outer walls = sum of LoD2 WallSurface areas
    (minus shared walls if ``subtract_shared_walls``) minus the window area, split by TEASER's orientation shares;
    roof = sum of LoD2 RoofSurface areas; ground floor = footprint. Windows keep TEASER's TABULA window ratio on
    the heated area, or ``lod2_window_ratio`` x heated area. Buildings without surfaces keep the TABULA envelope.
    ``attic: unheated`` (default) closes the heated zone at the eaves for pitched roofs (roof surface >
    ``pitched_roof_area_ratio`` x footprint): the top-floor ceiling (= footprint) with the temperature reduction
    factor ``attic_ceiling_b`` (0.8, DIN V 4108-6 for a ceiling to an unheated attic) replaces the roof surface,
    gets no solar or sky radiation, and the gable walls above the eaves leave the wall area. Flat roofs keep the
    roof surface. The eaves height follows InfDB (05_fill_floor_number.sql): the lowest point of the roof
    surfaces at the roof-area median, above the lowest ground point. ``attic: heated`` keeps the full LoD2 hull.

    S2 ``sfh_roof_fix``: TEASER stores the tilted SFH roof under orientations 0/90; the legacy loader counted it as
    wall and added a second roof of 1.2 x the ground floor. S3 ``solar_beam: horizontal`` passes GHI - DHI as the
    beam irradiance on the horizontal that the Perez/Duffie-Beckman code expects (legacy: DNI); window shading
    follows TABULA (shading 0.6, frame 0.3) instead of blinds closing above 100 W/m2 all year. S4
    ``ground_floor_b``: temperature reduction factor of the ground floor. S5 ``age_mapping: stock_share`` draws the
    TABULA class within the Zensus band by the IWU 2011 building counts (legacy: last year of the band) and
    apartment blocks after 1978 use the MFH archetype of their class (TEASER has no AB after 1978; legacy: AB 1978).
    S6 TABULA ventilation 0.4 1/h on the heated area x 2.5 m and a heating limit on the daily mean outdoor
    temperature (12 degC, TABULA) instead of the fixed heating break on days 135-258. S7 ``retrofit_share_*`` and S8
    ``consumption_factor: iwu`` are options, off by default.
    """

    preset: str = "corrected"
    heated_area_ratio: float = 0.55
    envelope_source: str = "lod2"
    subtract_shared_walls: bool = True
    attic: str = "unheated"
    attic_ceiling_b: float = 0.8
    pitched_roof_area_ratio: float = 1.05
    lod2_window_ratio: float | None = None
    sfh_roof_fix: bool = True
    solar_beam: str = "horizontal"
    window_shading_factor: float = 0.6
    window_frame_fraction: float = 0.3
    blind_threshold_w_per_m2: float | None = None
    ground_floor_b: float = 0.5
    age_mapping: str = "stock_share"
    apartment_block_after_1978: str = "multi_family_house"
    air_change_per_h: float = 0.4
    ventilation_height_m: float | None = 2.5
    heating_limit_c: float | None = 12.0
    fixed_heating_off_days: tuple[int, int] | None = None
    retrofit_share_before_1979: float = 0.0
    retrofit_share_1979_1994: float = 0.0
    consumption_factor: str = "none"

    @classmethod
    def legacy(cls) -> "TeaserHeatConfig":
        return cls(preset="legacy", **_TEASER_LEGACY)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TeaserHeatConfig":
        raw = dict(_mapping(raw or {}, "asset_sizing.heat.teaser"))
        fields = cls.__dataclass_fields__
        _only(raw, set(fields), "asset_sizing.heat.teaser")
        preset = str(raw.pop("preset", "corrected"))
        if preset not in {"corrected", "legacy"}:
            raise ValueError("asset_sizing.heat.teaser.preset must be corrected or legacy.")
        base = cls.legacy() if preset == "legacy" else cls()
        values = {**asdict(base), **raw, "preset": preset}
        label = "asset_sizing.heat.teaser"
        choices = {
            "envelope_source": {"tabula", "lod2"},
            "attic": {"heated", "unheated"},
            "solar_beam": {"normal", "horizontal"},
            "age_mapping": {"band_end", "stock_share"},
            "apartment_block_after_1978": {"cap_1978", "multi_family_house"},
            "consumption_factor": {"none", "iwu"},
        }
        for name, allowed in choices.items():
            values[name] = str(values[name])
            if values[name] not in allowed:
                raise ValueError(f"{label}.{name} must be one of {sorted(allowed)}.")
        for name in ("subtract_shared_walls", "sfh_roof_fix"):
            if not isinstance(values[name], bool):
                raise ValueError(f"{label}.{name} must be true or false.")
        values["heated_area_ratio"] = _positive(values["heated_area_ratio"], f"{label}.heated_area_ratio")
        if values["heated_area_ratio"] > 1:
            raise ValueError(f"{label}.heated_area_ratio must be <= 1.")
        values["pitched_roof_area_ratio"] = _positive(values["pitched_roof_area_ratio"], f"{label}.pitched_roof_area_ratio")
        for name in ("window_shading_factor", "window_frame_fraction", "ground_floor_b", "attic_ceiling_b",
                     "retrofit_share_before_1979", "retrofit_share_1979_1994"):
            values[name] = _share(values[name], f"{label}.{name}")
        values["air_change_per_h"] = _positive(values["air_change_per_h"], f"{label}.air_change_per_h", allow_zero=True)
        for name in ("lod2_window_ratio", "blind_threshold_w_per_m2", "ventilation_height_m"):
            if values[name] is not None:
                values[name] = _positive(values[name], f"{label}.{name}")
        if values["heating_limit_c"] is not None:
            values["heating_limit_c"] = float(_number_or_none(values["heating_limit_c"], f"{label}.heating_limit_c"))
        days = values["fixed_heating_off_days"]
        if days is not None:
            if not isinstance(days, (list, tuple)) or len(days) != 2 or not all(isinstance(d, int) for d in days):
                raise ValueError(f"{label}.fixed_heating_off_days must be null or [first_day, end_day).")
            values["fixed_heating_off_days"] = (int(days[0]), int(days[1]))
        return cls(**values)


@dataclass(frozen=True)
class HeatSizingConfig:
    space_heat_source: str
    indoor_design_temperature_c: float
    heating_limit_temperature_c: float
    heat_pump_design_share: float
    buffer_volume_l_per_kw_th: float
    buffer_usable_temperature_spread_k: float
    # TABULA variant for the TEASER source: 0 standard (as built),
    # 1 usual refurbishment, 2 advanced refurbishment.
    teaser_retrofit_level: int = 0
    # Degree-day base of the full-load hours; None is the indoor temperature (G20/15),
    # the heating limit gives Heizgradtage G15.
    degree_day_base_temperature_c: float | None = None
    # technology: charge efficiency of technologies.storages.thermal_storage;
    # cop_curve: per-building COP penalty of charging by the usable spread.
    buffer_charge_efficiency_method: str = "technology"
    heated_area_fraction: float = 0.8
    internal: InternalHeatConfig | None = None
    # TEASER path assumptions, see TeaserHeatConfig (corrected unless the block says preset: legacy).
    teaser: TeaserHeatConfig = field(default_factory=TeaserHeatConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HeatSizingConfig":
        raw = _mapping(raw, "asset_sizing.heat")
        required = {
            "space_heat_source",
            "indoor_design_temperature_c", "heating_limit_temperature_c",
            "heat_pump_design_share", "buffer_volume_l_per_kw_th",
            "buffer_usable_temperature_spread_k",
        }
        optional = {
            "teaser_retrofit_level", "degree_day_base_temperature_c",
            "buffer_charge_efficiency_method", "heated_area_fraction", "internal", "teaser",
        }
        _only(raw, required | optional, "asset_sizing.heat")
        if not required <= set(raw):
            raise ValueError("asset_sizing.heat is incomplete.")
        retrofit = raw.get("teaser_retrofit_level", 0)
        if isinstance(retrofit, bool) or retrofit not in (0, 1, 2):
            raise ValueError("asset_sizing.heat.teaser_retrofit_level must be 0, 1 or 2.")
        source = str(raw["space_heat_source"])
        if source not in {"teaser", "infdb_ro_heat", "internal"}:
            raise ValueError(
                "asset_sizing.heat.space_heat_source must be teaser, infdb_ro_heat or internal."
            )
        fraction = _positive(raw.get("heated_area_fraction", 0.8), "asset_sizing.heat.heated_area_fraction")
        if fraction > 1 or (source == "internal" and fraction != 0.8):
            raise ValueError("Internal InfDB RC inputs already embed heated_area_fraction=0.8; another fraction requires regenerated coefficients.")
        internal = InternalHeatConfig.from_dict(raw.get("internal", {})) if source == "internal" else None
        if source != "internal" and raw.get("internal") is not None:
            raise ValueError("The internal heat block requires space_heat_source=internal.")
        # Like teaser_retrofit_level, the block is validated for every source and read by the teaser source only.
        teaser = TeaserHeatConfig.from_dict(raw.get("teaser"))
        inside = _positive(raw["indoor_design_temperature_c"], "asset_sizing.heat.indoor_design_temperature_c")
        limit = _positive(raw["heating_limit_temperature_c"], "asset_sizing.heat.heating_limit_temperature_c")
        if limit >= inside:
            raise ValueError("asset_sizing.heat.heating_limit_temperature_c must be below indoor_design_temperature_c.")
        share = _positive(raw["heat_pump_design_share"], "asset_sizing.heat.heat_pump_design_share")
        if share > 1.0:
            raise ValueError("asset_sizing.heat.heat_pump_design_share must be <= 1.")
        base = raw.get("degree_day_base_temperature_c")
        if base is not None:
            base = _positive(base, "asset_sizing.heat.degree_day_base_temperature_c")
            if not limit <= base <= inside:
                raise ValueError(
                    "asset_sizing.heat.degree_day_base_temperature_c must lie between the "
                    "heating limit and the indoor design temperature."
                )
        method = str(raw.get("buffer_charge_efficiency_method", "technology"))
        if method not in {"technology", "cop_curve"}:
            raise ValueError(
                "asset_sizing.heat.buffer_charge_efficiency_method must be technology or cop_curve."
            )
        if source == "internal" and method == "cop_curve":
            # Internal tank charging already uses its own COP at the raised sink temperature.
            raise ValueError(
                "space_heat_source internal charges the tank through its own COP route; "
                "buffer_charge_efficiency_method must be technology."
            )
        return cls(
            space_heat_source=source,
            indoor_design_temperature_c=inside,
            heating_limit_temperature_c=limit,
            heat_pump_design_share=share,
            buffer_volume_l_per_kw_th=_positive(raw["buffer_volume_l_per_kw_th"], "asset_sizing.heat.buffer_volume_l_per_kw_th"),
            buffer_usable_temperature_spread_k=_positive(raw["buffer_usable_temperature_spread_k"], "asset_sizing.heat.buffer_usable_temperature_spread_k"),
            teaser_retrofit_level=int(retrofit),
            degree_day_base_temperature_c=base,
            buffer_charge_efficiency_method=method,
            heated_area_fraction=fraction,
            internal=internal,
            teaser=teaser,
        )

    def sizing_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of ``build_heat_asset_plan`` for this configuration."""
        return {
            "indoor_design_temperature_c": self.indoor_design_temperature_c,
            "heating_limit_temperature_c": self.heating_limit_temperature_c,
            "heat_pump_design_share": self.heat_pump_design_share,
            "buffer_volume_l_per_kw_th": self.buffer_volume_l_per_kw_th,
            "buffer_usable_temperature_spread_k": self.buffer_usable_temperature_spread_k,
            "degree_day_base_temperature_c": self.degree_day_base_temperature_c,
            "buffer_charge_efficiency_method": self.buffer_charge_efficiency_method,
        }


@dataclass(frozen=True)
class BatterySizingConfig:
    heuristic_method: str
    optimized_method: str
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
            "heuristic_method", "optimized_method",
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
    # HEMS spreads each charging session over its stay: at most this factor times
    # the session's average required power, except in hours of local PV surplus.
    # None: no limit beyond the charger rating. InFlex charging is unaffected.
    hems_session_power_factor: float | None = None

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
        _only(raw, allowed | {"hems_session_power_factor"}, "mobility")
        factor = raw.get("hems_session_power_factor")
        if factor is not None:
            factor = _positive(factor, "mobility.hems_session_power_factor")
            if factor < 1.0:
                raise ValueError("mobility.hems_session_power_factor must be at least 1 (sessions stay feasible).")
        probability = _positive(raw["commuting_probability"], "mobility.commuting_probability", allow_zero=True)
        if probability > 1:
            raise ValueError("mobility.commuting_probability must be <= 1.")
        cycle = str(raw["driving_cycle_type"])
        if cycle not in {"WLTC", "EPA"}:
            raise ValueError("mobility.driving_cycle_type must be WLTC or EPA.")
        reference_year = int(_positive(raw["reference_year"], "mobility.reference_year"))
        if reference_year != REFERENCE_YEAR:
            # The time axis (timeframes, DST hours, weather, run time stamps) is
            # the fixed calendar year gridexpand.common.timeframe.REFERENCE_YEAR.
            raise ValueError(
                f"mobility.reference_year must be {REFERENCE_YEAR} (the fixed calendar "
                f"year of every GridExpand time axis), got {reference_year}."
            )
        return cls(
            commuting_probability=probability,
            emobpy_timestep_hours=_positive(raw["emobpy_timestep_hours"], "mobility.emobpy_timestep_hours"),
            reference_year=reference_year,
            passenger_mass_kg=_positive(raw["passenger_mass_kg"], "mobility.passenger_mass_kg"),
            passenger_sensible_heat_w=_positive(raw["passenger_sensible_heat_w"], "mobility.passenger_sensible_heat_w", allow_zero=True),
            passengers_per_vehicle=_positive(raw["passengers_per_vehicle"], "mobility.passengers_per_vehicle"),
            cabin_heat_transfer_coefficient=_positive(raw["cabin_heat_transfer_coefficient"], "mobility.cabin_heat_transfer_coefficient"),
            cabin_air_flow_m3_per_s=_positive(raw["cabin_air_flow_m3_per_s"], "mobility.cabin_air_flow_m3_per_s", allow_zero=True),
            driving_cycle_type=cycle,
            road_type=int(_positive(raw["road_type"], "mobility.road_type", allow_zero=True)),
            road_slope=float(_number_or_none(raw["road_slope"], "mobility.road_slope")),
            hems_session_power_factor=factor,
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
    economics: EconomicsConfig
    pv: PvSizingConfig
    battery: BatterySizingConfig
    heat: HeatSizingConfig
    mobility: MobilityConfig
    electrification: ElectrificationConfig
    technologies: TechnologyParameters
    time_aggregation: TimeAggregationConfig
    # Optional ``ghd:`` block (GHD demand rules, all off by default; gridexpand.common.ghd).
    ghd: GhdConfig = field(default_factory=GhdConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ScenarioConfig":
        raw = _mapping(raw, "scenario configuration")
        _only(raw, {"scenario", "economics", "asset_sizing", "mobility", "electrification", "technologies", "time_aggregation", "ghd"}, "top-level scenario")
        scenario = _mapping(raw["scenario"], "scenario")
        _only(scenario, {"id", "milestone_year"}, "scenario")
        economics = _mapping(raw["economics"], "economics")
        _only(economics, {"electricity"}, "economics")
        assets = _mapping(raw["asset_sizing"], "asset_sizing")
        _only(assets, {"pv", "battery", "heat"}, "asset_sizing")
        return cls(
            scenario_id=str(scenario["id"]),
            milestone_year=int(_positive(scenario["milestone_year"], "scenario.milestone_year")),
            economics=EconomicsConfig.from_dict(economics["electricity"]),
            pv=PvSizingConfig.from_dict(assets["pv"]),
            battery=BatterySizingConfig.from_dict(assets["battery"]),
            heat=HeatSizingConfig.from_dict(assets["heat"]),
            mobility=MobilityConfig.from_dict(raw["mobility"]),
            electrification=ElectrificationConfig.from_dict(raw["electrification"]),
            technologies=TechnologyParameters.from_dict(raw["technologies"]),
            time_aggregation=TimeAggregationConfig.from_dict(raw["time_aggregation"]),
            ghd=GhdConfig.from_dict(raw.get("ghd")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the typed scientific configuration in canonical field form."""
        return asdict(self)

    def pv_sizing_method(self, model_case: str) -> str:
        plan = get_model_case(model_case).asset_plan
        if plan == "optimization":
            return self.pv.optimized_method
        if plan == "none":
            return "none"
        return self.pv.heuristic_method

    def battery_sizing_method(self, model_case: str) -> str:
        plan = get_model_case(model_case).asset_plan
        if plan == "optimization":
            return self.battery.optimized_method
        if plan == "none":
            return "none"
        return self.battery.heuristic_method

    def battery_capacity_coefficients(self, model_case: str) -> tuple[float, float]:
        """Return PV- and demand-based battery coefficients for one model case."""
        plan = get_model_case(model_case).asset_plan
        if plan == "optimization":
            return (
                self.battery.optimized_upper_kwh_per_pv_kwp,
                self.battery.optimized_upper_kwh_per_annual_mwh,
            )
        if plan == "none":
            return (0.0, 0.0)
        return (
            self.battery.heuristic_usable_kwh_per_pv_kwp,
            self.battery.heuristic_usable_kwh_per_annual_mwh,
        )

    def heat_sizing_method(self, model_case: str) -> str:
        plan = get_model_case(model_case).asset_plan
        if plan == "optimization":
            return "optimization"
        if plan == "none":
            return "none"
        return "full_load_hours_rule"
