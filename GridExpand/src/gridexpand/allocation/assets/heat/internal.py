"""Prepare internal 1R1C inputs from corrected ro-heat coefficients.

The source embeds an 80% zone. Only the residential component share is applied
at import; R/C are never multiplied by the heated-area fraction a second time.
No TEASER heat profile or COP library is used by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import os
from functools import cache
from importlib.metadata import version

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

from gridexpand.common.reproducibility import (
    stable_seed,
    legacy_random_state,
    frame_fingerprint,
)
from gridexpand.common.thermal import thermostat_reference
from gridexpand.db.engine import get_engine
from gridexpand.allocation.config import config
from gridexpand.allocation.functions.heat import (
    get_norm_outside_temperature,
    sample_statistics,
    site_data,
)
from gridexpand.allocation.functions.infdb_ro_heat import generate_opendhw
from ..urbs_rows import process_row, storage_parameter_fields
from .sizing import build_heat_asset_plan

MODEL_VERSION = "implicit_euler_1r1c_v1"


@cache
def generator_fingerprint():
    """Invalidate cached physics when its code or stochastic dependencies change."""
    import gridexpand.common.thermal as thermal
    from gridexpand.allocation.functions import dst
    from gridexpand.allocation.external.districtgenerator.classes import solar, profils
    from gridexpand.allocation import config as configuration

    digest = hashlib.sha256()
    for module_path in (
        __file__,
        thermal.__file__,
        dst.__file__,
        solar.__file__,
        profils.__file__,
        configuration.__file__,
    ):
        digest.update(Path(module_path).read_bytes())
    for package in ("OpenDHW", "richardsonpy", "numpy"):
        digest.update(f"{package}:{version(package)}".encode())
    return digest.hexdigest()


def load_rc(building_ids, *, engine=None):
    """Read traceable corrected SI coefficients and stored envelope geometry."""
    ids = sorted(set(map(str, building_ids)))
    if not ids:
        return pd.DataFrame(
            columns=[
                "building_objectid",
                "resistance",
                "capacitance",
                "changelog_id",
                "residential_component_area_m2",
            ]
        )
    sql = text("""SELECT r.building_objectid,r.resistance,r.capacitance,r.changelog_id,
        c.modified_at::text AS rc_modified_at,c.comment AS rc_source_comment,
        s.floor_area AS source_footprint_m2,s.floor_number AS source_floor_number,
        s.building_type AS rc_building_type,s.construction_year AS rc_construction_year,
        s.outer_wall AS wall_refurbishment_year,s.rooftop AS roof_refurbishment_year,
        s.window AS window_refurbishment_year,s.window_area AS source_window_area_m2,
        b.residential_floor_area AS residential_component_area_m2,b.households,b.occupants
        FROM ro_heat.buildings_rc r
        JOIN ro_heat.buildings_refurbished_status s USING(building_objectid)
        JOIN basedata.buildings b ON b.objectid=r.building_objectid
        JOIN public.changelog c ON c.id=r.changelog_id
        WHERE r.building_objectid IN :ids""").bindparams(
        bindparam("ids", expanding=True)
    )
    with (engine if engine is not None else get_engine()).connect() as con:
        frame = pd.read_sql(sql, con, params={"ids": ids})
    missing = sorted(set(ids) - set(frame.building_objectid.astype(str)))
    if missing:
        raise ValueError(f"Internal heat lacks RC/envelope rows for {missing[:10]}.")
    if frame.building_objectid.duplicated().any():
        raise ValueError("Duplicate internal RC records.")
    if (
        not np.isfinite(frame[["resistance", "capacitance"]].to_numpy()).all()
        or (frame[["resistance", "capacitance"]] <= 0).any().any()
    ):
        raise ValueError("Internal RC records must be finite and positive.")
    if (
        not frame.rc_source_comment.fillna("")
        .str.contains("specific heat converted from kJ", regex=False)
        .all()
    ):
        raise ValueError(
            "Internal heat requires corrected RC provenance documenting the TABULA kJ-to-J conversion."
        )
    return frame


def normalized_parameters(rc, buildings, heat_config):
    """Normalize SI coefficients, residential shares and once-only area provenance."""
    physical_columns = [
        name
        for name in (
            "building_objectid",
            "Site",
            "residential_effective_floor_area_m2",
            "households",
            "occupants",
            "occ_list",
            "number_of_households",
        )
        if name in buildings
    ]
    result = buildings[physical_columns].merge(
        rc, on="building_objectid", validate="one_to_one", suffixes=("", "_source")
    )
    area = pd.to_numeric(result["residential_effective_floor_area_m2"], errors="raise")
    gross = result.source_footprint_m2 * result.source_floor_number
    share = area / gross
    if not np.isfinite(share).all() or (share <= 0).any() or (share > 1 + 1e-6).any():
        raise ValueError(
            "Internal RC residential area must be positive and no larger than source gross area."
        )
    result["residential_component_area_m2"] = area
    result["thermal_area_m2"] = area * heat_config.heated_area_fraction
    result["residential_rc_share"] = share
    result["source_heated_area_fraction"] = 0.8
    result["target_heated_area_fraction"] = heat_config.heated_area_fraction
    result["heated_fraction_embedded_in_rc"] = True
    result["transmission_conductance_kw_per_k"] = share / result.resistance / 1000
    result["capacitance_kwh_per_k"] = result.capacitance * share / 3_600_000
    settings = heat_config.internal
    result["ventilation_conductance_kw_per_k"] = (
        1.2
        * 1000
        * settings.ventilation_air_changes_per_hour
        * settings.zone_height_m
        * result.thermal_area_m2
        / 3600
        / 1000
    )
    result["conductance_kw_per_k"] = (
        result.transmission_conductance_kw_per_k
        + result.ventilation_conductance_kw_per_k
    )
    result["window_area_m2"] = (
        result.source_window_area_m2 * share * heat_config.heated_area_fraction
    )
    result["model_version"] = MODEL_VERSION
    result["generator_sha256"] = generator_fingerprint()
    result["refurbishment_state"] = settings.refurbishment_state
    result["source_resistance_unit"] = "K/W"
    result["source_capacitance_unit"] = "J/K"
    result["heat_commodity"] = "room_heat_" + result.building_objectid.astype(str)
    return result


def service_cops(heating_type, ambient, buffer_spread_k):
    """Separate space, direct DHW and tank charging COPs.

    The tank is charged ``buffer_spread_k`` (the usable buffer spread) above the
    normal space-heating sink temperature.
    """
    ambient = np.asarray(ambient, float)
    sink = 40 - ambient if heating_type == "radiator" else 30 - 0.5 * ambient
    return tuple(
        np.asarray(config.ASHP_COP(np.maximum(delta, 15)), float).ravel()
        for delta in (sink - ambient, 50 - ambient, sink + buffer_spread_k - ambient)
    )


def solar_irradiance(weather, postcode):
    """Reuse DistrictGenerator's cardinal-plane solar calculation, no heat simulation."""
    from gridexpand.allocation.external.districtgenerator.classes.solar import Sun

    site = site_data().loc[site_data().Zip.eq(str(postcode).zfill(5))]
    if len(site) != 1:
        raise ValueError(f"Missing solar site for postcode {postcode}.")
    row = site.iloc[0]
    return Sun(filePath=config.DISTGEN_DATA_PATH).getSolarGains(
        initialTime=0,
        timeDiscretization=3600,
        timeSteps=len(weather),
        timeZone=1,
        location=[row.Latitude, row.Longitude],
        altitude=row.Altitude,
        beta=[90] * 4 + [0],
        gamma=[0, 90, 180, 270, 0],
        beamRadiation=weather.dni.to_numpy(),
        diffuseRadiation=weather.dhi.to_numpy(),
        albedo=0.2,
    )[:4]


def household_occupants(building, seed):
    """Residents per household with the household-size rule of the TEASER heat path.

    The households are sampled from the destatis size distribution as in
    ``electricity._assign_household_occupancy`` (same seed parts) and rounded as
    for TEASER's occupancy, so no household has more than five residents.
    """
    import random

    from gridexpand.allocation.functions.electricity import _get_occupancy_distribution
    from gridexpand.common.reproducibility import physical_building_id

    sizes = config.HH_SIZE_DISTRIBUTION
    rng = random.Random(
        stable_seed(seed, physical_building_id(building), "Residential", "electricity", "occupancy")
    )
    sampled = _get_occupancy_distribution(
        dict(zip(sizes["size"], sizes["probability"])),
        max(1, int(round(float(building["households"])))),
        max(0.0, float(building["occupants"])),
        rng=rng,
    )
    return [int(round(size)) for size in sampled]


def internal_gains(building, electricity, settings, seed, hours):
    """70 W per present resident plus the existing 0.362 household-electricity share."""
    from gridexpand.allocation.external.districtgenerator.classes.profils import (
        Profiles,
    )

    occupancy = np.zeros(hours)
    occ = building.get("occ_list")
    if not isinstance(occ, (list, tuple, np.ndarray)):
        occ = household_occupants(building, seed)
    with legacy_random_state(
        stable_seed(seed, building["building_objectid"], "internal_heat", "occupancy")
    ):
        for residents in occ:
            if residents == 0:
                continue
            if residents > 5:
                raise ValueError(
                    "Richardson occupancy supports at most five residents per household."
                )
            profile = Profiles(
                residents, sum(occ), 4, hours // 24, 3600, building["rc_building_type"]
            )
            occupancy += profile.generate_occupancy_profiles_residential()
    from gridexpand.allocation.functions.dst import dst_shift_output

    occupancy = dst_shift_output(
        pd.DataFrame({"occupancy": occupancy})
    ).occupancy.to_numpy()
    return (
        occupancy * settings.person_gain_w / 1000
        + np.asarray(electricity, float) * settings.electricity_gain_fraction
    )


@dataclass
class InternalHeatInputs:
    demand: pd.DataFrame
    eff_factor: pd.DataFrame
    process: pd.DataFrame
    commodity: pd.DataFrame
    process_commodity: pd.DataFrame
    storage: pd.DataFrame
    audit: pd.DataFrame
    parameters: pd.DataFrame
    timeseries: pd.DataFrame
    reference: pd.DataFrame
    asset_plan: pd.DataFrame
    metadata: dict


def prepare_internal_heat(
    buildings,
    weather,
    postcode,
    *,
    heat_config,
    technologies,
    sizing_method,
    seed,
    electricity_by_building,
    rc=None,
):
    """Produce physical thermal trajectories, assets and generic urbs heat routing."""
    if len(weather) != 8760 or not {"temp_air", "dni", "dhi"}.issubset(weather):
        raise ValueError(
            "Internal heat preparation requires complete chronological 8760-hour temperature/DNI/DHI weather."
        )
    if buildings.building_objectid.duplicated().any():
        raise ValueError("Internal preparation requires unique physical building IDs.")
    rc = load_rc(buildings.building_objectid) if rc is None else rc
    params = normalized_parameters(rc, buildings, heat_config)
    params = sample_statistics(
        params.assign(
            objectid=params.building_objectid,
            construction_year=params.rc_construction_year.astype(str),
        ),
        seed,
    )
    irradiance = solar_irradiance(weather, postcode)
    settings = heat_config.internal
    shaded = np.where(
        irradiance > settings.blind_irradiance_threshold_w_per_m2,
        irradiance * settings.closed_blind_transmittance,
        irradiance,
    )
    demand_entries, series_entries, reference_entries, cops = {}, {}, {}, {}
    per_building_plans = []
    from gridexpand.paths import ALLOCATION_RESULTS_DIR

    cache_dir = ALLOCATION_RESULTS_DIR / "internal_heat_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weather_hash = frame_fingerprint(weather[["temp_air", "dni", "dhi"]])
    fingerprints = {}
    for row in params.to_dict("records"):
        bid, site = str(row["building_objectid"]), int(row["Site"])
        occ = row.get("occ_list")
        if not isinstance(occ, (list, tuple, np.ndarray)):
            row["occ_list"] = household_occupants(row, seed)
        row["households"] = len(row["occ_list"])
        row["number_of_households"] = len(row["occ_list"])
        cache_parameters = {
            k: v for k, v in row.items() if k not in {"Site", "bus", "objectid"}
        }
        cache_settings = {
            k: v for k, v in asdict(settings).items() if k != "hems_preheat_uplift_k"
        }
        payload = json.dumps(
            {
                "model": MODEL_VERSION,
                "parameters": cache_parameters,
                "settings": cache_settings,
                "buffer_usable_temperature_spread_k": heat_config.buffer_usable_temperature_spread_k,
                "postcode": str(postcode),
                "seed": seed,
                "weather": weather_hash,
                "electricity": frame_fingerprint(
                    pd.DataFrame({"electricity": electricity_by_building[bid]})
                ),
            },
            sort_keys=True,
            default=str,
        )
        cache_key = hashlib.sha256(payload.encode()).hexdigest()
        cache_path = cache_dir / f"{cache_key}.npz"
        fingerprints[bid] = cache_key
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as saved:
                gain, solar, heat, temp, water, cop_space, cop_water, cop_buffer = (
                    saved[key]
                    for key in (
                        "gain",
                        "solar",
                        "heat",
                        "temperature",
                        "water",
                        "space_cop",
                        "water_cop",
                        "buffer_cop",
                    )
                )
                initial = float(saved["initial"])
        else:
            gain = internal_gains(
                row, electricity_by_building[bid], settings, seed, len(weather)
            )
            solar = (
                shaded.sum(axis=0)
                * row["window_area_m2"]
                / 4
                * settings.glazing_solar_transmittance
                / 1000
            )
            heat, temp, initial = thermostat_reference(
                weather.temp_air,
                gain + solar,
                conductance_kw_per_k=row["conductance_kw_per_k"],
                capacitance_kwh_per_k=row["capacitance_kwh_per_k"],
                minimum_temperature_c=settings.minimum_temperature_c,
            )
            water_building = pd.DataFrame(
                [
                    dict(
                        row,
                        bus=site,
                        objectid=bid,
                        building_type=row["rc_building_type"],
                    )
                ]
            )
            from gridexpand.allocation.functions.dst import dst_shift_output

            water = dst_shift_output(generate_opendhw(water_building, base_seed=seed))[
                site, "water_heat"
            ].to_numpy()
            cop_space, cop_water, cop_buffer = service_cops(
                row["heating_type"],
                weather.temp_air,
                heat_config.buffer_usable_temperature_spread_k,
            )
            with tempfile.NamedTemporaryFile(
                dir=cache_dir, suffix=".npz", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                np.savez_compressed(
                    temporary_path,
                    gain=gain,
                    solar=solar,
                    heat=heat,
                    temperature=temp,
                    water=water,
                    space_cop=cop_space,
                    water_cop=cop_water,
                    buffer_cop=cop_buffer,
                    initial=initial,
                )
                os.replace(temporary_path, cache_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        for field, values in {
            "outside_temperature_c": weather.temp_air.to_numpy(),
            "internal_gains_kw": gain,
            "solar_gains_kw": solar,
            "minimum_temperature_c": np.full(
                len(weather), settings.minimum_temperature_c
            ),
            "upper_temperature_c": np.maximum(
                settings.minimum_temperature_c + settings.hems_preheat_uplift_k, temp
            ),
        }.items():
            series_entries[bid, field] = values
        for field, values in {
            "space_heat_kw": heat,
            "water_heat_kw": water,
            "temperature_c": temp,
            "space_cop": cop_space,
            "water_cop": cop_water,
            "buffer_cop": cop_buffer,
        }.items():
            reference_entries[bid, field] = values
        params.loc[params.building_objectid.eq(bid), "initial_temperature_c"] = initial
        params.loc[params.building_objectid.eq(bid), "terminal_temperature_c"] = initial
        # Each physical building is sized separately even on a shared electrical bus.
        space_frame = pd.DataFrame({(site, "space_heat"): heat})
        water_frame = pd.DataFrame({(site, "water_heat"): water})
        cop_frame = pd.DataFrame({(site, "heatpump_air"): cop_space})
        plan, _ = build_heat_asset_plan(
            pd.DataFrame([dict(row, floor_area=row["thermal_area_m2"])]),
            space_frame,
            water_frame,
            cop_frame,
            weather.temp_air,
            sizing_method=sizing_method,
            norm_outside_temperature_c=get_norm_outside_temperature(postcode),
            **heat_config.sizing_kwargs(),
            water_heat_pump_cop=pd.DataFrame({(site, "heatpump_air"): cop_water}),
        )
        per_building_plans.append(plan)
        for commodity, values in (("space_heat", heat), ("water_heat", water)):
            key = site, commodity
            demand_entries[key] = demand_entries.get(key, 0) + values
        for name, values in (
            (f"HP_space_{bid}", cop_space),
            (f"HP_water_{bid}", cop_water),
            (f"HP_buffer_{bid}", cop_buffer),
        ):
            cops[site, name] = values
    plan = pd.concat(per_building_plans, ignore_index=True)
    process, commodity, ratios, storage = materialize_internal_assets(
        plan, technologies, sizing_method
    )
    params["room_heat_upper_kw"] = params.building_objectid.map(
        plan.set_index("building_objectid").heat_conversion_capacity_kw_th
    )
    series = pd.DataFrame(series_entries)
    reference = pd.DataFrame(reference_entries)
    params["input_fingerprint"] = params.building_objectid.map(fingerprints)
    audit = plan.assign(sector="heat", audit_record_type="heat_asset_plan")
    return InternalHeatInputs(
        pd.DataFrame(demand_entries),
        pd.DataFrame(cops),
        process,
        commodity,
        ratios,
        storage,
        audit,
        params,
        series,
        reference,
        plan,
        {
            "space_heat_source": "internal",
            "thermal_model": MODEL_VERSION,
            "heated_area_fraction": 0.8,
            "rc_heated_area_fraction_applied_again": False,
            "refurbishment_state": "stored_ro_heat",
            "internal_heat_settings": asdict(settings),
            "thermal_input_fingerprints": dict(
                zip(params.building_objectid, params.input_fingerprint)
            ),
        },
    )


def materialize_internal_assets(plan, technologies, sizing_method):
    """One electrical HP/rod investment per site; service routes share its input.

    Intermediate commodities carry allocated electrical input, not heat. Tank
    charging has its own COP and cannot charge from the direct-room heat bus.
    """
    fixed = sizing_method == "full_load_hours_rule"
    processes, commodities, ratios, storages = [], [], [], []

    def route(site, name, cin, cout, capacity, parameters=None, installed=None):
        parameters = (
            technologies.processes["heat_dummy"] if parameters is None else parameters
        )
        processes.append(
            process_row(
                site,
                name,
                capacity if installed is None else installed,
                capacity,
                fixed=installed is None or fixed,
                parameters=parameters,
            )
        )
        ratios.extend(
            [
                {"Process": name, "Commodity": cin, "Direction": "In", "ratio": 1.0},
                {"Process": name, "Commodity": cout, "Direction": "Out", "ratio": 1.0},
            ]
        )

    for site, group in plan.groupby("Site"):
        for parent, stem, installed_key, upper_key in (
            (
                "heatpump_air",
                "hp",
                "heat_pump_installed_kw_el",
                "heat_pump_capacity_upper_kw_el",
            ),
            (
                "heatpump_booster",
                "rod",
                "auxiliary_installed_kw_el",
                "auxiliary_capacity_upper_kw_el",
            ),
        ):
            route(
                site,
                parent,
                "electricity",
                f"{stem}_allocated_electricity",
                float(group[upper_key].sum()),
                technologies.processes[parent],
                float(group[installed_key].sum()),
            )
            commodities.append(
                {
                    "Site": site,
                    "Commodity": f"{stem}_allocated_electricity",
                    "Type": "Stock",
                    "price": np.nan,
                }
            )
        commodities.append(
            {"Site": site, "Commodity": "water_heat", "Type": "Demand", "price": np.nan}
        )
        for row in group.to_dict("records"):
            bid = row["building_objectid"]
            room = f"room_heat_{bid}"
            tank = f"tank_heat_{bid}"
            upper = row["heat_conversion_capacity_kw_th"]
            commodities.extend(
                [
                    {"Site": site, "Commodity": room, "Type": "Stock", "price": np.nan},
                    {"Site": site, "Commodity": tank, "Type": "Stock", "price": np.nan},
                ]
            )
            for service, output in (
                ("space", room),
                ("water", "water_heat"),
                ("buffer", tank),
            ):
                route(
                    site,
                    f"HP_{service}_{bid}",
                    "hp_allocated_electricity",
                    output,
                    upper,
                )
            for service, output in (("space", room), ("water", "water_heat")):
                route(
                    site,
                    f"Rod_{service}_{bid}",
                    "rod_allocated_electricity",
                    output,
                    upper,
                )
            # All tank output is room heat; no tank bypass or reverse room charging.
            route(site, f"Tank_delivery_{bid}", tank, room, upper)
            energy = row["buffer_capacity_upper_kwh_th"]
            power = row["buffer_power_upper_kw_th"]
            if energy > 0 and power > 0:
                storage = {
                    "Site": site,
                    "Storage": f"heat_storage_{bid}",
                    "Commodity": tank,
                    "inst-cap-c": row["buffer_installed_kwh_th"],
                    "cap-up-c": energy,
                    "inst-cap-p": row["buffer_installed_power_kw_th"],
                    "cap-up-p": power,
                    **storage_parameter_fields(
                        technologies.storages["thermal_storage"],
                        fixed=fixed,
                        ep_ratio=energy / power,
                    ),
                    # The HP_buffer COP already carries the raised-sink penalty; the
                    # standing loss stays the configured tank loss.
                    "eff-in": 1.0,
                }
                if not fixed:
                    storage.update(
                        {
                            "linked-process": "heatpump_air",
                            "max-energy-per-process-capacity": energy
                            / float(group.heat_pump_capacity_upper_kw_el.sum()),
                        }
                    )
                storages.append(storage)
    return (
        pd.DataFrame(processes),
        pd.DataFrame(commodities),
        pd.DataFrame(ratios).drop_duplicates(),
        pd.DataFrame(storages),
    )
