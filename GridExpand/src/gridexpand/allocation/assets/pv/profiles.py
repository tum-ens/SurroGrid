"""pvlib profile generation for shared LoD2 angle bins."""

from __future__ import annotations

import pandas as pd
from pvlib import location, modelchain, pvsystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

try:
    from config import config
except ModuleNotFoundError:
    from ....config import config


from .labels import profile_label

def calculate_pv_power(lat, lon, altitude, tilt, azimuth, weather_df):
    """Return AC watts for the repository's normalized 1 kWp PV system."""
    weather = weather_df.copy()
    weather.index = weather["time(inst)"]
    site = location.Location(latitude=lat, longitude=lon, altitude=altitude, tz="Etc/GMT-1")
    system = pvsystem.PVSystem(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        module_parameters=config.MODULE_PARAMETERS,
        inverter_parameters=config.INVERTER_PARAMETERS,
        temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"],
        losses_parameters=config.SOLAR_LOSSES,
        albedo=config.ALBEDO,
    )
    chain = modelchain.ModelChain(
        system, site, aoi_model="physical", spectral_model="no_loss",
        temperature_model="sapm", dc_model="pvwatts", ac_model="pvwatts",
        losses_model="pvwatts",
    )
    chain.run_model(weather)
    return chain.results.ac


def generate_profile_library(catalog, weather, location_data, altitude) -> pd.DataFrame:
    """Generate each required normalized binned profile once."""
    usable = catalog[catalog["profile_usable"]]
    angles = sorted(set(usable[["profile_tilt_deg", "profile_azimuth_deg"]].itertuples(index=False, name=None)))
    result = {}
    for tilt, azimuth in angles:
        result[profile_label(tilt, azimuth)] = (
            calculate_pv_power(location_data["lat"], location_data["lon"], altitude, tilt, azimuth, weather)
            .reset_index(drop=True)
            .fillna(0.0)
            / 1000.0
        )
    frame = pd.DataFrame(result)
    frame.index.name = "t"
    return frame
