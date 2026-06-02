"""Shared pylovo grid export helpers for Step 1."""

import pandas as pd

import src.grid_topol as grdtpl
import src.save_grid as svgrd
import src.weather as wth


def build_region_row(db, plz: int, kcid: int, bcid: int) -> pd.DataFrame:
    """Build minimal region metadata for a directly selected pylovo grid."""
    grid_specs = {"plz": plz, "kcid": kcid, "bcid": bcid}

    df_region = db.read_regional_stats(plz)
    if df_region.empty:
        df_region = pd.DataFrame([{"plz": plz}])
    else:
        df_region = df_region.iloc[[0]].copy().reset_index(drop=True)

    location = db.read_trafo_pos(grid_specs)
    df_region["lat"] = location["lat"]
    df_region["lon"] = location["lon"]
    df_region["kcid"] = kcid
    df_region["bcid"] = bcid

    return df_region


def _as_region_frame(region_specs) -> pd.DataFrame:
    if isinstance(region_specs, pd.Series):
        return region_specs.to_frame().T.reset_index(drop=True)
    if isinstance(region_specs, pd.DataFrame):
        return region_specs.copy().reset_index(drop=True)
    return pd.DataFrame([region_specs])


def _read_weather(lat: float, lon: float) -> tuple[pd.DataFrame, float]:
    weather_tuple = wth.get_pvgis_tmy_sarah3_dataframe(lat, lon)
    if weather_tuple is None:
        raise RuntimeError("PVGIS weather retrieval failed.")

    df_weather, altitude, selected_months = weather_tuple
    df_weather["dew_point"] = wth.get_dew_point(
        df_weather["temp_air"], df_weather["relative_humidity"]
    )

    df_soil = wth.get_open_meteo_soil_temperature(lat, lon, selected_months)
    soil_series = df_soil.iloc[:, 0] if isinstance(df_soil, pd.DataFrame) else df_soil
    df_weather["soil_temp"] = pd.Series(soil_series).reset_index(drop=True)

    return df_weather, float(altitude)


def export_pylovo_grid(db, grid_specs: dict, region_specs, skip_weather: bool = False) -> str:
    """Export one pylovo grid to the Step-1 HDF5 raw-data format."""
    grid_specs = dict(grid_specs)
    df_region = _as_region_frame(region_specs)

    net = db.read_single_ppgrid(grid_specs)
    net = grdtpl.assign_min_linelen(net)
    net = grdtpl.remove_duplicate_loads(net)

    df_buildings = db.read_buildings(grid_specs, net.bus)

    df_weather = None
    if not skip_weather:
        lat = float(df_region.iloc[0]["lat"])
        lon = float(df_region.iloc[0]["lon"])
        df_weather, altitude = _read_weather(lat, lon)
        df_region["altitude"] = altitude

    save_file = svgrd.SaveFile(grid_specs)
    save_file.save_topology(net, "/raw_data/")
    save_file.save_df(df_region, "/raw_data/region")
    save_file.save_df(df_buildings, "/raw_data/buildings")
    if df_weather is not None:
        save_file.save_df(df_weather, "/raw_data/weather")

    return save_file.path
