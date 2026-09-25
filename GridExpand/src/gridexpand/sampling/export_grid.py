"""Shared pylovo grid export helpers for Step 1."""

from gridexpand.common.building_components import build_building_components
import pandas as pd

import gridexpand.sampling.grid_topol as grdtpl
import gridexpand.sampling.save_grid as svgrd
import gridexpand.common.weather as wth


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
    df_weather, altitude = wth.get_pvgis_tmy_sarah3_dataframe(lat, lon)
    df_weather["dew_point"] = wth.get_dew_point(
        df_weather["temp_air"], df_weather["relative_humidity"]
    )
    return df_weather, float(altitude)


def export_pylovo_grid(db, grid_specs: dict, region_specs, skip_weather: bool = False) -> str:
    """Export one pylovo grid to the Step-1 HDF5 raw-data format."""
    grid_specs = dict(grid_specs)
    df_region = _as_region_frame(region_specs)

    net = db.read_single_ppgrid(grid_specs)
    net = grdtpl.assign_min_linelen(net)
    net = grdtpl.normalize_scenario_loads(net)

    df_buildings = db.read_buildings(grid_specs, net.bus)
    df_building_components = build_building_components(df_buildings)

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
    save_file.save_df(df_building_components, "/raw_data/building_components")
    if df_weather is not None:
        save_file.save_df(df_weather, "/raw_data/weather")

    return save_file.path
