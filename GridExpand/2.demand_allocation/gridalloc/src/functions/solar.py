from config import config
import pandas as pd
import numpy as np

from pvlib import pvsystem, modelchain, location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS


##############################################################
################### Sampling + Generation ####################
##############################################################

def _get_roof_sections(area, tilt_hist):
    ### First sample whether flat roof:
    roof_type = np.random.choice(["flat","gabled"], p=[config.PROB_FLAT_ROOF,config.PROB_GABLED_ROOF])

    ### If flat roof return one flat roof section
    if roof_type == "flat":
        A_eff = area*config.FLAT_ROOF_UTILIZATION
        cap = round(A_eff*config.PV_AREA_FACTOR, 3)
        return [(cap,0,0)]
    ### Else assume two opposing gabled roof sections
    else:
        # Sample azimuth
        azimuth_1 = int(round(np.random.uniform(0, 180),0))
        azimuth_2 = azimuth_1 + 180                         # Opposing roof section  
    
        # Sample tilt
        u = np.random.uniform()
        bin_idx = tilt_hist['cum_prob'].searchsorted(u)     # Find the bin corresponding to the cumulative probability
        row = tilt_hist.iloc[bin_idx]
        tilt = int(round(np.random.uniform(row['bin_left'], row['bin_right']),0)) # Sample uniformly within the selected bin
        
        A_eff = round((area/2)*config.SLANTED_ROOF_UTILIZATION/np.cos(np.radians(tilt)),3)
        cap = round(A_eff*config.PV_AREA_FACTOR, 3)
        return [(cap,tilt,azimuth_1),(cap,tilt,azimuth_2)]

def _calculate_pv_power(lat, lon, altitude, tilt, azimuth, weather_df):
    """
    Calculate AC power output for a 1 kW PV panel with standard German system settings,
    using a DataFrame containing time-series weather data.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.
    altitude : float
        Altitude in meters.
    tilt : float
        Panel tilt angle in degrees from horizontal.
    azimuth : float
        Panel azimuth angle in degrees (180=south, 90=east, 270=west, 0=north).
    weather_df : pandas.DataFrame
        DataFrame containing weather data with the following columns:
          - 'time': Local time in UTC+1 (e.g., '2025-04-16 13:00:00')
          - 'temp_air': Ambient air temperature in Celsius.
          - 'relative_humidity': Relative humidity in percent (0–100).
          - 'ghi': Global horizontal irradiance (W/m²).
          - 'dni': Direct normal irradiance (W/m²).
          - 'dhi': Diffuse horizontal irradiance (W/m²).
          - 'wind_speed': Wind speed in m/s.
          - 'wind_direction': Wind direction in degrees.
          - 'pressure': Atmospheric pressure in Pascal.

    Returns
    -------
    pandas.Series
        AC power output (in Watts) for the 1 m² panel, indexed by the UTC timestamps.
    """
    # Copy the DataFrame to avoid modifying the original data
    df = weather_df.copy()

    # Time specifically in UTC+1 (European winter time), for solar zenith matching the weathe time
    df.index = df['time(inst)']
    site = location.Location(latitude=lat, longitude=lon, altitude=altitude, tz='Etc/GMT-1')

    # Define all system parameters
    albedo = config.ALBEDO                              # Default value of 0.2 for taking ground reflectance into account
    module_parameters = config.MODULE_PARAMETERS        # define module like average German PV module
    losses_parameters = config.SOLAR_LOSSES             # Typical system losses (percent)
    inverter_parameters = config.INVERTER_PARAMETERS    # Default inputs for inverter
    temperature_model_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']  # Typical for pv roof installations
    
    # Create the PV system object using the defined module, inverter, temperature, and losses parameters.
    system = pvsystem.PVSystem(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        module_parameters=module_parameters,
        inverter_parameters=inverter_parameters,
        temperature_model_parameters=temperature_model_params,
        losses_parameters=losses_parameters,
        albedo=albedo)
    
    # Create a ModelChain object linking the PV system to the location.
    mc = modelchain.ModelChain(
        system, site,
        aoi_model='physical',
        spectral_model='no_loss',
        temperature_model='sapm',
        dc_model='pvwatts',
        ac_model='pvwatts',
        losses_model='pvwatts')
    
    # Run the model using the weather DataFrame.
    mc.run_model(df)
    
    # Return the AC power output for each timestamp.
    return mc.results.ac


##############################################################
#################### Publicly callable #######################
##############################################################

def sample_statistics(df_buildings):
    ### Sample buildings' roof sections: [(capacity, tilt, azim), ...]
    df_buildings["roofs"] = df_buildings["area"].apply(lambda x: _get_roof_sections(x, config.ROOF_TILT_DIST)) 
    return df_buildings

def create_supim_solar(df_buildings, df_weather, location, altitude):
    ts_dict = {}                          # dictionary to store all generated solar timeseries
    roof_list = []                        # dictionary to store all roof names

    # Loop over all roof sections on all buildings and generate timeseries for unseen (tilt, azim) combinations
    for index, row in df_buildings.iterrows():
        for roof_sec in row["roofs"]:
            tilt = roof_sec[1]
            azim = roof_sec[2]
            ts_name = f"solar_{tilt}_{azim}"

            roof_list.append((row["bus"], ts_name))
            if ts_name in ts_dict.keys(): pass
            else: ts_dict[f"solar_{tilt}_{azim}"] = _calculate_pv_power(location["lat"],location["lon"], altitude, tilt, azim, df_weather)

    # Now create urbs input dataframe by assigning timeseries to every roof section
    dfs = [ts_dict[ts_name] for (bus, ts_name) in roof_list]
    df_solar_supim = pd.concat(dfs, axis=1, keys=roof_list)
    df_solar_supim/=1000   # Convert to kWh
    df_solar_supim = df_solar_supim.reset_index(drop=True)
    df_solar_supim.index.name = "t"

    return df_solar_supim

def create_pro_solar(df_buildings):
    # Create processes for solar
    panel_list = []
    for index, row in df_buildings.iterrows():
        for roof_sec in row["roofs"]:
            bus = row["bus"]
            cap = roof_sec[0]
            tilt = roof_sec[1]
            azim = roof_sec[2]
            panel_list.append((bus, f"Rooftop PV_{tilt}_{azim}", cap))
    df_pro = pd.DataFrame(panel_list, columns=['Site', 'Process', 'cap-up'])
    
    # Assign other constants
    df_pro[["inst-cap","inv-cost-fix","inv-cost","fix-cost","var-cost","wacc","depreciation","pf-min"]] = (
    config.PV_INST_CAP, config.PV_INV_COST_FIX, config.PV_INV_COST, config.PV_FIX_COST, 
    config.PV_VAR_COST, config.PV_WACC, config.PV_DEPRECIATION, config.PV_PF_MIN)

    return df_pro.reset_index(drop=True)

def create_com_solar(panel_list):
    # panel list: [("bus1", "solar_64_178"), ("bus1", "solar_23_160"), ("bus2", ...), ...]
    df_com = pd.DataFrame(panel_list, columns=['Site', 'Commodity'])
    df_com[["Type", "price"]] = ("SupIm", np.nan)
    return df_com.reset_index(drop=True)

def create_pro_com_solar(unique_panel_list):
    df_pro_com_base = pd.DataFrame(unique_panel_list, columns=['Commodity'])
    df_pro_com_base["Process"] = df_pro_com_base['Commodity'].str.replace('^solar', 'Rooftop PV', regex=True)
    # Solar input
    df_pro_com_in = df_pro_com_base.copy()
    df_pro_com_in[["Direction", "ratio"]] = ("In", 1)
    # P output
    df_pro_com_out_p = df_pro_com_base.copy()
    df_pro_com_out_p[["Commodity", "Direction", "ratio"]] = ("electricity", "Out", 1)
    # Q output
    # df_pro_com_out_q = df_pro_com_base.copy()
    # df_pro_com_out_q[["Commodity", "Direction", "ratio"]] = ("electricity-reactive", "Out", 1)
    # Concat and reorder
    # df_pro_com = pd.concat([df_pro_com_in, df_pro_com_out_p, df_pro_com_out_q], axis=0)[["Process", "Commodity", "Direction", "ratio"]]
    df_pro_com = pd.concat([df_pro_com_in, df_pro_com_out_p], axis=0)[["Process", "Commodity", "Direction", "ratio"]]
    return df_pro_com.reset_index(drop=True)