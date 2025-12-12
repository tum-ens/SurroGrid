from config import config

import requests
import numpy as np
import pandas as pd
from datetime import timedelta, timezone
import calendar


def get_pvgis_tmy_sarah3_dataframe(latitude, longitude):
    """
    Retrieve TMY data from PVGIS using the SARAH3 satellite dataset and return as a pandas DataFrame
    
    Parameters:
    -----------
    latitude : float
        Latitude in decimal degrees (between -90 and 90)
    longitude : float
        Longitude in decimal degrees (between -180 and 180)
    startyear : int
        Start year for the TMY calculation
    endyear : int
        End year for the TMY calculation
        
    Returns:
    --------
    pandas.DataFrame
        TMY hourly data as a DataFrame with processed column names
    """
    # Base URL for PVGIS API
    base_url = config.PVGIS_URL
    
    # Parameters for the API request
    params = {
        'lat': latitude,
        'lon': longitude,
        'raddatabase': 'PVGIS-SARAH3',
        'outputformat': 'json',
        'usehorizon': 1, 
        'database': 'SARAH3',  # Specify SARAH3 dataset
    }
    
    # Make the request
    print(f"Requesting TMY data from PVGIS (SARAH3) for coordinates ({latitude}, {longitude})...")
    try:
        response = requests.get(base_url, params=params)
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
        
        # Parse the JSON response
        tmy_data = response.json()
        
        # Extract the hourly data
        if 'outputs' not in tmy_data or 'tmy_hourly' not in tmy_data['outputs']:
            print("Error: Unexpected API response structure")
            print("Full response:", tmy_data)
            return None
            
        hourly_data = tmy_data['outputs']['tmy_hourly']
        offset = tmy_data['inputs']["location"]['irradiance_time_offset']
        altitude = tmy_data['inputs']["location"]['elevation']
        
        # Convert to DataFrame
        df = pd.DataFrame(hourly_data)
        

        # Handle time column - check if it exists
        if 'time(UTC)' in df.columns:
            df['time(UTC)'] = pd.to_datetime(df['time(UTC)'], format="%Y%m%d:%H%M")
            df['time(UTC)'] = df['time(UTC)'].apply(lambda x: x.replace(year=config.REF_YEAR))

            # Define a fixed UTC+1 timezone (constant 1 hour ahead)
            df['time(UTC)'] = df['time(UTC)'].dt.tz_localize(timezone.utc)
            fixed_utc_plus_1 = timezone(timedelta(hours=config.TIME_ZONE))
            df['time(UTC+1)'] = df['time(UTC)'].dt.tz_convert(fixed_utc_plus_1)
            df['time(inst)'] = df['time(UTC+1)'] + pd.DateOffset(hours=offset) # actual measurement time in UTC+1
            # df.drop(columns=["time(UTC)"], inplace=True)

            # Potentially shift last row of dataset by one hour to the beginning as actually satellite instantaneous measurement time lying in next year already
            if offset >= 0:
                # Step 1: Extract last row
                last_row = df.iloc[-1].copy()
                df = df.iloc[:-1]
                # Step 2: Subtract one year from the date
                last_row['time(UTC+1)'] = last_row['time(UTC+1)'] - pd.DateOffset(years=1)
                last_row['time(inst)'] = last_row['time(inst)'] - pd.DateOffset(years=1)
                # Step 3: Append the modified row
                df = pd.concat([df, pd.DataFrame([last_row])], ignore_index=True)
                # Step 4: Sort by date
                df = df.sort_values('time(inst)').reset_index(drop=True)

        else:
            print("Warning: 'time' column not found in API response")
            

        # Rename columns for clarity
        column_mapping = {
            'G(h)': 'ghi',                  # W/m2
            'Gb(n)': 'dni',                 # W/m2
            'Gd(h)': 'dhi',                 # W/m2
            'T2m': 'temp_air',              # °C
            'RH': 'relative_humidity',      # %
            'SP': 'pressure',               # Pa
            'WS10m': 'wind_speed',          # m/s
            'WD10m': 'wind_direction'       # degrees
        }
        
        # Rename only the columns that exist in the DataFrame
        existing_columns = set(df.columns).intersection(set(column_mapping.keys()))
        df = df.rename(columns={col: column_mapping[col] for col in existing_columns})
        

        ### Also extract selected (year, month) combinations:
        selected_months = tmy_data["outputs"]["months_selected"]

        # Return dataframe, altitude, and selected months
        return df, altitude, selected_months
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
def get_open_meteo_soil_temperature(lat, lon, selected_months):
    data_dict = {}
    for id in range(len(selected_months)):
        month = selected_months[id]["month"]
        year = selected_months[id]["year"]

        ### Construct start/end dates for each month
        start = f"{year}-{month:02d}-01"
        if month == 2: last_day = 28  # Always end on February 28
        else: last_day = calendar.monthrange(year, month)[1]
        end = f"{year}-{month:02d}-{last_day:02d}"

        ### API call
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": "soil_temperature_100_to_255cm",
            "timezone": config.OPENMETEO_TIME_ZONE
        }
        response = requests.get(config.OPENMETEO_URL, params=params)
        
        data = response.json()
        data_dict[month] = pd.DataFrame(data["hourly"])

    # Post-processing
    monthly_data = pd.concat([dat for key, dat in data_dict.items()], axis = 0)
    monthly_data.drop(columns=["time"], inplace=True)
    monthly_data.reset_index(drop=True, inplace=True)
    return monthly_data
    
def get_dew_point(temp_celsius, relative_humidity):
    """
    Calculate the dew point temperature in degrees Celsius according to Marcus-Tetens.

    Parameters:
    temp_celsius (float): Air temperature in degrees Celsius.
    relative_humidity (float): Relative humidity in percentage (0-100).

    Returns:
    float: Dew point temperature in degrees Celsius.
    """
    a = 17.27
    b = 237.7  # degrees Celsius

    gamma = (a * temp_celsius) / (b + temp_celsius) + np.log(relative_humidity / 100.0)
    dew_point = (b * gamma) / (a - gamma)
    return dew_point