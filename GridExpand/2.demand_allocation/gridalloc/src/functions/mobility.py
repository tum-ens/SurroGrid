from config import config
from src.external.emobpy import Charging
from src.external.emobpy import Mobility
from src.external.emobpy import Availability
from src.external.emobpy import DataBase
from src.external.emobpy import Consumption, HeatInsulation, BEVspecs
from src.external.emobpy.tools import set_seed

import tempfile
import warnings
import numpy as np
import pandas as pd

##############################################################
#################### Sampling Statistics #####################
##############################################################
def _sample_cars_in_building(occ_list, df_cars_per_household):
    car_list = []
    for n_occ in occ_list:
        if n_occ>5: n_occ = 5
        subset = df_cars_per_household[df_cars_per_household['hh_size'] == n_occ]
        car_list.append(np.random.choice(subset['vehicle_count'].values, size=1, p=subset['probability'].values).item())
    return car_list, sum(car_list)
    
def _sample_model_and_commuting(n_cars, prob_commuter, df_car_model_dist, df_region, bus):
    if n_cars == 0: return {}
    else:
        ### First create all column of the car dataframe
        buses = pd.DataFrame([bus]*n_cars, columns=["bus"])     # bus at which car is placed, always the same
        ids = pd.DataFrame(range(n_cars), columns=["id"])       # id of car at the bus (starting from 0)
        car_model = pd.DataFrame(np.random.choice(df_car_model_dist["model"].values, size=n_cars, p=df_car_model_dist["probability"].values), columns=["model"])    # sampling a car model based on current sales statistics
        commuter = pd.DataFrame(np.random.choice(["commuter", "non-commuter"], size=n_cars, p=[prob_commuter, 1-prob_commuter]), columns=["schedule"])              # samling whether car is used mainly for commuting
        seeds = pd.DataFrame([int("".join(str(i) for i in [df_region["bcid"],df_region["plz"],df_region["kcid"],bus,idx])) for idx in range(n_cars)], columns=["seed"])       # creating a unique seed for running emobpy on the car later

        # From all columns create a dictionary now
        df_cars = pd.concat([buses, ids, car_model, commuter, seeds], axis=1)
        car_dict = (df_cars.set_index(["bus","id"]))[["model","schedule","seed"]].to_dict(orient="index")
        return car_dict

##############################################################
####################### Running emobpy #######################
##############################################################
def _simulate_vehicles(vehicle_configs, weather):
    """
    Simulate driving, consumption, availability, and charging demand for multiple EVs.

    Parameters:
    - vehicle_configs: list of dicts, each with keys:
        * brand: EV manufacturer (str)
        * model: EV model name (str)
        * spec_name: matching spec in evspecs.json (str)
        * year: model year (int)
        * schedule: "commuter" or "non-commuter"
        * seed: integer seed for reproducibility
    - config_folder: folder with emobpy config files

    Returns:
    - List of pandas.DataFrame: charging time series for each vehicle, in same order
    """
    ### Settings
    config_folder = config.EMOBPY_DATA_PATH
    stat_ntrip_path="TripsPerDay.csv"
    stat_km_duration_path="DistanceDurationTrip.csv"
    
    total_hours = config.TOTAL_HOURS


    ### Code
    results = {}
    batteries = {}
    # Load BEV specs once
    BEVS = BEVspecs(filename=config_folder+"/evspecs.json")   # Database that contains BEV models

    for idx, cfg in vehicle_configs.items():
        print(f"Currently generating vehicle {idx}...")
        # Determine departure-destination trips file based on schedule
        if cfg['schedule'] == "commuter": stat_dest_path = "DepartureDestinationTrip_commuter.csv"
        elif cfg['schedule'] == "non-commuter": stat_dest_path = "DepartureDestinationTrip_noncommuter.csv"
        else: raise ValueError(f"Unknown schedule '{cfg['schedule']}' for vehicle {idx}")

        # Temporary directory for this vehicle's DB
        with tempfile.TemporaryDirectory() as tmpdb:
            # Apply a new seed for each vehicle
            set_seed(seed=cfg['seed'], dir=tmpdb)

            # ------------- Mobility Profile -----------------
            m = Mobility(config_folder=config_folder)
            m.set_params(
                name_prefix=f"EV{idx}",
                total_hours=total_hours,
                time_step_in_hrs=config.MBL_TIME_STEP_LENGTH,
                category="user_defined",
                reference_date=config.MBL_REF_DATE
            )
            m.set_stats(
                stat_ntrip_path=stat_ntrip_path,
                stat_dest_path=stat_dest_path,
                stat_km_duration_path=stat_km_duration_path
            )
            m.set_rules(
                rule_key="user_defined"
            )
            m.run()
            m.save_profile(folder=tmpdb, description=f"Mobility EV{idx}")
            # -------------- Consumption -----------------------
            DB = DataBase(tmpdb)
            DB.loadfiles()
            DB.update()  

            mname = m.name
            HI = HeatInsulation(True)                                 # Creating the heat insulation by copying the default configuration
            ev_model = BEVS.model(("A",cfg["model"],"0"))             # Model instance that contains vehicle parameters
            batteries[idx] = ev_model.parameters["battery_cap"]
            ev_model.parameters["battery_cap"]=100                    # Add slack to prevent time consuming infeasibilities (this only affects the sampled charging behavior, but as we only intend to charge at home and assign every other location to home later anyways, this does not matter too much - importantly the real battery cap has to be used later in urbs!!!)
            c = Consumption(mname, ev_model)
            c.load_setting_mobility(DB)

            c.run(
                heat_insulation = HI,
                weather = weather,
                passenger_mass = config.PASSENGER_MASS,
                passenger_sensible_heat = config.PASSENGER_HEAT,
                passenger_nr = config.PASSENGER_NR,
                air_cabin_heat_transfer_coef = config.CABIN_HEAT_TRANSFER_COEF,
                air_flow = config.AIR_FLOW,
                driving_cycle_type = config.DRIVIG_CYCLE_TYPE,
                road_type = config.ROAD_TYPE,
                road_slope = config.ROAD_SLOPE
                )
            c.save_profile(tmpdb)
            #----------------- Availability -----------------------
            DB.update()
            cname = c.name
            charging_scenario = {
                "prob_charging_point": {
                    "home":      {"home": 1.0},
                    "driving":   {"none": 0.99, "slack":0.01},
                    "errands":   {"none": 0.99, "slack":0.01},
                    "escort":    {"none": 0.99, "slack":0.01},
                    "leisure":   {"none": 0.99, "slack":0.01},
                    "shopping":  {"none": 0.99, "slack":0.01},
                    "workplace": {"none": 0.99, "slack":0.01}},
                "capacity_charging_point": {
                    "home": config.CAPACITY_HOME_CHARGING, 
                    "none": 0, 
                    "slack": 50}
            }

            GA = Availability(cname, DB)
            GA.set_scenario(charging_scenario)
            GA.run()
            GA.save_profile(tmpdb)
            #------------------- Charging Demand -----------------------
            DB.update()
            aname = GA.name
            strategy = "immediate"

            Ch = Charging(aname)
            Ch.load_scenario(DB)
            Ch.set_sub_scenario(strategy)
            Ch.run()

            # Append the result timeseries
            results[idx] = Ch.timeseries

    return results, batteries

def _most_frequent_random_tie(series):
    """A single element: one of the most frequent values from the original series, chosen uniformly at random in case of ties."""
    counts = series.value_counts()
    max_count = counts.max()
    top_values = counts[counts == max_count].index
    return np.random.choice(top_values)

def _reallocate_to_home(df_hourly):
    """Any grid charge that occurred away from “home” is re-assigned to the temporally closest home‐charging hour, and zeroed out at the original hour."""
    # Ensure the index is sorted by time
    df_hourly = df_hourly.sort_index()

    # Find indices where charging_point is 'home'
    home_indices = df_hourly[df_hourly['charging_point'] == 'home'].index

    # Iterate over non-home, non-zero charge rows
    for ts, row in df_hourly[(df_hourly['charging_point'] != 'home') & (df_hourly['charge_grid'] > 0)].iterrows():
        # Calculate time differences
        time_diffs = abs(home_indices - ts)
        if not time_diffs.empty:
            nearest_home_ts = home_indices[time_diffs.argmin()]
            # Add value to nearest home timestamp
            df_hourly.at[nearest_home_ts, 'charge_grid'] += row['charge_grid']
            # Zero out the original value
            df_hourly.at[ts, 'charge_grid'] = 0.0

    return df_hourly

def _consolidate_home_stretches(df_hourly):
    """Each continuous block of “home” hours has its grid-charging amounts aggregated into a single entry at the end of the block, with all intermediate entries set to zero."""
    df = df_hourly.copy()
    
    # Create a mask for 'home'
    is_home = df['charging_point'] == 'home'
    
    # Identify contiguous stretches using a group id
    # By comparing each row with the previous, we can group identical consecutive values
    home_stretch_id = (is_home != is_home.shift()).cumsum()
    
    # Filter only the 'home' stretches
    home_stretches = df[is_home].copy()
    home_stretches['stretch_id'] = home_stretch_id[is_home]
    
    # For each stretch, get the indices and sum the values
    for stretch_id, stretch_df in home_stretches.groupby('stretch_id'):
        total = stretch_df['charge_grid'].sum()
        if not stretch_df.empty:
            all_indices = stretch_df.index
            # Determine max total allowed charge: aggregated_power*timestep_len/n_steps = avg_energy_charged per timestep <= home_charging_power*timestep_len
            # >>> aggregated_power <= n_steps*home_charging_power 
            max_charge_allowed = len(all_indices)*config.CAPACITY_HOME_CHARGING-0.1   # No modification with timestep length here as still in power and not energy frame
            total = min(total, max_charge_allowed)
            # Zero all
            df.loc[all_indices, 'charge_grid'] = 0.0
            # Set total only at the last timestamp
            df.at[all_indices[-1], 'charge_grid'] = total
            
    return df

##############################################################
###################### Publicly callable #####################
##############################################################
def sample_statistics(df_buildings, df_region):
    # Select correct car owner statistics
    region = df_region["regio7"]
    df_cars_per_hh = config.CARS_PER_HH_BY_REGION[config.CARS_PER_HH_BY_REGION["region"]==region]

    # Now sample number of owned cars, and their model + driver type
    df_buildings[["cars_by_flat", "n_cars_tot"]] = df_buildings["occ_list"].apply(lambda x: pd.Series(_sample_cars_in_building(x, df_cars_per_hh)))
    df_buildings["car_dict"] = df_buildings.apply(lambda x: _sample_model_and_commuting(x["n_cars_tot"], config.PROB_COMMUTING, config.CAR_MODEL_DISTRIBUTION, df_region, x["bus"]), axis=1)
    
    return df_buildings

def prepare_weather_input(df_weather):
    ### Prepare weather data in correct input format
    weather = {
        "temp_air": np.array(df_weather["temp_air"].copy()),
        "pressure": np.array(df_weather["pressure"].copy())/100,    # convert to mbar
        "dew_point": np.array(df_weather["dew_point"].copy()),
        "relative_humidity": np.array(df_weather["relative_humidity"].copy())
    }
    return weather

def get_mobility_demand(vehicles, weather):
    ### Run emobpy for all grid vehicles
    # print(f"Running mobility generator for {len(vehicles)} vehicles...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # all_timeseries, all_batteries = _simulate_vehicles(vehicles, weather)
        max_retries = 3
        for attempt in range(1, max_retries):
            try:
                all_timeseries, all_batteries = _simulate_vehicles(vehicles, weather)
                break
            except Exception as e:
                if attempt == max_retries:
                    # give up: something in the code, not randomness, is broken
                    raise RuntimeError(
                        f"Vehicle(s) {vehicles.keys()} failed after {max_retries} attempts: {e}"
                    )
                # otherwise bump the seed and retry
                for key, car in vehicles.items():
                    car['seed'] += 1
                print(f"Retry #{attempt} for vehicle(s) {vehicles.keys()}.")

    ### Compile emobpy results to urbs friendly dataframe
    avai_dict = {}
    demand_dict = {}
    for key, value in all_timeseries.items():
        df = value.resample('h').agg({                      # Resample to hourly, take most occuring driving state if tie take random
            'charge_grid': 'sum',
            'charging_point': _most_frequent_random_tie
        })
        df = _reallocate_to_home(df)                        # Shift all demands occuring outside of home to nearest home
        df = _consolidate_home_stretches(df)                # Shift all demands with "home"-stretch to end of stretch
        
        df['availability'] = np.where(df['charging_point'] == 'home', 1, 0)

        avai_dict[key] = df["availability"]
        
        # Clip charging demand below battery capacity as otherwise infeasibility in urbs
        charge = df["charge_grid"]*config.MBL_TIME_STEP_LENGTH
        max_cap_allowed = all_batteries[key]-0.1      # Have some slack to prevent rounding caused MILP infeasibilities
        if max_cap_allowed<0: max_cap_allowed=0
        charge = charge.clip(lower=0, upper=max_cap_allowed)
        demand_dict[key] = charge

    # availability = pd.DataFrame(avai_dict).reset_index(drop=True)
    # new_ids = [f"charging_station{id}" for id in availability.columns.levels[1]]
    # availability.columns = availability.columns.set_levels(new_ids, level=1)

    # mob_demand = pd.DataFrame(demand_dict).reset_index(drop=True)
    # new_ids = [f"mobility{id}" for id in mob_demand.columns.levels[1]]
    # mob_demand.columns = mob_demand.columns.set_levels(new_ids, level=1)

    # Handle availability DataFrame
    availability = pd.DataFrame(avai_dict).reset_index(drop=True)
    if isinstance(availability.columns, pd.MultiIndex) and len(availability.columns.levels) > 1:
        new_ids = [f"charging_station{id}" for id in availability.columns.levels[1]]
        availability.columns = availability.columns.set_levels(new_ids, level=1)
    else:
        # If not MultiIndex, create an empty or default structure
        new_ids = []
        availability.columns = pd.MultiIndex.from_arrays([availability.columns, new_ids])

    # Handle mob_demand DataFrame
    mob_demand = pd.DataFrame(demand_dict).reset_index(drop=True)
    if isinstance(mob_demand.columns, pd.MultiIndex) and len(mob_demand.columns.levels) > 1:
        new_ids = [f"mobility{id}" for id in mob_demand.columns.levels[1]]
        mob_demand.columns = mob_demand.columns.set_levels(new_ids, level=1)
    else:
        # If not MultiIndex, create an empty or default structure
        new_ids = []
        mob_demand.columns = pd.MultiIndex.from_arrays([mob_demand.columns, new_ids])

    # Replace hour at end of year with predecessing 
    # (important as we shifted all demands to last timestep that car is home to allow flexible charging,
    #  but for last day of year all charging is shifted into last possible hour creating an unreasonable demand peak)
    # Tried to extend emobpy simulation by another 24 hours and cut of that result, but emobpy is only working internally up 8760 hours 
    mob_demand.iloc[-1] = mob_demand.iloc[-2]
    availability.iloc[-1] = availability.iloc[-2]
    return mob_demand, availability, all_batteries

def create_pro_mob(battery_dict):
    if not battery_dict: return pd.DataFrame()
    else:
        df_pro = pd.DataFrame([(bus, f"charging_station{id}") for (bus, id) in battery_dict.keys()], columns=["Site","Process"])
        df_pro[["inst-cap","cap-up","inv-cost-fix","inv-cost","fix-cost","var-cost","wacc","depreciation","pf-min"]] = (
                config.CS_INST_CAP, config.CS_CAP_UP, config.CS_INV_COST_FIX, config.CS_INV_COST, 
                config.CS_FIX_COST, config.CS_VAR_COST, config.CS_WACC, config.CS_DEPRECIATION, config.CS_PF_MIN)
        return df_pro.reset_index(drop=True)
    
def create_com_mob(battery_dict):
    if not battery_dict: return pd.DataFrame()
    else:
        df_com = pd.DataFrame([(bus, f"mobility{id}") for (bus, id) in battery_dict.keys()], columns=["Site","Commodity"])
        df_com[["Type","price"]] = ("Demand", np.nan)
        return df_com.reset_index(drop=True)

def create_pro_com_mob(battery_dict):
    if not battery_dict: return pd.DataFrame()
    else:
        max_id = max([id for _,id in battery_dict.keys()])
        df_pro_com_in = pd.DataFrame([(f"charging_station{id}", "electricity", "In", 1) for id in range(max_id+1)], columns=["Process","Commodity","Direction","ratio"])
        df_pro_com_out = pd.DataFrame([(f"charging_station{id}", f"mobility{id}", "Out", 1) for id in range(max_id+1)], columns=["Process","Commodity","Direction","ratio"])
        return pd.concat([df_pro_com_in, df_pro_com_out], axis=0).reset_index(drop=True)

def create_sto_mob(battery_dict):
    if not battery_dict: return pd.DataFrame()
    else:
        df_sto = pd.DataFrame([(bus, f"mobility_storage{id}", f"mobility{id}",cap,cap,cap,cap) for (bus, id), cap in battery_dict.items()], columns=["Site","Storage","Commodity","inst-cap-c","cap-up-c","inst-cap-p","cap-up-p"])
        df_sto[["eff-in","eff-out","discharge","ep-ratio","inv-cost-p","inv-cost-c","fix-cost-p","fix-cost-c","var-cost-p","wacc","depreciation"]] = (
        config.MS_EFF_IN, config.MS_EFF_OUT, config.MS_DISCHARGE, config.MS_EP_RATIO,
            config.MS_INV_COST_P, config.MS_INV_COST_C, config.MS_FIX_COST_P, config.MS_FIX_COST_C,
            config.MS_VAR_COST_P, config.MS_WACC, config.MS_DEPRECIATION)
        return df_sto.reset_index(drop=True)