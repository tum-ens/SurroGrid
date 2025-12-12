import numpy as np
import pandas as pd

class Config:
    #--------------------------------------------------------------#
    #----------------- Paths/Dataset Readout ----------------------#
    #--------------------------------------------------------------#
    ##### Important data paths #####
    DATA_GRID_DIR = "data/grids"           # Directory from which to read the pylovo grid input data
    DATA_STAT_DIR = "data/statistics"      # Directory from which to read data for computing demands
    STORAGE_DIR = "results"                # Directory in which to store resulting urbs input files

    #--------------------------------------------------------------#
    #-------------- Weather Data API Connections ------------------#
    #--------------------------------------------------------------#
    # PVGIS API
    PVGIS_URL = "https://re.jrc.ec.europa.eu/api/tmy"               # URL from which to fetch typical meterological year weather data
    REF_YEAR = 2009                                                 # Reference year on which whole simulation is based (in terms of holidays!, 2009 is choosen as the real electricity data matches its holidays)
    TIME_ZONE = 1   # Currently only implemented for UTC+1!!!       # Shift between the location's time and GMT in hours. CET would be 1.

    # OpenMeteo API
    OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive" # URL from which to fetch soil temperature data
    OPENMETEO_TIME_ZONE = "UTC+01:00"  # Currently only  UTC+1!!!   # Timezone at location (for alignment of weather with human actions)

    #--------------------------------------------------------------#
    #---------------- Solar Generator Constants -------------------#
    #--------------------------------------------------------------#
    ### Statistics
    # Roof type distribution
    PROB_FLAT_ROOF = 0.15               # Source: https://www.researchgate.net/publication/317686603_A_Decision_Support_System_for_Photovoltaic_Potential_Estimation
    PROB_GABLED_ROOF = 0.64 + 0.21      # Assuming other roof types similar to gabled
    
    # How much of roof useable for PV
    FLAT_ROOF_UTILIZATION = 0.27        # Source: https://www.sciencedirect.com/science/article/pii/S0038092X14002114, Table 2
    SLANTED_ROOF_UTILIZATION = 0.58

    # Roof tilt distribution
    ROOF_TILT_DIST = pd.read_csv(f"{DATA_STAT_DIR}/general/tilt_distribution.csv", skiprows=1)

    ### Generator
    ALBEDO = 0.2                        # Ground reflectance. 0 refers to 0% and 1 refers to 100%, default value: https://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
    PV_AREA_FACTOR = 0.202              # kW/m^2 roof area, typical capacity for mono-SI under standard test conditions STC, https://publica.fraunhofer.de/entities/publication/f6b3dc37-454c-4d29-9040-76ce1b8454da

    MODULE_PARAMETERS = {
        'pdc0': 1000,                   # set to 1kW reference cap, will later be multiplied by (sys_cap_kWh/m^2)*roof_area
        'gamma_pdc': -0.0035}           # Temperature coefficient for mono-SI, https://publica.fraunhofer.de/entities/publication/f6b3dc37-454c-4d29-9040-76ce1b8454da
    SOLAR_LOSSES = {                    # Default values: https://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
        'soiling': 2,
        'shading': 3,
        'snow': 0,
        'mismatch': 2,
        'wiring': 2,
        'connections': 0.5,
        'lid': 1.5,
        'nameplate_rating': 1,
        'age': 1,
        'availability': 0.3}
    INVERTER_PARAMETERS = {
        'pdc0': 1000/1.1,               # Default dc to ac ratio of 1.1: https://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
        'eta_inv_nom': 0.96,            # Nominal inverter efficiency, default value: https://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
        'eta_inv_ref': 0.9637}          # Reference inverter efficiency, default value: https://pvwatts.nrel.gov/downloads/pvwattsv5.pdf


    #--------------------------------------------------------------#
    #---------------- Electrical Demand Assignment ----------------#
    #--------------------------------------------------------------#
    ### Private households
    ELEC_BY_HHSIZE_CDFS_NOHEAT = pd.read_csv(f'{DATA_STAT_DIR}/inhabited_buildings/elec_by_hhsize_cdfs_noheat.csv', header=[0,1])   
    HH_SIZE_DISTRIBUTION = pd.read_csv(f'{DATA_STAT_DIR}/inhabited_buildings/hh_size_distribution.csv', header=[0], skiprows=1) 
    ELEC_LPS_PATH = f"{DATA_STAT_DIR}/inhabited_buildings/elec_lps.h5"
    ### Uninhabited buildings
    ELEC_GHD_PATH = f"{DATA_STAT_DIR}/uninhabited_buildings/elec_ghd_per_m2.csv"
    DHW_GHD_PATH = f"{DATA_STAT_DIR}/uninhabited_buildings/dhw_ghd_per_m2.csv"
    TYPE_GHD_DISTRIBUTION = pd.read_csv(f'{DATA_STAT_DIR}/uninhabited_buildings/nonresbuilding_usetype_distribution.csv', header=[0], skiprows=1) 
    AGE_GHD_DISTRIBUTION = pd.read_csv(f'{DATA_STAT_DIR}/uninhabited_buildings/nonresbuilding_age_distribution.csv', header=[0], skiprows=1) 


    #--------------------------------------------------------------#
    #----------------- Heat Generator Constants -------------------#
    #--------------------------------------------------------------#
    DISTGEN_DATA_PATH = f"{DATA_STAT_DIR}/general/"

    # Time
    HOLIDAYS = [1,6,100,121,141,152,162,276,305,358,359,360,365]    # Holidays of 2009 in Bavaria, this is chosen to match the original electricity data source
    INITIAL_DAY = [4]                               # Initial day of the 1: Monday, ..., 7: Sunday
    DATA_LENGTH = 31536000                          # Temporal length of input data in seconds

    # Building design
    T_SET_MIN = 20.0                                # °C, Required minimum indoor temperature (for heating load calculation)
    T_SET_MIN_NIGHT = 18.0                          # °C, Required minimum indoor temperature at night (for heating load calculation)
    T_SET_MAX = 23.0                                # °C, Required maximum indoor temperature (for cooling load calculation)
    T_SET_MAX_NIGHT = 28.0                          # °C, Required maximum indoor temperature at night (for cooling load calculation)
    VENTILATION_RATE = 0.5                          # 1/h, Room ventilation rate
    BUILDINGS_SHORT = ["SFH", "MFH", "TH", "AB"]    # Abbreviations of the selectable building types
    BUILDINGS_LONG = ["single_family_house", "multi_family_house", "terraced_house", "apartment_block"]     # Names of the four selectable building types.
    RETROFIT_SHORT = [0, 1, 2]                      # Abbreviations of the retrofit levels.
    RETROFIT_LONG = ["tabula_standard", "tabula_retrofit", "tabula_adv_retrofit"]   # Names of the retrofit levels.
    DHWLOAD = [4662.1, 4662.1, 4662.1, 3999.8]      # Watt, Maximal power for domestic hot water for each of the four building types (SFH, MFH, TH and AB)
    MEAN_DRAWOFF_VOL = [40, 40, 40, 40]             # Liters, Mean drawoff DHW volume per day for each of the four building types (SFH, MFH, TH and AB). Source: 12831-3/A100 Table NA.4

    # Physics
    RHO_AIR = 1.2                                   # kg/m3, density air
    C_P_AIR = 1000.0                                # J/kgK, specific heat capacity
    RHO_WATER = 1000.0                              # kg/m3, density water
    C_P_WATER = 4.18                                # kJ/kgK, specific heat capacity

    ### Heat pump data, source: https://www.nature.com/articles/s41597-019-0199-y
    @staticmethod
    def ASHP_COP(delta_T):                          # COP of air source heat pump
        return pd.DataFrame(6.08 - 0.09*delta_T + 0.0005*np.square(delta_T))
    @staticmethod
    def GSHP_COP(delta_T):                          # COP of ground source heat pump
        return pd.DataFrame(10.29 - 0.21*delta_T + 0.0012*np.square(delta_T))
    # @staticmethod
    # def WSHP_COP(delta_T):                          # COP of water source heat pump
    #     return pd.DataFrame(9.97 - 0.20*delta_T + 0.0012*np.square(delta_T))
    
    # HP_TYPE_DIST = pd.read_csv(f'{DATA_STAT_DIR}/general/heat_pump_type.csv', header=[0], skiprows=1)

    ### Heating system type, source: https://www.umweltbundesamt.de/sites/default/files/medien/11850/publikationen/11_2024_cc_waermepumpensysteme.pdf, Abbildung 26
    PROB_RADIATOR = 0.727                           # Probability for building to be heating with radiators 
    PROB_FLOOR = 0.273                              # Probability for building to be heating with floor heating

    #--------------------------------------------------------------#
    #----------- Emobpy Mobility Generator Constants --------------#
    #--------------------------------------------------------------#
    # Statistical and input data
    EMOBPY_DATA_PATH = f"{DATA_STAT_DIR}/general/"
    CARS_PER_HH_BY_REGION = pd.read_csv(f"{DATA_STAT_DIR}/general/cars_per_household_by_region.csv", 
                                    dtype={"region": int, "hh_size": int, "vehicle_count": int, "probability": float},
                                    skiprows=1)
    CAR_MODEL_DISTRIBUTION = pd.read_csv(f"{DATA_STAT_DIR}/general/cars_by_model.csv", 
                                    dtype={"model": str, "probability": float},
                                    skiprows=1)
    PROB_COMMUTING = 0.62              # Probability of a car having commuter profile, source: https://www.nature.com/articles/s41597-021-00932-9

    # Emobpy data
    MBL_TIME_STEP_LENGTH = 0.5
    MBL_REF_DATE = f"01/01/{REF_YEAR}"
    TOTAL_HOURS = 8760

    PASSENGER_MASS = 75                # kg
    PASSENGER_HEAT = 70                # W
    PASSENGER_NR = 1.5                 # Passengers per vehicle including driver
    CABIN_HEAT_TRANSFER_COEF = 10      # W/(m2K). Interior walls
    AIR_FLOW = 0.01                    # m3/s. Ventilation
    DRIVIG_CYCLE_TYPE ='WLTC'          # Two options "WLTC" or "EPA"
    ROAD_TYPE = 0                      # For rolling resistance, Zero represents a new road.
    ROAD_SLOPE = 0

    CAPACITY_HOME_CHARGING = 11        # kW
    
    #--------------------------------------------------------------#
    #------------------ Urbs Input Constants ----------------------#
    #--------------------------------------------------------------#
    ###### Buy-Sell Price constants ######
    BSP_IMPORT = 0.398        # €/kWh buy price
    BSP_FEED_IN = 0.0         # €/kWh sell price

    ########## Demand Constants ##########
    # ELEC_REACT_PF = 0.9

    ######### Process constants ##########
    # PV solar constants
    PV_INST_CAP = 0           # kW power capacity installed already
    PV_INV_COST_FIX = 6565    # €/installation decision
    PV_INV_COST = 533.7       # €/kW newly installed capacity
    PV_FIX_COST = 0           # €/h (if operating)
    PV_VAR_COST = 0           # €/kWh process flux (operation) 
    PV_WACC = 0.022           # decimal not %
    PV_DEPRECIATION = 15      # years
    PV_PF_MIN = np.nan
    # PV_PF_MIN = 0.95          # tan(phi_min), powerfactor for reactive power
    
    # HP + booster constants
    HP_AIR_INST_CAP = 0           # kW power capacity installed already
    HP_AIR_CAP_UP = 2000          # kW arbitrary upper capacity
    HP_AIR_INV_COST_FIX = 6600    # €/installation decision
    HP_AIR_INV_COST = 750         # €/kW newly installed capacity
    HP_AIR_FIX_COST = 0           # €/h (if operating)
    HP_AIR_VAR_COST = 0           # €/kWh process flux (operation) 
    HP_AIR_WACC = 0.0216          # decimal not %
    HP_AIR_DEPRECIATION = 20      # years
    HP_AIR_PF_MIN = np.nan        # no power factor as handled differently
    # HP_AIR_Q_IN_RATIO = 0.292     # 1/HP_Q_IN_RATIO is amount of Q taken in relative to P

    # HP_GRD_INST_CAP = 0           # kW power capacity installed already
    # HP_GRD_CAP_UP = 2000          # kW arbitrary upper capacity
    # HP_GRD_INV_COST_FIX = 26200   # €/installation decision
    # HP_GRD_INV_COST = 466.7       # €/kW newly installed capacity
    # HP_GRD_FIX_COST = 0           #€/h (if operating)
    # HP_GRD_VAR_COST = 0           # €/kWh process flux (operation) 
    # HP_GRD_WACC = 0.0017          # decimal not %
    # HP_GRD_DEPRECIATION = 27      # years
    # HP_GRD_PF_MIN = np.nan        # no power factor as handled differently
    # HP_GRD_Q_IN_RATIO = 0.292     # 1/HP_Q_IN_RATIO is amount of Q taken in relative to P

    HP_BST_INST_CAP = 0        # kW power capacity installed already
    HP_BST_CAP_UP = 2000       # kW arbitrary upper capacity
    HP_BST_INV_COST_FIX = 100  # €/installation decision
    HP_BST_INV_COST = 83.3     # €/kW newly installed capacity
    HP_BST_FIX_COST = 0        # €/h (if operating)
    HP_BST_VAR_COST = 0        # €/kWh process flux (operation) 
    HP_BST_WACC = 0.0216       # decimal not %
    HP_BST_DEPRECIATION = 20   # years
    HP_BST_PF_MIN = np.nan     # no power factor as handled differently

    # Heating dummy (space/water)
    HDM_INST_CAP = 2000       # kW power capacity installed already
    HDM_CAP_UP = 2000         # kW arbitrary upper capacity
    HDM_INV_COST_FIX = np.nan # €/installation decision
    HDM_INV_COST = 0          # €/kW newly installed capacity
    HDM_FIX_COST = 0          # €/h (if operating)
    HDM_VAR_COST = 0          # €/kWh process flux (operation) 
    HDM_WACC = 0.07           # decimal not %
    HDM_DEPRECIATION = 1      # years
    HDM_PF_MIN = np.nan       # no power factor as handled differently

    # Charging station constants
    CS_INST_CAP = 11          # kW power capacity installed already
    CS_CAP_UP = 11            # kW arbitrary upper capacity
    CS_INV_COST_FIX = np.nan  # €/installation decision
    CS_INV_COST = 0           # €/kW newly installed capacity
    CS_FIX_COST = 0           # €/h (if operating)
    CS_VAR_COST = 0           # €/kWh process flux (operation) 
    CS_WACC = 0.07            # decimal not %
    CS_DEPRECIATION = 1       # years
    CS_PF_MIN = np.nan        # no power factor as handled differently

    # Import/Feed-In/Q_feeder_central
    IMP_INST_CAP = 2000        # kW power capacity installed already
    IMP_CAP_UP = 2000          # kW arbitrary upper capacity
    IMP_INV_COST_FIX = np.nan  # €/installation decision
    IMP_INV_COST = 0           # €/kW newly installed capacity
    IMP_FIX_COST = 0           # €/h (if operating)
    IMP_VAR_COST = 0           # €/kWh process flux (operation) 
    IMP_WACC = 0.07            # decimal not %
    IMP_DEPRECIATION = 30      # years
    IMP_PF_MIN = np.nan        # no power factor as handled differently

    ######### Storage constants ##########
    # Thermal storage:
    TS_INST_CAP_C = 0         # kWh storage capacity installed already
    TS_CAP_UP_C = 10000       # kWh storage arbitrary upper capacity
    TS_INST_CAP_P = 0         # kW power capacity installed already
    TS_CAP_UP_P = 1500        # kW power arbitrary upper capacity
    TS_EFF_IN = 0.932         # charging efficiency (decimal not percent)
    TS_EFF_OUT = 1            # discharge efficiency (decimal not percent)
    TS_DISCHARGE = 0          # self-discharge per timestep (decimal not percent)
    TS_EP_RATIO = 0.15        # energy to power storage ratio 
    TS_INV_COST_P = 0         # €/kW variable investment cost
    TS_INV_COST_C = 58        # €/kWh variable investment cost
    TS_FIX_COST_P = 0         # €/installation decision
    TS_FIX_COST_C = 0         # €/installation decision
    TS_VAR_COST_P = 0.001     # €/kWh small cost for charging/discharging to prevent useless storage cycling
    TS_WACC = 0.0216          # decimal not %
    TS_DEPRECIATION = 20      # years

    # Battery storage:
    BS_INST_CAP_C = 0         # kWh storage capacity installed already
    BS_CAP_UP_C = 6000        # kWh storage arbitrary upper capacity
    BS_INST_CAP_P = 0         # kW power capacity installed already
    BS_CAP_UP_P = 2000        # kW power arbitrary upper capacity
    BS_EFF_IN = 0.961         # charging efficiency (decimal not percent)
    BS_EFF_OUT = 1            # discharge efficiency (decimal not percent)
    BS_DISCHARGE = 0          # self-discharge per timestep (decimal not percent)
    BS_EP_RATIO = 1.58        # energy to power storage ratio 
    BS_INV_COST_P = 0         # €/kW variable investment cost
    BS_INV_COST_C = 976       # €/kWh variable investment cost
    BS_FIX_COST_P = 0         # €/installation decision
    BS_FIX_COST_C = 0         # €/installation decision
    BS_VAR_COST_P = 0.001     # €/kWh small cost for charging/discharging to prevent useless storage cycling
    BS_WACC = 0.022           # decimal not %
    BS_DEPRECIATION = 15      # years

    # Mobility storage:
    MS_EFF_IN = 1             # charging efficiency (decimal not percent)
    MS_EFF_OUT = 1            # discharge efficiency (decimal not percent)
    MS_DISCHARGE = 0          # self-discharge per timestep (decimal not percent)
    MS_EP_RATIO = np.nan      # energy to power storage ratio 
    MS_INV_COST_P = 0         # €/kW variable investment cost
    MS_INV_COST_C = 0         # €/kWh variable investment cost
    MS_FIX_COST_P = 0         # €/installation decision
    MS_FIX_COST_C = 0         # €/installation decision
    MS_VAR_COST_P = 0.001     # €/kWh small cost for charging/discharging to prevent useless storage cycling
    MS_WACC = 0.07            # decimal not %
    MS_DEPRECIATION = 20      # years


config = Config()