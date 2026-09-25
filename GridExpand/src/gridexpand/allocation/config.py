"""Step 2 constants, statistics tables and scenario-derived settings.

``config`` is the one instance. Paths are absolute (see :mod:`gridexpand.paths`);
the statistics CSVs are read on first access and cached per process, so
importing this module does not touch the data directory. The cached frames are
shared: callers must not modify them in place.
"""

import functools

import numpy as np
import pandas as pd

from gridexpand.paths import ALLOCATION_GRIDS_DIR, ALLOCATION_RESULTS_DIR, STATISTICS_DIR

_STAT_DIR = str(STATISTICS_DIR)


@functools.cache
def _read_statistics(relative_path: str, header: int = 0, skiprows: int = 0) -> pd.DataFrame:
    header_arg = [0, 1] if header == 2 else [header]
    return pd.read_csv(f"{_STAT_DIR}/{relative_path}", header=header_arg, skiprows=skiprows)


@functools.cache
def _read_car_statistics(relative_path: str, dtypes: tuple[tuple[str, type], ...]) -> pd.DataFrame:
    return pd.read_csv(f"{_STAT_DIR}/{relative_path}", dtype=dict(dtypes), skiprows=1)


class Config:
    #--------------------------------------------------------------#
    #----------------- Paths/Dataset Readout ----------------------#
    #--------------------------------------------------------------#
    ##### Important data paths (absolute; see gridexpand.paths) #####
    DATA_GRID_DIR = str(ALLOCATION_GRIDS_DIR)     # Directory from which to read the pylovo grid input data (HDF5 mode)
    DATA_STAT_DIR = str(STATISTICS_DIR)           # Directory from which to read data for computing demands
    STORAGE_DIR = str(ALLOCATION_RESULTS_DIR)     # Directory in which to store resulting urbs input files

    #--------------------------------------------------------------#
    #------------------------ Weather -----------------------------#
    #--------------------------------------------------------------#
    # PVGIS weather is downloaded by gridexpand.common.weather. The reference
    # year and scientific mobility/urbs parameters come from the scenario YAML.
    TIME_ZONE = 1   # Currently only implemented for UTC+1!!!       # Shift between the location's time and GMT in hours. CET would be 1.

    #--------------------------------------------------------------#
    #---------------- Solar Generator Constants -------------------#
    #--------------------------------------------------------------#
    # Roof geometry, utilization, profile bins, and sizing live in scenario YAML.

    ### Generator
    ALBEDO = 0.2                        # Ground reflectance. 0 refers to 0% and 1 refers to 100%, default value: https://pvwatts.nrel.gov/downloads/pvwattsv5.pdf

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
    @property
    def ELEC_BY_HHSIZE_CDFS_NOHEAT(self):          # Annual demand CDFs per household size
        return _read_statistics("inhabited_buildings/elec_by_hhsize_cdfs_noheat.csv", header=2)

    @property
    def HH_SIZE_DISTRIBUTION(self):                # Household size probabilities
        return _read_statistics("inhabited_buildings/hh_size_distribution.csv", skiprows=1)

    ELEC_LPS_PATH = f"{DATA_STAT_DIR}/inhabited_buildings/elec_lps.h5"
    ### Uninhabited buildings
    ELEC_GHD_PATH = f"{DATA_STAT_DIR}/uninhabited_buildings/elec_ghd_per_m2.csv"

    @property
    def TYPE_GHD_DISTRIBUTION(self):               # Use-type probabilities of commercial/public components
        return _read_statistics("uninhabited_buildings/nonresbuilding_usetype_distribution.csv", skiprows=1)

    @property
    def AGE_GHD_DISTRIBUTION(self):                # Construction-year classes of non-residential buildings
        return _read_statistics("uninhabited_buildings/nonresbuilding_age_distribution.csv", skiprows=1)


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

    ### Heating system type, source: https://www.umweltbundesamt.de/sites/default/files/medien/11850/publikationen/11_2024_cc_waermepumpensysteme.pdf, Abbildung 26
    PROB_RADIATOR = 0.727                           # Probability for building to be heating with radiators 
    PROB_FLOOR = 0.273                              # Probability for building to be heating with floor heating

    #--------------------------------------------------------------#
    #----------- Emobpy Mobility Generator Constants --------------#
    #--------------------------------------------------------------#
    # Statistical and input data
    EMOBPY_DATA_PATH = f"{DATA_STAT_DIR}/general/"
    MOBILITY_PROFILE_POOL_DIR = f"{DATA_STAT_DIR}/general/mobility_profile_pool_old"
    MOBILITY_PROFILE_POOL_METADATA_PATH = f"{MOBILITY_PROFILE_POOL_DIR}/mobility_profile_pool_metadata.csv"
    MOBILITY_PROFILE_POOL_DEMAND_PATH = f"{MOBILITY_PROFILE_POOL_DIR}/mobility_demand_pool.csv"
    MOBILITY_PROFILE_POOL_AVAILABILITY_PATH = f"{MOBILITY_PROFILE_POOL_DIR}/mobility_availability_pool.csv"
    MOBILITY_PROFILE_POOL_WEATHER_PATH = f"{MOBILITY_PROFILE_POOL_DIR}/mobility_weather_central_germany_tmy.csv"
    MOBILITY_PROFILE_POOL_WEATHER_KEY = "central_germany_tmy"
    MOBILITY_PROFILE_POOL_WEATHER_SOURCE = "PVGIS SARAH3 TMY at lat=51.16 lon=10.45"
    MOBILITY_PROFILE_POOL_LAT = 51.16
    MOBILITY_PROFILE_POOL_LON = 10.45
    MOBILITY_PROFILE_POOL_GENERATION_VERSION = "emobpy_pool_v1"

    @property
    def CARS_PER_HH_BY_REGION(self):               # Vehicle count probabilities per RegioStaR 7 region and household size
        return _read_car_statistics(
            "general/cars_per_household_by_region.csv",
            (("region", int), ("hh_size", int), ("vehicle_count", int), ("probability", float)),
        )

    @property
    def CAR_MODEL_DISTRIBUTION(self):              # EV model shares
        return _read_car_statistics("general/cars_by_model.csv", (("model", str), ("probability", float)))
    TOTAL_HOURS = 8760



    def apply_scenario(self, scenario):
        """Set the scenario values read by the heat and emobpy generators.

        The urbs process and storage parameters are passed explicitly from
        ``scenario.technologies``; only the TEASER retrofit level, the
        reference year and the mobility settings are still module state (they
        are read inside worker processes and the vendored emobpy code).
        """
        mobility = scenario.mobility
        self.TEASER_RETROFIT_LEVEL = scenario.heat.teaser_retrofit_level
        self.REF_YEAR = mobility.reference_year
        self.PROB_COMMUTING = mobility.commuting_probability
        self.MBL_TIME_STEP_LENGTH = mobility.emobpy_timestep_hours
        self.MBL_REF_DATE = f"01/01/{mobility.reference_year}"
        self.PASSENGER_MASS = mobility.passenger_mass_kg
        self.PASSENGER_HEAT = mobility.passenger_sensible_heat_w
        self.PASSENGER_NR = mobility.passengers_per_vehicle
        self.CABIN_HEAT_TRANSFER_COEF = mobility.cabin_heat_transfer_coefficient
        self.AIR_FLOW = mobility.cabin_air_flow_m3_per_s
        self.DRIVIG_CYCLE_TYPE = mobility.driving_cycle_type
        self.ROAD_TYPE = mobility.road_type
        self.ROAD_SLOPE = mobility.road_slope
        self.CAPACITY_HOME_CHARGING = scenario.technologies.processes["home_charger"]["installed_capacity_kw"]


config = Config()
