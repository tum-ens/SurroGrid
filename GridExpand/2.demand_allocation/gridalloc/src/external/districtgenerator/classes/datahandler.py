# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from teaser.project import Project
from .envelope import Envelope
from .solar import Sun
from .users import Users
from config import config
import math
import warnings

class Datahandler:
    """
    Abstract class for data handling.
    Collects data from input files, TEASER, User and Envelope.

    Attributes
    ----------
    site:
        Dict for site data, e.g. weather.
    time:
        Dict for time settings.
    district:
        List of all buildings within district.
    scenario_name:
        Name of scenario file.
    scenario:
        Scenario data.
    counter:
        Dict for counting number of equal building types.
    srcPath:
        Source path.
    dataPath:
        File path.
    """

    def __init__(self, scenario, scenario_name = "example", zip_code="80939"): 
        """
        Constructor of Datahandler class.
        - zip_code: zip code of buildings' location
        - scenario: data of every buildings in district as df
        Returns
        -------
        None.
        """
        self.site = {}
        self.time = {}
        self.district = []
        self.scenario_name = scenario_name
        self.scenario = None
        self.design_building_data = {}
        self.physics = {}

        self.dataPath = config.DISTGEN_DATA_PATH

        self.site["zip"] = zip_code
        self.site["timeZone"] = config.TIME_ZONE
        self.site["albedo"] = config.ALBEDO
        
        self.time["timeResolution"] = 3600
        self.time["holidays"] = config.HOLIDAYS
        self.time["initial_day"] = config.INITIAL_DAY
        self.time["dataLength"] = config.DATA_LENGTH

        # # %% load scenario file with building information
        # self.scenario = pd.read_csv(self.scenario_file_path + "/" + self.scenario_name + ".csv",
        #                             header=0, delimiter=";")
        # self.scenario["nb_occ"] = self.scenario["nb_occ"].apply(ast.literal_eval)
        # self.scenario["demand_per_flat"] = self.scenario["demand_per_flat"].apply(ast.literal_eval)
        self.scenario = scenario

        self.design_building_data["T_set_min"] = config.T_SET_MIN
        self.design_building_data["T_set_min_night"] = config.T_SET_MIN_NIGHT
        self.design_building_data["T_set_max"] = config.T_SET_MAX
        self.design_building_data["T_set_max_night"] = config.T_SET_MAX_NIGHT
        self.design_building_data["ventilation_rate"] = config.VENTILATION_RATE
        self.design_building_data["buildings_short"] = config.BUILDINGS_SHORT
        self.design_building_data["buildings_long"] = config.BUILDINGS_LONG
        self.design_building_data["retrofit_short"] = config.RETROFIT_SHORT
        self.design_building_data["retrofit_long"] = config.RETROFIT_LONG
        self.design_building_data["dhwload"] = config.DHWLOAD
        self.design_building_data["mean_drawoff_vol_per_day"] = config.MEAN_DRAWOFF_VOL


        self.physics["rho_air"] = config.RHO_AIR
        self.physics["c_p_air"] = config.C_P_AIR
        self.physics["rho_water"] = config.RHO_WATER
        self.physics["c_p_water"] = config.C_P_WATER


    def generateEnvironment(self, weatherData, site_data):
        """
        Load physical district environment - site and weather.

        Returns
        -------
        None.
        """

        # get weather data of interest
        [temp_sunDirect, temp_sunDiff, temp_temp] = weatherData

        self.time["timeSteps"] = int(self.time["dataLength"] / self.time["timeResolution"])

        # load the holidays
        self.time["holidays"] = self.time["holidays"]

        # interpolate input data to achieve required data resolution
        # transformation from values for points in time to values for time intervals
        self.site["SunDirect"] = temp_sunDirect
        self.site["SunDiffuse"] = temp_sunDiff
        self.site["T_e"] = temp_temp

        self.site["SunTotal"] = self.site["SunDirect"] + self.site["SunDiffuse"]


        # Filter data for the specific zip code
        filtered_data = site_data[site_data['Zip'] == self.site["zip"]]
        # extract the needed values
        self.site["altitude"] = filtered_data.iloc[0]['Altitude']
        self.site["location"] = [filtered_data.iloc[0]['Latitude'],filtered_data.iloc[0]['Longitude']]
        self.site["T_ne"] = filtered_data.iloc[0]['T_ne'] # norm outside temperature for calculating the design heat load
        self.site["T_me"] = filtered_data.iloc[0]['T_me'] # mean annual temperature for calculating the design heat load

        # Calculate solar irradiance per surface direction - S, W, N, E, Roof represented by angles gamma and beta
        global sun
        sun = Sun(filePath=self.dataPath)
        self.site["SunRad"] = sun.getSolarGains(initialTime=0,
                                        timeDiscretization=self.time["timeResolution"],
                                        timeSteps=self.time["timeSteps"],
                                        timeZone=self.site["timeZone"],
                                        location=self.site["location"],
                                        altitude=self.site["altitude"],
                                        beta=[90, 90, 90, 90, 0],
                                        gamma=[0, 90, 180, 270, 0],
                                        beamRadiation=self.site["SunDirect"],
                                        diffuseRadiation=self.site["SunDiffuse"],
                                        albedo=self.site["albedo"])

    def initializeBuildings(self):
        """
        Fill district with buildings from scenario file.

        Parameters
        ----------
        scenario_name: string, optional
            Name of scenario file to be read. The default is 'example'.

        Returns
        -------
        None.
        """

        """ BUILDING:
        building_features:  id;building;year;retrofit;construction_type;night_setback;area;heater;STC;EV;BAT;f_TES;f_BAT;f_EV;f_STC;
        unique_name:  "id"
        """
        
        name_pool = []

        # initialize buildings for scenario
        # loop over all buildings
        for id in self.scenario["id"]:
            try:
                # Check if ID is a number
                try:
                    # Try to convert id to float to check if it's numeric
                    float(id)
                except (ValueError, TypeError):
                    raise ValueError(f"Building ID '{id}' is not a number")

                # Create empty dict for observed building
                building = {}

                # Store features of the observed building
                building["buildingFeatures"] = self.scenario.loc[id]

                # %% Create unique building name
                # needed for loading and storing data with unique name
                # name is composed of building id, and building type
                name = str(id)
                # Check if the name is already in the district
                if name in name_pool:
                    raise ValueError(f"Building name '{name}' is not unique. ID collision detected.")
                name_pool.append(name)
                # Assign the unique name to the building
                building["unique_name"] = name

                # Append building to district
                self.district.append(building)

            except ValueError as e:
                # Handle the case where we have a duplicate name
                print(f"Error: {e}")
                print(f"The building ID must be a unique number to ensure proper identification and data tracking.")
                print(f"Building with ID {id} will be skipped and not added to the district")
                continue
            except Exception as e:
                # Handle any other unexpected errors
                print(f"Unexpected error processing building ID {id}: {e}")
                print(f"Building with ID {id} will be skipped and not added to the district.")
                continue


    def generateBuildings(self):
        """
        Load building envelope and user data.

        Returns
        -------
        None.
        """

        # %% load general building information
        # contains definitions and parameters that affect all buildings
        bldgs = self.design_building_data

        # %% create TEASER project
        # create one project for the whole district
        prj = Project(load_data=True)
        prj.name = self.scenario_name

        for building in self.district:
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
                if building["buildingFeatures"]["building"] not in ["SFH", "MFH", "TH", "AB"]: building["buildingFeatures"]["building"]="AB"
                if building["buildingFeatures"]["building"] == "AB" and building["buildingFeatures"]["year"]>1978: building["buildingFeatures"]["year"]=1978


            # convert short names into designation needed for TEASER
            building_type = \
                bldgs["buildings_long"][bldgs["buildings_short"].index(building["buildingFeatures"]["building"])]
            retrofit_level = \
                bldgs["retrofit_long"][bldgs["retrofit_short"].index(building["buildingFeatures"]["retrofit"])]

            
            if building["buildingFeatures"]["year"] < 1860: building["buildingFeatures"]["year"] = 1860
            if math.isnan(building["buildingFeatures"]["year"]): building["buildingFeatures"]["year"] = 1970
            # add buildings to TEASER project
            prj.add_residential(method='tabula_de',
                                usage=building_type,
                                name="ResidentialBuildingTabula",
                                year_of_construction=building["buildingFeatures"]["year"],
                                number_of_floors=building["buildingFeatures"]["floors"],
                                height_of_floors=3.125,
                                net_leased_area=building["buildingFeatures"]["area"],
                                construction_type=retrofit_level)
            

            # %% create envelope object
            # containing all physical data of the envelope
            building["envelope"] = Envelope(prj=prj,
                                            building_params=building["buildingFeatures"],
                                            construction_type=retrofit_level,
                                            physics=self.physics,
                                            design_building_data=self.design_building_data,
                                            file_path=self.dataPath)

            # %% create user object
            # containing number occupants, electricity demand,...
            building["user"] = Users(building_features=building["buildingFeatures"],
                                     area=building["buildingFeatures"]["area"])

            # %% calculate design heat loads
            # at norm outside temperature
            building["envelope"].heatload = building["envelope"].calcHeatLoad(site=self.site, method="design")
            # at bivalent temperature
            # building["envelope"].bivalent = building["envelope"].calcHeatLoad(site=self.site, method="bivalent")
            # at heating limit temperature
            # building["envelope"].heatlimit = building["envelope"].calcHeatLoad(site=self.site, method="heatlimit")
            # for drinking hot water
            if building["user"].building in {"SFH", "MFH", "TH", "AB"}:
                building["dhwload"] = bldgs["dhwload"][bldgs["buildings_short"].index(building["user"].building)] * \
                building["user"].nb_flats
            else:
                building["dhwload"] = bldgs["dhwload"][bldgs["buildings_short"].index(building["user"].building)] * \
                building["user"].nb_main_rooms

            # index = bldgs["buildings_short"].index(building["buildingFeatures"]["building"])
            # building["buildingFeatures"]["mean_drawoff_dhw"] = bldgs["mean_drawoff_vol_per_day"][index]
            # print(bldgs)
            # print(building)
            if isinstance(building["buildingFeatures"], pd.Series): building["buildingFeatures"] = building["buildingFeatures"].to_dict()
            index = bldgs["buildings_short"].index(building["buildingFeatures"]["building"])
            building["buildingFeatures"]["mean_drawoff_dhw"] = bldgs["mean_drawoff_vol_per_day"][index]


    def generateDemands(self, elecDemand):
        """
        Generate occupancy profile, heat demand, domestic hot water demand and heating demand.

        Parameters
        ----------
        calcUserProfiles: bool, optional
            True: calculate new user profiles.
            False: load user profiles from file.
            The default is True.
        saveUserProfiles: bool, optional
            True for saving calculated user profiles in workspace (Only taken into account if calcUserProfile is True).
            The default is True.

        Returns
        -------
        None.
        """
        dhw_res_profiles = {}
        space_heat_profiles = {}
        gain_profiles = {}

        for building in self.district:

            building["user"].gains = np.zeros(len(self.site["SunTotal"]))
            building["user"].dhw = np.zeros(len(self.site["SunTotal"]))
            building["user"].occ = np.zeros(len(self.site["SunTotal"]))
            building["user"].elec = np.zeros(len(self.site["SunTotal"]))

            building["user"].calcProfiles(site=self.site,
                                          holidays=self.time["holidays"],
                                          time_resolution=self.time["timeResolution"],
                                          time_horizon=self.time["dataLength"],
                                          building=building,
                                          elecDemand=elecDemand
                                          )

            building["envelope"].calcNormativeProperties(self.site["SunRad"], building["user"].gains)


            # calculate or load heating profiles
            building["user"].calcHeatingProfile(site=self.site,
                                                envelope=building["envelope"],
                                                holidays=self.time["holidays"],
                                                time_resolution=self.time["timeResolution"]
                                                )

            dhw_res_profiles[building["buildingFeatures"]["bus"]] = building["user"].dhw
            space_heat_profiles[building["buildingFeatures"]["bus"]] = building["user"].heat
            gain_profiles[building["buildingFeatures"]["bus"]] = building["user"].gains

        df_dhw_res = pd.DataFrame(dhw_res_profiles)/1000
        df_space_heat = pd.DataFrame(space_heat_profiles)/1000
        df_gains = pd.DataFrame(gain_profiles)/1000

        # print("Finished generating heat demands!")
        return df_space_heat, df_dhw_res, df_gains