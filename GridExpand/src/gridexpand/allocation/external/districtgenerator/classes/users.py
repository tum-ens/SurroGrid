# -*- coding: utf-8 -*-
import numpy as np
from .profils import Profiles
import gridexpand.allocation.external.districtgenerator.functions.heating_profile_5R1C as heating


class Users:
    """
    Building Users class describing the number of occupants and their configs.

    Parameters
    ----------
    building : string
        Building type according to TABULA database.
    area : integer
        Total floor area of the building [m²].

    Attributes
    ----------
    building : string
        Building type according to TABULA database.
    nb_flats : integer
        Number of flats in building.
    annual_el_demand_per_flat : array-like
        Annual electricity consumption in dependency of the building type and the number of occupants.
    lighting_index : integer
        This index defines the lighting configuration of the household.
    el_wrapper : object
        This objects holds information about the lighting and appliance configuration.
    nc_occ : list
        List with the number of occupants for each flat of the current building.
    occ : array-like
        Occupancy profile for each flat of the current building.
    dhw : array-like
        Drinking hot water profile for each building.
    elec : array-like
        Electrical demand for each building.
    gains : array-like
        Internal gains for each building.
    heat : array-like
        Heat demand for each building.
    """

    def __init__(self, building_features, area):
        """
        Constructor of Users class.

        Returns
        -------
        None.
        """

        self.building = building_features["building"]
        self.nb_flats = None
        self.annual_el_demand_per_flat = None
        self.annual_el_demand = None
        self.annual_heat_demand = None
        self.annual_dhw_demand = None
        self.annual_cooling_demand = None
        self.lighting_index = []
        self.el_wrapper = []
        self.nb_occ = []
        self.occ = None
        self.dhw = None
        self.elec = None
        self.gains = None
        self.heat = None
        self.cooling = None

        # GridExpand: flats and occupants come from the component manifest;
        # the upstream samplers and the appliance/lighting wrappers are removed.
        self.nb_flats = building_features["nb_flat"]
        self.nb_occ = building_features["nb_occ"]

    def calcProfiles(self, site, holidays, time_resolution, time_horizon, building, elecDemand, initial_day=4):
        """
        Calculate profiles for every flat and summarize them for the whole building

        Parameters
        ----------
        site: dict
            Site data, e.g. weather.
        time_resolution : integer
            Resolution of time steps of output array in seconds.
        time_horizon : integer
            Time horizon for which a stochastic profile is generated.
        initial_day : integer, optional
            Day of the week with which the generation starts.
            1-7 for monday-sunday. The default is 1.

        Returns
        -------
        None.
        """
        
        time_day = 24 * 60 * 60
        nb_days = int(time_horizon/time_day)
        if self.building in {"SFH", "TH", "MFH", "AB"}:

            self.occ = np.zeros(int(time_horizon / time_resolution))
            self.dhw = np.zeros(int(time_horizon / time_resolution))
            self.elec = np.zeros(int(time_horizon / time_resolution))
            self.gains = np.zeros(int(time_horizon / time_resolution))
            self.annual_dhw_demand = 0

            if building["buildingFeatures"]["NWG"] == 0:
                for j in range(self.nb_flats):
                    temp_obj = Profiles(number_occupants=self.nb_occ[j], number_occupants_building=sum(self.nb_occ),
                                        initial_day=initial_day, nb_days=nb_days, time_resolution=time_resolution,
                                        building=self.building)

                    self.dhw = self.dhw + temp_obj.generate_dhw_profile(building=building, holidays=holidays)
                    self.annual_dhw_demand += self.dhw

                    # Occupancy profile in a flat
                    self.occ = self.occ + temp_obj.generate_occupancy_profiles_residential()

            self.gains = self.generate_gain_profile_elec_given((elecDemand[building["buildingFeatures"]["bus"]]*1000).to_numpy().flatten())


    def generate_gain_profile_elec_given(self,elecDemand_i):
        personGain = 70.0  # [Watt]
        lightGain = 0.65
        appGain = 0.33

        gains = self.occ * personGain + (lightGain*0.1 + appGain*0.9)*elecDemand_i

        return gains

    def calcHeatingProfile(self, site, envelope, holidays, time_resolution):
        """
        Calculate heat demand for each building.

        Parameters
        ----------
        site: dict
            Site data, e.g. weather.
        envelope: object
            Containing all physical data of the envelope.
        time_resolution : integer
            Resolution of time steps of output array in seconds.
        Q_HC : float
            Heating (positive) or cooling (negative) load for the current time
            step in Watt.

        Returns
        -------
        None.
        """

        dt = time_resolution / (60 * 60)
        # calculate the temperatures (Q_HC, T_op, T_m, T_air, T_s)
        (Q_H, Q_C, T_op, T_m, T_i, T_s) = heating.calc(envelope, site["T_e"], holidays, dt, self.building)
        # heating and cooling loads for the current time step in Watt
        self.heat = Q_H
        self.cooling = Q_C
        self.annual_heat_demand = np.sum(Q_H)
        self.annual_cooling_demand = np.sum(Q_C)
