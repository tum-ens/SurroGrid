# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pylightxl as xl
import richardsonpy.classes.occupancy as occ_residential
import richardsonpy.functions.change_resolution as cr
import OpenDHW
import src.external.districtgenerator.functions.change_resolution as chres

class Profiles:
    """
    Profile class.
    Calculating user related profiles of a building or flat.

    Parameters
    ----------
    number_occupants : integer
        Number of occupants who live in the house or flat.
    number_occupants_building : integer
        Number of occupants who live in the building.
    initial_day : integer
        Day of the week with which the generation starts.
        1-7 for monday-sunday.
    nb_days : integer
        Number of days for which a stochastic profile is generated.
    time_resolution : integer
        resolution of time steps of output array in seconds.

    Attributes
    ----------
    activity_profile : array-like
        Numpy-array with active occupants 10-minutes-wise.
    occ_profile : array-like
        Stochastic occupancy profile.
    app_load : array-like
        Electric load profile of appliances in W.
    light_load : array-like
        Electric load profile of lighting in W.
    """

    def __init__(self, number_occupants, number_occupants_building, initial_day, nb_days, time_resolution, building):
        """
        Constructor of Profiles class.

        Returns
        -------
        None.
        """

        self.number_occupants = number_occupants
        self.number_occupants_building = number_occupants_building
        self.initial_day = initial_day
        self.nb_days = nb_days
        self.time_resolution = time_resolution

        self.building = building
        self.activity_profile = []
        self.occ_profile = []
        self.occ_profile_building = []
        self.building_profiles = {}
        self.temperature_difference = []
        self.light_load = []
        self.app_load = []

        self.generate_activity_profile_residential()

    def generate_activity_profile_residential(self):
        """
        Generate a stochastic activity profile
        (on base of ridchardsonpy).

        Parameters
        ----------
        number_occupants : integer
            Number of occupants who live in the house or flat.
        initial_day : integer
            Day of the week with which the generation starts
            1-7 for monday-sunday.
        nb_days : integer
            Number of days for which a stochastic profile is generated.

        Returns
        -------
        None.
        """

        if self.building in {"SFH", "TH", "MFH", "AB"}:
            activity = occ_residential.Occupancy(self.number_occupants, self.initial_day, self.nb_days)
            self.activity_profile = activity.occupancy

    def generate_occupancy_profiles_residential(self):
        """
        Generate stochastic occupancy profiles for a district for calculating internal gains.
        Change time resolution of 10 min profiles to required resolution.

        Parameters
        ----------
        time_resolution : integer
            Resolution of time steps of output array in seconds.
        activity_profile : array-like
            Numpy-arry with active occupants 10-minutes-wise.

        Returns
        -------
        self.occ_profile : array-like
            Number of present occupants.
        """

        tr_min = int(self.time_resolution/60)
        sia_profile_daily_min = np.concatenate((np.ones(60*8),
                                                np.zeros(60*13),
                                                np.ones(60*3)),
                                                axis=None)

        # generate array for minutely profile
        activity_profile_min = np.zeros(len(self.activity_profile) * 10)
        # generate array for time adjusted profile
        self.occ_profile = np.zeros(int(len(self.activity_profile) * 10 / tr_min))

        # append minutely sia profiles until nb_days is reached
        sia_profile = []
        while len(sia_profile) < len(activity_profile_min):
            sia_profile = np.concatenate((sia_profile, sia_profile_daily_min), axis=None)
        sia_profile = sia_profile * max(self.activity_profile)

        # calculate minutely profile
        for t in range(len(activity_profile_min)):
            activity_profile_min[t] = max(self.activity_profile[int(t/10)], sia_profile[t])
        for t in range(len(self.occ_profile)):
            self.occ_profile[t] = np.round(np.mean(activity_profile_min[(t * tr_min):(t * tr_min + tr_min)]))

        return self.occ_profile

    def loadProbabilitiesDhw(self):
        """
        Load probabilities of dhw usage.

        Returns
        -------
        None.
        """

        #  Define src path
        src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        filename = 'dhw_stochastical.xlsx'
        path_DHW = os.path.join(src_path, 'districtgenerator', 'data', filename)

        # Initialization
        profiles = {"we": {}, "wd": {}}
        # book = xlrd.open_workbook(filename)
        book = xl.readxl(fn=path_DHW)
        sheetnames = book.ws_names

        # Iterate over all sheets
        for sheetname in sheetnames:
            # sheet = xl.readxl(fn=filename, ws=sheetname)

            # Read values
            values = [book.ws(ws=sheetname).index(row=i, col=1) for i in
                      range(1, 1441)]  # [sheet.cell_value(i,0) for i in range(1440)]

            # Store values in dictionary
            if sheetname in ("wd_mw", "we_mw"):
                profiles[sheetname] = np.array(values)
            elif sheetname[1] == "e":
                profiles["we"][int(sheetname[2])] = np.array(values)
            else:
                profiles["wd"][int(sheetname[2])] = np.array(values)

        # Load profiles
        self.prob_profiles_dhw = profiles

    def generate_dhw_profile(self, building, holidays):
        """
        Generate a stochastic dhw profile
        (on base of DHWclac).
        https://www.uni-kassel.de/maschinenbau/institute/thermische-energietechnik/fachgebiete/solar-und-anlagentechnik/downloads
        https://github.com/RWTH-EBC/OpenDHW

        Parameters
        ----------
        s_step : integer
            Resolution of time steps of output array in seconds.
        Categories: 1 or 4
            Either one or four categories with different mean volume rates, tapping times and frequencies can be defined.
        occupancy: integer
            Maximum number of occupants in this building.
        mean_drawoff_vol_per_day : array-like
            Total mean daily draw-off volume per person per day in liter.
        temp_dT : array-like
        The temperature difference (ΔT) between the cold water and the water at the tapping point (mixed water).

        Returns
        -------
        dhw_heat : array-like
            Numpy-array with heat demand of dhw consumption in W.
        """

        temperatur_mixed_water = []
        temperatur_cold_water = []
        for day in range(365):
            temperatur_mixed_water.append(45 + (3 * np.cos(math.pi * (2 / 365 * (day) - 2 * 355 / 365))))
            # This formula introduces a seasonal fluctuation to take account of the fluctuations in the desired water temperature throughout the year.
            # The amplitude is ±3°C, reflecting higher hot water temperature requirements during colder months (winter) and lower during warmer months (summer).

            temperatur_cold_water.append(10 + (7 * np.cos(math.pi * (2 / 365 * (day) - 2 * 225 / 365))))
            # This formula introduces a seasonal fluctuation to take account of the fluctuations in the cold water temperature throughout the year.
            # The water temperature is assumed to be equal to the ground temperature at a depth of 1.5 m.
            # Source: M. Böhme, F. Böttcher. Bodentemperaturen im Klimawandel: Auswertungen der Messreihe der Säkularstation Potsdam
            #         https://www.dwd.de/DE/leistungen/klimastatusbericht/publikationen/ksb2011_pdf/ksb2011_art2.pdf?__blob=publicationFile&v=1

        temperature_difference_day = [T_out - T_in for T_out, T_in in
                                      zip(temperatur_mixed_water, temperatur_cold_water)]

        temperature_difference = [T for T in temperature_difference_day for _ in range(24)]
        self.temperature_difference = chres.changeResolution(temperature_difference, 3600, self.time_resolution,
                                                             "mean")

        dhw_profile = OpenDHW.generate_dhw_profile(
            s_step=60,
            categories=1,
            occupancy=self.number_occupants if self.building in {"SFH", "TH", "MFH", "AB"} else self.number_occupants_building,
            building_type=self.building,
            weekend_weekday_factor=1.2 if self.building in {"SFH", "TH", "MFH", "AB"} else 1,
            holidays = holidays,
            mean_drawoff_vol_per_day=building["buildingFeatures"]["mean_drawoff_dhw"]
        )

        dhw_timeseries = OpenDHW.resample_water_series(dhw_profile, self.time_resolution)
        dhw_heat = OpenDHW.compute_heat(timeseries_df=dhw_timeseries, temp_dT=self.temperature_difference)

        return dhw_heat["Heat_W"].values

    def generate_el_profile_residential(self, holidays, irradiance, el_wrapper, annual_demand, do_normalization=True):
        """
        Generate electric load profile for one household efficiently.
        """
        # Check time resolution compatibility
        timesteps_irr = int(self.nb_days * 3600 * 24 / len(irradiance))
        if self.time_resolution != timesteps_irr:
            msg = f'Time discretization mismatch: {self.time_resolution} vs {timesteps_irr} seconds.'
            raise AssertionError(msg)

        _timestep_rich = 60  # 1-minute resolution

        # Precompute entire interpolated irradiance to 1-minute resolution
        time_old = np.arange(len(irradiance)) * self.time_resolution
        time_new = np.arange(0, len(irradiance) * self.time_resolution, _timestep_rich)
        irradiance_minutewise = np.interp(time_new, time_old, irradiance)

        # Verify interpolated data length matches expected days
        expected_minutes = self.nb_days * 1440
        if len(irradiance_minutewise) != expected_minutes:
            raise ValueError("Interpolated irradiance length mismatch.")

        # Preallocate arrays for results
        n_days = self.nb_days
        demand = np.zeros((n_days, 1440))
        light_load = np.zeros((n_days, 1440))
        app_load = np.zeros((n_days, 1440))

        for i in range(n_days):
            # Determine if non-working day
            day_check = i + self.initial_day
            not_working_day = (day_check % 7 in (0, 6)) or (day_check in holidays)

            # Slice precomputed 1-minute irradiance for the day
            start, end = i * 1440, (i + 1) * 1440
            irrad_day = irradiance_minutewise[start:end]

            # Extract 10-minute occupancy data
            occ_start, occ_end = 144 * i, 144 * (i + 1)
            current_occupancy = self.activity_profile[occ_start:occ_end]

            # Simulate power usage
            el_p_curve, light_p_curve, app_p_curve = el_wrapper.power_sim(
                irradiation=irrad_day,
                weekend=not_working_day,
                day=i,
                occupancy=current_occupancy
            )

            # Store results directly in preallocated arrays
            demand[i] = el_p_curve
            light_load[i] = light_p_curve
            app_load[i] = app_p_curve

        # Flatten arrays and adjust resolution
        res = demand.ravel()
        self.light_load = light_load.ravel()
        self.app_load = app_load.ravel()

        # Adjust time resolution
        loadcurve = cr.change_resolution(res, _timestep_rich, self.time_resolution)
        self.light_load = cr.change_resolution(self.light_load, _timestep_rich, self.time_resolution)
        self.app_load = cr.change_resolution(self.app_load, _timestep_rich, self.time_resolution)

        # Normalize to annual demand if required
        if do_normalization:
            # Convert to energy (Ws) and sum
            energy_light = self.light_load * self.time_resolution
            energy_app = self.app_load * self.time_resolution

            # Calculate factors using vectorized operations
            total_days = self.nb_days / 365
            curr_light = np.sum(energy_light) / 3_600_000
            curr_app = np.sum(energy_app) / 3_600_000

            factor_light = (0.1 * annual_demand * total_days) / curr_light
            factor_app = (0.9 * annual_demand * total_days) / curr_app

            # Apply normalization factors
            self.light_load *= factor_light
            self.app_load *= factor_app
            loadcurve = self.light_load + self.app_load

        return loadcurve


    def generate_gain_profile_residential(self):
        """
        Generate profile of internal gains

        Parameters
        -------
        personGain : float
            Heat dissipation of one person
            Source: SIA 2024/2015 D - Raumnutzungsdaten für Energie- und Gebäudetechnik
        lightGain : float
            share of waste heat (LED)
            Source: Elsland, Rainer ; Peksen, Ilhan ; Wietschel, Martin: Are Internal Heat
            Gains Underestimated in Thermal Performance Evaluation of Buildings? In: Energy Procedia
            62 (2014), January, 32–41.
        appGain :
            share of waste heat (assumed)
            Source: Elsland, Rainer ; Peksen, Ilhan ; Wietschel, Martin: Are Internal Heat
            Gains Underestimated in Thermal Performance Evaluation of Buildings? In: Energy Procedia
            62 (2014), January, 32–41.
        occ_profile : float
             stochastic occupancy profiles for a district.
        app_load : array-like
            Electric load profile of appliances in W.
        light_load : array-like
            Electric load profile of lighting in W.

        Returns
        -------
        gains : array-like
            Internal gain of each flat.
        """

        personGain = 70.0  # [Watt]
        lightGain = 0.65
        appGain = 0.33

        gains = self.occ_profile * personGain + self.light_load * lightGain + self.app_load * appGain

        return gains
