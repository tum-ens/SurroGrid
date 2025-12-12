"""
This module contains the Availability class that creates the grid availability time series.
The class requires to be provided with the name of the driving consumption profile on which
will build up the new time series. It will also require the charging station power rating
and charging station probability distribution based on the location. The location can also
be associated with the trip purpose or destination. The time series indicate the location as
a state.

The Availability class reads the consumption profile. Every row contains a location, arriving
time, duration, distance and energy consumption from battery for driving. Following the
availability probability distribution, the tool looks at every row's location (state) and
sample a charging station. After the charging stations have been allocated to every row, the
tool tests if the allocation complies with the energy requirements. The state of charge (SOC)
is determined assuming an immediate charging strategy. This strategy consists of charging at
the maximum power rate whenever the current SOC is between 0 - 1. If the resulting SOC never
goes below zero. Then the allocation is considered correct, and the charging availability time
series is done; otherwise, a new allocation occurs.

The allocation of charging stations is carried out several times until a successful allocation
is reached or the maximum number of attempts is attained. If the latter, the file name will
contain FAIL word.

For more details see the article and cite:

.. code-block:: python

    @article{Gaete-Morales_2021,
    author={Gaete-Morales, Carlos and Kramer, Hendrik and Schill, Wolf-Peter and Zerrahn, Alexander},
    title={An open tool for creating battery-electric vehicle time series from empirical data, emobpy},
    journal={Scientific Data}, year={2021}, month={Jun}, day={11}, volume={8}, number={1}, pages={152},
    issn={2052-4463}, doi={10.1038/s41597-021-00932-9}, url={https://doi.org/10.1038/s41597-021-00932-9}}

See also the examples in the documentation https://diw-evu.gitlab.io/emobpy/emobpy

"""

import pandas as pd
import numpy as np
import uuid
import os
import pickle
import gzip
from numba import jit
from src.external.emobpy.constants import TIME_FREQ, CWD
from src.external.emobpy.tools import (check_for_new_function_name, _add_column_datetime, display_all)
from src.external.emobpy.init import copy_to_user_data_dir
from src.external.emobpy.logger import get_logger

logger = get_logger(__name__)


################################################################
# These functions are for grid availability profile creation ###
################################################################


class Availability:
    """
    Instance that represents a grid availability time series.
    It requires the driving consumption profile name (inpt) on which will build up
    the new time series and the database instace (db) where the consumption profiles
    are hosted.

    Args:
        inpt (str): driving consumption profile name
        db (DataBase()): class instance that contains the profiles

    Example:
        
        .. code-block:: python

            GA = Availability('ev1_abc_tesla3_def', DB)
            GA.set_scenario(charging_data)
            GA.run()
            GA.save_profile('path to folder')
    """

    def __init__(self, inpt, db):
        copy_to_user_data_dir()
        self.kind = "availability"
        self.input = inpt
        self._load_setting_driving(db)
        self.discharging_eff = self.vehicle.parameters["battery_discharging_eff"]
        self._set_vehicle_feature(
            self.vehicle.parameters["battery_cap"],
            self.vehicle.parameters["battery_charging_eff"],
        )
        self._set_battery_rules()

    def __getattr__(self, item):
        check_for_new_function_name(item)
        # if the return value is not callable, we get TypeError:

    def _load_setting_driving(self, database):
        """
        Loads setting data from DataBase.

        Then, the following attributes can be called
            self.df
            self.t
            self.totalrows
            self.hours
            self.freq
            self.refdate
            self.states

        Args:
            database (DataBase()): E.g. manager = DataBase(dir)
            "manager" is a class instance that contains the profiles

        Raises:
            ValueError: Raised if driving profile can not be found.
        """
        if database.db[self.input]:
            if database.db[self.input]["kind"] == "consumption":
                self.df = (
                    database.db[self.input]["profile"][
                        [
                            "hr",
                            "state",
                            "distance",
                            "consumption kWh",
                            "consumption kWh/100 km",
                        ]
                    ]
                    .fillna(0)
                    .copy()
                )
                self.t = database.db[self.input]["t"]
                self.totalrows = database.db[self.input]["totalrows"]
                self.hours = database.db[self.input]["hours"]
                self.freq = TIME_FREQ[self.t]["f"]
                self.refdate = database.db[self.input]["refdate"]
                self.states = database.db[self.input]["states"]
                self.vehicle = database.db[self.input]["vehicle"]

            else:
                raise ValueError(
                    "The driving profile {} can not be found in the database".format(
                        self.input
                    )
                )
        else:
            raise ValueError(
                "The driving profile {} can not be found in the database".format(
                    self.input
                )
            )

    def _set_vehicle_feature(self, battery_capacity, charging_eff):
        """
        Sets given battery_capacity and charging eff to object.

        Args:
            battery_capacity (int): Battery capacity in kwh.
            charging_eff (float): Charging efficiency in percent.
        """
        self.battery_capacity = battery_capacity
        self.charging_eff = charging_eff

    def _set_battery_rules(self, soc_init=1.0, soc_min=0.02, altern=[]):
        """
        Sets Battery rules to object.
        Args:
            soc_init (float [0-1], optional): Initiated state of charge of the battery. Defaults to 0.5.
            soc_min (float [0-1], optional): Minimum state of charge of the battery. Defaults to 0.02.
            altern (list, optional): kWh larger than battery_capacity. Defaults to [].
        """
        self.soc_init = soc_init
        self.soc_min = soc_min
        self.storage_altern = altern[:]

    def set_scenario(self, charging_data):
        """
        Sets given charging_data to object.

        Args:
            charging_data (dict): E.g.

                .. code-block:: python

                    {
                        'prob_charging_point' :
                            {'errands':  {'public':0.3,'none':0.7},
                            'escort':   {'public':0.3,'none':0.7},
                            'leisure':  {'public':0.3,'none':0.7},
                            'shopping': {'public':0.3,'none':0.7},
                            'home':     {'public':0.3,'none':0.7},
                            'workplace':{'public':0.0,'workplace':0.3,'none':0.7},
                            'driving':  {'none':1.0}
                            },
                        'capacity_charging_point' :
                            {'public':11,'home':1.8,'workplace':5.5,'none':0}
                    }
        """
        self.chargingdata = charging_data

    def _initial_conf(self):
        """
        Initialize configuration from self.chargingdata and creates a unique name.
        """
        self.prob_charging_point = self.chargingdata["prob_charging_point"]
        self.capacity_charging_point = self.chargingdata["capacity_charging_point"]
        self.name = self.input + "_avai_" + uuid.uuid4().hex[0:5]

    def _choose_charging_point(self, state):
        """
        Choose charging point depending on probability.

        Args:
            state (str): State of the vehicle.

        Returns:
            str: Name of the chosen charging point.
        """
        if state != "driving":
            self.chrg_points = [key for key in self.prob_charging_point[state].keys()]
            self.prob = [val for val in self.prob_charging_point[state].values()]
            self.rnd_name = np.random.choice(self.chrg_points, p=self.prob)
            return self.rnd_name
        else:
            return "none"

    def _choose_charging_point_fast(self, row):
        """
        Select a charging point.

        Args:
            row (row df.DataFrame): Timeseries.

        Raises:
            Exception: Long distance trip and battery capacity does not match.

        Returns:
            self.rnd_name: Chosen charging point.
        """

        if row["consumption kWh"] > 0.85 * self.battery_capacity:
            self.chrg_points = [key for key in self.prob_charging_point[row["state"]].keys()]
            if "none" in self.chrg_points:
                idx = self.chrg_points.index("none")
                del self.chrg_points[idx]
                if not self.chrg_points:
                    raise Exception(
                        f'This profile has a long distance trip ({row["consumption kWh"]} kWh), higher than battery '
                        f'capacity (0.85 * {self.battery_capacity}). Add fast charging stations')
                self.prob = [
                    val for val in self.prob_charging_point[row["state"]].values()
                ]
                del self.prob[idx]
                total = sum(self.prob)
                if total == 0:
                    self.rnd_name = self.chrg_points[0]
                    return self.rnd_name
                self.prob = [x / total for x in self.prob]
                self.rnd_name = np.random.choice(self.chrg_points, p=self.prob)
                return self.rnd_name
            else:
                if not self.chrg_points:
                    raise Exception(
                        f'This profile has a long distance trip ({row["consumption kWh"]} kWh), higher than battery '
                        f'capacity (0.85 * {self.battery_capacity}). Add fast charging stations')
                self.prob = [
                    val for val in self.prob_charging_point[row["state"]].values()
                ]
                self.rnd_name = np.random.choice(self.chrg_points, p=self.prob)
                return self.rnd_name
        else:
            self.chrg_points = [
                key for key in self.prob_charging_point[row["state"]].keys()
            ]
            self.prob = [val for val in self.prob_charging_point[row["state"]].values()]
            self.rnd_name = np.random.choice(self.chrg_points, p=self.prob)
            return self.rnd_name

    def _drawing_soc(self):
        """
        Drawing the state of charge of vehicle.
        """
        self.point = "driving"
        self.statesplusdrv = list(set(self.dt.loc[:, "state"]))
        self.pointcode = self.statesplusdrv.index(self.point)
        self.numpy_array3 = self.dt[["state"]].values.T
        self.arraystringstate = self.numpy_array3[0]
        self.arraycodestate = np.array(
            [self.statesplusdrv.index(s) for s in self.arraystringstate]
        )
        self.dt["consumption"] = pd.to_numeric(self.dt["consumption"], errors="raise")
        self.dt["charging_cap"] = pd.to_numeric(self.dt["charging_cap"], errors="raise")
        numpy_array = self.dt[["consumption", "charging_cap"]].values.T
        self.dt.loc[:, "soc"] = self._soc(
            self.pointcode,
            self.charging_eff,
            self.battery_capacity,
            self.soc_init,
            self.arraycodestate,
            *numpy_array,
            self.t,
        )

    @staticmethod
    @jit(nopython=True)
    def _soc(driving_code, charging_eff, battery_capacity, soc_init, state, consumption, charging_cap, t):
        """
        Calculate state of charge of vehicle.
        #TODO DOCSTRING
        Args:
            driving_code ([type]): [description]
            charging_eff ([type]): [description]
            battery_capacity ([type]): [description]
            soc_init ([type]): [description]
            state ([type]): [description]
            consumption ([type]): [description]
            charging_cap ([type]): [description]
            t ([type]): [description]

        Returns:
            [type]: Calculated soc.
        """
        soc = np.empty(consumption.shape)
        rows = soc.shape[0]
        for i in range(rows):
            if i == 0:
                zero = soc_init
                current_soc = (
                    zero
                    - consumption[i] / battery_capacity
                    + charging_cap[i] * t * charging_eff / battery_capacity
                )
                if current_soc > 1:
                    soc[i] = 1
                else:
                    soc[i] = current_soc
            else:
                zero = soc[i - 1]
                if state[i] == driving_code:
                    if zero == 1:
                        current_soc = zero - consumption[i] / battery_capacity
                        soc[i] = current_soc
                    else:
                        current_soc = (
                            zero
                            - consumption[i] / battery_capacity
                            + charging_cap[i] * t * charging_eff / battery_capacity
                        )
                        if current_soc > 1:
                            soc[i] = 1
                        else:
                            soc[i] = current_soc
                else:
                    current_soc = (
                        zero
                        - consumption[i] / battery_capacity
                        + charging_cap[i] * t * charging_eff / battery_capacity
                    )
                    if current_soc > 1:
                        soc[i] = 1
                    else:
                        soc[i] = current_soc
        return soc

    def _testing_soc(self):
        """
        Tests state of charge for errors.
        """
        self.failed_chrg = self.dt[self.dt["soc"] < self.soc_min].copy()
        if self.failed_chrg.empty:
            if self.dt["soc"].iloc[-1] >= self.soc_init:
                self.soc_end = round(self.dt["soc"].iloc[-1], 3)
                logger.info(f"soc_init:{str(round(self.soc_init, 3))} --> soc_end:{str(self.soc_end)}")
                self.success = True
                self.notation = "True"
                self.ready = True
            else:
                self.drivlist = self.dt[self.dt["state"] == "driving"].index.to_list()[
                    ::-1
                ]
                self.len = len(self.dt)
                for ix in self.drivlist:
                    self.dt.loc[ix, "consumption"] = 0.0
                    self._drawing_soc()
                    if self.dt["soc"].iloc[-1] >= self.soc_init:
                        break
                self.new_len = len(self.dt[:ix])
                self.proportion_ts_modified = round(self.new_len / self.len, 3)
                if self.dt["soc"].iloc[-1] >= self.soc_init:
                    self.stored_n += 1
                    self.stored_success_prop.append(self.proportion_ts_modified)
                    self.stored_success.append(self.dt.copy())
                    if self.stored_n == 3:
                        self.dt = self.stored_success[
                            max(
                                enumerate(self.stored_success_prop),
                                key=lambda tup: tup[1],
                            )[0]
                        ].copy()
                        self.proportion_ts_modified = max(enumerate(self.stored_success_prop), key=lambda tup: tup[1])[1]
                        self.success = True
                        self.notation = str(self.proportion_ts_modified)
                        self.ready = True
                        logger.debug(f"Consumption set zero for the last trips. Time steps share:{str(self.proportion_ts_modified)}")
                        self.soc_end = round(self.dt["soc"].iloc[-1], 3)
                        logger.info(f"soc_init:{str(round(self.soc_init, 3))} --> soc_end:{str(self.soc_end)}")

        if not self.ready:
            if self.n % 40 == 0:
                if self.n != 0:
                    logger.debug(f"still in while loop after {str(self.n)} iterations. Battery may be small, or few charging points available...")
            if self.n % 80 == 0:
                if self.n != 0:
                    if self.battopt:
                        logger.debug("Change battery capacity from {} kWh to {} kWh".format(self.battery_capacity, self.battopt[0]))
                        self.battery_capacity = self.battopt[0]
                        self.battopt.remove(self.battopt[0])
                    else:
                        self.success = False
                        self.notation = "Faulty"  # save anyway but it must be verified
                        self.name += "_FAIL"
                        logger.info(" ----- !!! UNSUCCESSFUL profile creation !!! ----- please check this '{}', it may need to increase battery capacity or soc init is too low".format(self.name))
                        self.ready = True

    def _fill_rows(self):
        """
        Sets data for many attributes and is executed in self.run.
        """
        self.repeats = [
            "hr",
            "state",
            "charging_point",
            "charging_cap",
            "distance",
            "consumption",
        ]
        self.fixed = ["consumption kWh"]
        self.copied = []
        self.calc = ["dayhrs"]
        self.same = []

        self.dt = pd.DataFrame(columns=self.db.columns)
        self.dt.loc[:, "hh"] = np.arange(0, self.hours, self.t)

        # Start New version, which works for 1s-based profiles:
        temp_timeseries = [round(num*3600) for num in self.dt["hh"]]
        temp_db = [round(num*3600) for num in self.db["hr"]]
        temp_intersection_list = list(set(temp_timeseries).intersection(temp_db))

        self.idx = []
        for i in temp_intersection_list:
            self.idx.append(temp_timeseries.index(i))
        self.idx = np.sort(self.idx).tolist()
        # End new version

        self.mixed = self.repeats + self.fixed + self.copied
        for r in self.mixed:
            self.val = self.db[r].values.tolist()
            self.dt.loc[self.idx, r] = self.val
        self.dt.loc[self.totalrows - 1, "state"] = self.db["state"].iloc[-1]
        self.dt.loc[self.totalrows - 1, "hr"] = self.dt["hh"][self.totalrows - 1]
        self.rp = self.dt[::-1].reset_index(drop=True)
        self.rp.loc[:, self.repeats] = self.rp[self.repeats].fillna(method="ffill")
        self.rp.loc[:, self.fixed] = self.rp[self.fixed].fillna(0)
        self.dt = self.rp[::-1].reset_index(drop=True)
        for sm in self.same:
            self.dt.loc[:, sm] = self.db[sm].values.tolist()[0]
        for cal in self.calc:
            self.dt.loc[:, cal] = self.dt["hh"].apply(lambda x: x % 24)
        self.dt.loc[:, "count"] = self.dt.groupby(["hr", "state"])[
            "consumption"
        ].transform("count")
        self.dt.loc[:, "consumption"] = (
            self.dt.loc[:, "consumption"] / self.dt.loc[:, "count"]
        )
        self.dt.loc[:, "distance"] = (
            self.dt.loc[:, "distance"] / self.dt.loc[:, "count"]
        )
        # convert this section to numba
        flag = False
        for i, row in self.dt.iterrows():
            if flag:
                if row["state"] == "driving":
                    flag = True
                    if self.cumcons != 0 and self.cumchrg == 0:
                        self.cumcons += row["consumption"]
                        if self.cumcons < self.battery_capacity * 0.50:
                            self.dt.loc[i, "charging_cap"] = 0
                            self.dt.loc[i, "charging_point"] = "none"
                            self.cumchrg = 0
                        else:
                            self.cumchrg += row["charging_cap"] * self.t
                            self.cumcons = 0
                    else:
                        self.cumchrg += row["charging_cap"] * self.t
                        if self.cumchrg > self.battery_capacity * 0.5:
                            self.cumchrg = 0
                            self.cumcons += 0.001
                        else:
                            pass
                else:
                    flag = False
            elif row["state"] == "driving":
                flag = True
                self.cumcons = row["consumption"]
                if self.cumcons < self.battery_capacity * 0.65:
                    self.dt.loc[i, "charging_cap"] = 0
                    self.dt.loc[i, "charging_point"] = "none"
                    self.cumchrg = 0
                else:
                    self.cumchrg = row["charging_cap"] * self.t
                    self.cumcons = 0

    def run(self):
        """
        No input required.
        Once it finishes the following attributes can be called.

        Attributes:

        - kind
        - input
        - chargingdata
        - battery_capacity
        - charging_eff
        - discharging_eff
        - soc_init
        - soc_min
        - storage_altern
        - profile
        - timeseries
        - success
        - name
        - proportion_ts_modified
        """
        self._initial_conf()
        self.battopt = self.storage_altern[:]
        self.battopt.sort()
        self.ready = False
        self.proportion_ts_modified = 1.0
        self.stored_success = []
        self.stored_success_prop = []
        self.stored_n = 0
        self.n = 0
        self.df.loc[:, "dayhrs"] = self.df["hr"] % 24
        self.df.loc[:, "consumption"] = self.df["consumption kWh"]
        count_in_loop = 0
        logger.debug(f"Entering to a while loop for {self.name}")
        while True:
            self.db = self.df.copy()
            self.db.loc[:, "charging_point"] = self.df["state"].apply(
                lambda state: self._choose_charging_point(state)
            )
            self.db.loc[
                self.df[self.df["state"] == "driving"].index, "charging_point"
            ] = self.df[self.df["state"] == "driving"].apply(
                self._choose_charging_point_fast, axis=1
            )
            self.db.loc[:, "charging_cap"] = self.db["charging_point"].apply(
                lambda charging_point: self.capacity_charging_point[charging_point]
            )
            logger.debug(f"Fill rows for {self.name}")
            self._fill_rows()
            logger.debug(f"Draw SOC for {self.name}")
            self._drawing_soc()
            logger.debug(f"Testing SOC for {self.name}")
            self._testing_soc()
            if self.ready:
                break
            else:
                self.n += 1
            count_in_loop += 1
            if count_in_loop%100 == 0:
                logger.debug(f"{count_in_loop} iterations in loop")
                
        self.profile = self.dt[
            [
                "hh",
                "state",
                "distance",
                "consumption",
                "charging_point",
                "charging_cap",
                "soc",
                "consumption kWh",
                "count",
            ]
        ].copy()
        self.timeseries = _add_column_datetime(
            self.profile.copy(), self.totalrows, self.refdate, self.t
        )
        self.timeseries = self.timeseries[["hh","state","distance","consumption","charging_point","charging_cap","soc"]]

        to_rem = list(self.__dict__.keys())[:]

        keep_attr = [
            "kind",
            "input",
            "chargingdata",
            "battery_capacity",
            "charging_eff",
            "discharging_eff",
            "soc_init",
            "soc_min",
            "soc_end",
            "storage_altern",
            "profile",
            "timeseries",
            "success",
            "name",
            "proportion_ts_modified",
            "totalrows",
            "refdate",
            "t",
            "notation",
        ]

        for r in keep_attr:
            if r in to_rem:
                to_rem.remove(r)
        for attr in to_rem:
            self.__dict__.pop(attr, None)
        del to_rem
        logger.info("Profile done: " + self.name)

    def save_profile(self, folder, description=" "):
        """
        Saves object profile as a pickle file.

        Args:
            folder (str): Where the files will be stored. Folder is created in case it does not exist.
            description (str, optional): Description which can be saved in object attribute. Defaults to " ".
        """
        self.description = description
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, self.name + ".pickle")
        with gzip.open(filepath, "wb") as file:
            pickle.dump(self.__dict__, file)
        logger.debug("Profile saved : " + filepath)
        logger.info(" ")
        logger.info("See Log files")
        for handler in logger.handlers:
            if handler.__class__.__name__ == 'FileHandler':
                logger.info(handler.baseFilename)
        try:
            display_all()
        except:
            pass
