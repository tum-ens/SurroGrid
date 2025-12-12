"""
Mobility class consists of creating individual driver time series that contains vehicle location
and distance travelled at every time step. The vehicle's locations are a group of common places
obtained from mobility data.

Our approach consists of using three probability distributions from which to extract relevant
features by samplings, such as the number of trips per day,  the destination, departure time,
trip distance and trip duration. Time steps and total time for every time series are parameters
that have to be provided in advance.

To reach consistent and feasible mobility patterns, a set of rules can also be provided,
for instance, establishing home as the destination of the last trip of the day.

Two different approaches have been developed—commuters and free-time drivers (non-commuters).
Commuters are drivers that go during the weekdays to the workplace.

An instance of a mobility class contains a driving profile and a time series of one driver.
The instance can be saved in a pickle file to access it later on and create other time series,
such as consumption time series, grid availability, and eventually the grid demand.

For more details see the article and cite:

.. code-block:: python

    @article{Gaete-Morales_2021,
    author={Gaete-Morales, Carlos and Kramer, Hendrik and Schill, Wolf-Peter and Zerrahn, Alexander},
    title={An open tool for creating battery-electric vehicle time series from empirical data, emobpy},
    journal={Scientific Data}, year={2021}, month={Jun}, day={11}, volume={8}, number={1}, pages={152},
    issn={2052-4463}, doi={10.1038/s41597-021-00932-9}, url={https://doi.org/10.1038/s41597-021-00932-9}}

See also the examples in the documentation https://diw-evu.gitlab.io/emobpy/emobpy

"""

from collections import Counter
import pickle
import uuid
import copy
import gzip
import os
import time
import yaml
from numba import jit, boolean, prange
import numpy as np
import pandas as pd
from src.external.emobpy.constants import OPERATORS, WEEKS, RULE, CWD
from src.external.emobpy.tools import (check_for_new_function_name, _add_column_datetime, cmp, mobility_progress_bar, display_all)
from src.external.emobpy.logger import get_logger
from src.external.emobpy.init import copy_to_user_data_dir

logger = get_logger(__name__)


######################################################
# These functions are for driving profile creation ###
######################################################


@jit(nopython=True)
def rand_choice_nb(arr, prob):
    """
    Choose random sample from numpy array with a explicit probability.

    Args:
        arr (numpy.array): A 1D numpy array of values to sample from.
        prob (numpy.array): A 1D numpy array of probabilities for the given samples.

    Returns:
        int or float: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.rand(), side="right")]


@jit(nopython=True)
def distance_duration_sample(km_array_, duration_array_, km_dur_prob_array, km_sorted_, duration_sorted):
    """
    #TODO DOCSTRING
    Randomly choose a distance duration based on distance and probability distribution.
    Args:
        km_array_ ([type]): [description]
        duration_array_ ([type]): [description]
        km_dur_prob_array ([type]): [description]
        km_sorted_ ([type]): [description]
        duration_sorted ([type]): [description]

    Returns:
        [type]: [description]
    """
    km_array = (km_array_ * 1000).astype(np.int32)
    km_sorted = (km_sorted_ * 1000).astype(np.int32)
    duration_array = duration_array_.astype(np.int32)

    flag = 0
    idxs = np.arange(km_array.shape[0])
    idx = rand_choice_nb(
        idxs, prob=km_dur_prob_array
    )  # select a range of distances based on destinations and their probability distribution.
    km = km_array[idx]
    dur = duration_array[idx]

    km_sorted_idx = np.where(km_sorted == km)[0][0]
    dur_sorted_idx = np.where(duration_sorted == dur)[0][0]

    if km_sorted_idx == 0:
        km_before = km
        dur_before = dur
        flag = 1
    else:
        if dur_sorted_idx == 0:
            km_before = km
            dur_before = dur
            flag = 1
        else:
            km_before = km_sorted[km_sorted_idx - 1]
            dur_before = duration_sorted[dur_sorted_idx - 1]

    if flag:
        return km_before / 1000, dur_before

    amount = km - km_before + 1000
    if amount < 2000:
        amount = 2
    else:
        amount = np.int32(amount / 1000)

    kms_between = np.linspace(km_before, km, amount)
    same_prob = np.ones(kms_between.shape[0]) / amount
    km_new = rand_choice_nb(kms_between, prob=same_prob)
    dur_new = np.interp(km_new, [km_before, km], [dur_before, dur])
    return km_new / 1000, dur_new


@jit(nopython=True)
def distances_collection(nr_trips, km_array, duration_array, km_dur_prob_array, km_sorted, duration_sorted,
                        edge_dist=-1, edge_dur=-1, edge_nr=0, ):
    """
    #TODO DOCSTRING
    Sample distances from nba - grams.

    Args:
        nr_trips ([type]): [description]
        km_array ([type]): [description]
        duration_array ([type]): [description]
        km_dur_prob_array ([type]): [description]
        km_sorted ([type]): [description]
        duration_sorted ([type]): [description]
        edge_dist (int, optional): [description]. Defaults to -1.
        edge_dur (int, optional): [description]. Defaults to -1.
        edge_nr (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    # while True:
    distances = np.empty(nr_trips, dtype=np.float64)
    durations = np.empty(nr_trips, dtype=np.float64)
    j = 0
    if edge_dist != -1:
        for _ in range(edge_nr):
            distances[j] = edge_dist
            durations[j] = edge_dur
            j += 1
    for i in range(j, nr_trips):
        distances[i], durations[i] = distance_duration_sample(
            km_array, duration_array, km_dur_prob_array, km_sorted, duration_sorted
        )
        # TODO:
        # this must be improved, find another way to select distance for trips.
        # Bearing in mind that overall travelled distance in a day are consistent for each trip.
        # if distances.mean()*0.5 >= distances.max()-distances.min(): # distance consistency check
        # break
    return distances, durations


def days_week_sequence(reference_date, weeks):
    """
    This determines the day of the week of the reference date and sets the day-sequence.
    It helps to differentiate the statistics from week working days to weekends.

    Args:
        reference_date (str): Reference date in form of '01/01/2020'
        weeks (dict): Contains tag, tag type and day code per day ID.  

    Returns:
        list: List of day IDs (Integers) indicating the sequence of the week.  
    """
    df = pd.DataFrame({"ref_dates": [reference_date]})
    df["ref_dates"] = pd.to_datetime(df["ref_dates"])
    df["day_of_week"] = df["ref_dates"].dt.day_name()
    day_of_week = df["day_of_week"].values[0]
    for key, val in weeks.items():
        if val["day"] == day_of_week:
            day_numb = key
    sequence = [k % 7 for k in range(day_numb, 7 + day_numb)]
    return sequence


@jit(nopython=True)
def make_list(x, factor):
    """
    Make a list of integers representing a number x .

    Args:
        x (float): Number which is repeated in the list.
        factor (int): Len of the produced list.

    Returns:
        list: List which contains value x for factor times.
    """
    return [x / factor for _ in range(int(factor))]


@jit(nopython=True)
def make_interval(x, factor, time_step):
    """
    Make a new interval for the given factor.

    Args:
        x (float): Start value.
        factor (int): Number of repetition.
        time_step (float): Time step for each repetition.

    Returns:
        [type]: [description]
    """
    y = x
    ret = []
    for _ in range(int(factor)):
        ret.append(y)
        y += time_step
    return ret


def expand_table(df, factor, time_step):
    """
    Expands a table (DataFrame) by a factor. The table can then be more granular.
    Args:
        df ([pd.DataFrame]): Table that is to be extended.
        factor (float): By how many times the table should be extended.
        time_step (float): The new time_step that should apply.

    Returns:
        [pd.DataFrame]: Table in an expanded format.
    """
    df = df.set_index("days").copy()
    for j in df.columns.difference(["time"]):
        df.loc[:, j] = df[j].apply(make_list, factor=factor)
    df.loc[:, "time"] = df["time"].apply(
        make_interval, factor=factor, time_step=time_step
    )
    dc_col = {}
    for col in df.columns:
        dc_col[col] = df[col].explode()
    new_df = pd.DataFrame(dc_col).reset_index()
    return new_df


@jit(nopython=True, parallel=True, fastmath=True)
def nb_isin(A, C):
    """
    #TODO DOCSTRING
    Return nb in - place of nb in c .

    Args:
        A ([type]): [description]
        C ([type]): [description]

    Returns:
        [type]: [description]
    """
    B = np.empty(A.shape[0], dtype=boolean)
    s = set(C)
    for i in prange(A.shape[0]):
        B[i] = A[i] in s
    return B


@jit(nopython=True)
def creation_unsorted_trips_nb(
        Rnd_week_trips,
        km_array,
        duration_array,
        km_dur_prob_array,
        km_sorted,
        duration_sorted,
        t,
        time_purpose_array,
        daycode,
):
    """
    #TODO DOCSTRING
    It returns a unsorted time dataframe with trips. It includes number of trips, departure time,
    destination (purpose).
    
    Args:
        Rnd_week_trips ([type]): [description]
        km_array ([type]): [description]
        duration_array ([type]): [description]
        km_dur_prob_array ([type]): [description]
        km_sorted ([type]): [description]
        duration_sorted ([type]): [description]
        t ([type]): [description]
        time_purpose_array ([type]): [description]
        daycode ([type]): [description]

    Returns:
        [type]: [description]
    """
    wdc = np.zeros((Rnd_week_trips, 7))
    deltime = np.array([-1.0], dtype=np.float32)
    dists = distances_collection(
        Rnd_week_trips,
        km_array,
        duration_array,
        km_dur_prob_array,
        km_sorted,
        duration_sorted,
    )
    for trip in range(Rnd_week_trips):
        time_purp_tuple = time_purpose_iter(time_purpose_array, daycode, deltime)
        timestep = np.ceil(dists[1][trip] / 60 / t)
        timestep_ = np.array(timestep, dtype=np.float32)
        times = time_purp_tuple[0]
        purp = time_purp_tuple[1]
        K = timestep_ * t
        arrival = np.add(times, K)
        wdc[trip, 0] = dists[1][trip]  # trip_duration
        wdc[trip, 1] = dists[0][trip]  # distance
        wdc[trip, 2] = timestep  # timesteps
        wdc[trip, 3] = Rnd_week_trips  # trips
        wdc[trip, 4] = times  # departure
        wdc[trip, 5] = arrival  # arrival
        wdc[trip, 6] = purp  # purpose

        for i in range(timestep):
            deltime = np.append(
                deltime, time_purp_tuple[0] + i * t
            )  # adding to the lists the time and purpose of the current trip
        deltime = np.append(deltime, arrival)
    return wdc


@jit(nopython=True)
def create_tour_nb(unsortedtrips, previous_dest, lasttriparrival, t):
    """
    #TODO DOCSTRING
    Create tour tuple .

    Args:
        unsortedtrips ([type]): [description]
        previous_dest ([type]): [description]
        lasttriparrival ([type]): [description]
        t ([type]): [description]

    Returns:
        [type]: [description]
    """
    tour = np.zeros((unsortedtrips.shape[0], unsortedtrips.shape[1] + 4))
    tour[:, :-4] = unsortedtrips
    state = previous_dest
    prev_triptime = lasttriparrival
    for i in range(tour.shape[0]):
        timestep = np.ceil(tour[i, 0] / 60 / t)
        timestep_ = np.array(timestep, dtype=np.float32)
        arrival = tour[i, 4] + timestep_ * t
        tour[i, 2] = timestep  # 'timesteps'
        tour[i, 5] = arrival  # 'arrival'
        tour[i, 7] = state  # 'state'
        tour[i, 8] = prev_triptime  # 'last_arrival'
        tour[i, 9] = tour[i, 4] - prev_triptime  # 'duration'
        tour[i, 10] = tour[i, 6] == tour[i, 7]  # 'equal'
        state = tour[i, 6]
        prev_triptime = tour[i, 5]
    return tour, state


@jit(nopython=True)
def time_purpose_iter(array_tp, daycode, deltime):
    """
    #TODO DOCSTRING
    Return a random value for time purpose

    Args:
        array_tp ([type]): [description]
        daycode ([type]): [description]
        deltime ([type]): [description]

    Returns:
        [type]: [description]
    """
    D = array_tp[array_tp[:, 0] == daycode].copy()
    H = D[~nb_isin(D[:, 1], deltime)]
    H[:, 3] = H[:, 3] / H[:, 3].sum()
    tuple_prob = H[:, 3]
    rng_list = np.arange(tuple_prob.shape[0])
    rand = rand_choice_nb(rng_list, prob=tuple_prob)
    value = H[rand: rand + 1, 1:3]
    return value[0]


class Mobility:
    '''
    To create a mobility time series, after creating the Mobility class instance call the methods in the following order:

        - self.set_params(param)
        - self.set_stats(stat_ntrip, stat_dest, stat_km)
        - self.set_rules(rules_key, rules_file)
        - self.run()
        - self.save_profile(destination_folder)

    Args:
        config_folder (string): Folder name where the configuration files are hosted. Configuration files names are specified when setting stats and rules.
    '''

    def __init__(self, config_folder='config_files'):
        copy_to_user_data_dir()
        self.kind = "driving"
        self.config_dir = config_folder
        self.param_flag = False
        self.stats_flag = False
        self.rules_flag = False
        self.initial_state = None
        self.name_prefix = None
        self.refdate = None
        self.hours = None
        self.t = None
        self.user_defined = None
        self.df1 = None
        self.df2 = None
        self.df3 = None
        self.user_rules = None
        self.operators = None
        self.weeks = None
        self.numb_weeks = None
        self.sequence = None
        self.totalrows = None
        self.df2_t = None
        self.df2mod = None
        self.states = None
        self.uniquedays = None
        self.time_purp_array = None
        self.df4 = None
        self.km_array = None
        self.duration_array = None
        self.km_dur_prob_array = None
        self.km_sorted = None
        self.duration_sorted = None
        self.rules = None
        self.name = None
        self.Rnd_week_trips = None
        self.flag = None
        self.cause = None
        self.flag1 = None
        self.day = None
        self.weektype = None
        self.cut = None
        self.rules_ = None
        self.no_trip = None
        self.warningB = None
        self.causes = None
        self.rate = None
        self.flagiter = None
        self.tour = None
        self.start = None
        self.prev_dst = None
        self.repeats = None
        self.fixed = None
        self.copied = None
        self.same = None
        self.db = None
        self.timeseries = None
        self.idx = None
        self.mixed = None
        self.rp = None
        self.logdir = None
        self.logdf = None
        self.days = None
        self.prev_dest = None
        self.n_day = None
        self.total_days = None
        self.profile = None
        self.hr = None
        self.calc = None
        self.val = None
        self.final_time = None
        self.total_time = None
        self.description = None
        self.ratesteps = None
        self.counts = None
        self.countbefore = None
        self.rank = None
        self.WEEKS = None

    def __getattr__(self, item):
        check_for_new_function_name(item)
        # if the return value is not callable, we get TypeError:

    def set_params(self, name_prefix='', total_hours=168, time_step_in_hrs=0.5, category='user_defined', reference_date="01/01/2020", initial_state='home'):
        """
        Defines the attributes given to the class objects. 

        Args:
            name_prefix (str): It is an identifier that takes part of the file name. The user can freely choose it, e.g. 'worker'. Defaults to ''
            total_hours (int): Defaults 168 hrs (one week). Maximum value 8760.
            time_step_in_hrs (float): It is the time step in hours. Defaults to 0.5 . Options are 0.25, 0.5, 1. Recommennded values below 1.
            category (str): This is an identifier. A column in the time series will contain this string e.g. 'worker'. Defauts to 'user_defined'.
            reference_date (str, optional): With this date the time series will start. All consecutive time steps will follow this date, e.g. '01/01/2020'. Defaults to "01/01/2020".
        """
        self.name_prefix = name_prefix
        self.refdate = reference_date
        self.hours = total_hours
        self.t = time_step_in_hrs
        self.user_defined = category
        self.param_flag = True
        self.initial_state = initial_state

    def set_stats(self, stat_ntrip_path, stat_dest_path, stat_km_duration_path):
        """
        Name of files that contain the required probabilities to generate a mobility time series. They are three: DepartureDestinationTrip, DistanceDurationTrip, and TripsPerDay.
        
        The DepartureDestinationTrip file must contain a wide data format (https://en.wikipedia.org/wiki/Wide_and_narrow_data). 
        The first two columns are days and departure time in hours. The following columns represent the expected destinations where 'home' must be one of the destinations. 
        At least 2 destinations (columns) should be included.
        As emobpy takes into account weekend days as well, the column days must contain 'weekdays', 'sunday' and 'saturday'. 
        Suppose your data does not differentiate from these days. Repeat the probabilities in each of the three options.

        The DistanceDurationTrip file must contain a wide data format. The first column is distance in kilometers. 
        The following columns are the duration of the trip in minutes.
        The tool determines the distance of trips and the duration. Then calculates the average velocity. 
        Make sure that all possible combinations in your joint probabilities lead to consistent average velocities.

        The TripsPerDay file must contain a wide data format. The first column is 'trip' and represnts the number of trips per day. 
        The column weekday and weekend contain the probability distribution.
        Bear in mind that zero trip per day represent not trips at all, two trips means a trip to some destination and then a retun trip. 
        One trip is not possible, and considered a limitation of the tool, as the model needs to starts the following day from home again.
        Always set 1 trip with zero as probability value. Also the maximun number of trip per day is limited with the time steps selected. 
        If the time step is one hour, then having a distribution with 8 trips per day could be challenging for the model, and it might hang off.

        Args:
            stat_ntrip_path (str): e.g. 'DepartureDestinationTrip.csv'
            stat_dest_path (str): e.g. 'DistanceDurationTrip.csv'
            stat_km_duration_path (str): 'TripsPerDay.csv

        Raises:
            Exception: Raised, if .set_params(args) is not called before.
        """
        if self.param_flag:
            self.df1 = pd.read_csv(os.path.join(self.config_dir, stat_ntrip_path))
            self.df2 = (
                pd.read_csv(os.path.join(self.config_dir, stat_dest_path)).replace(0, np.nan)
                    .dropna(axis=1, how="all")
                    .replace(np.nan, 0)
            )
            self.df3 = pd.read_csv(os.path.join(self.config_dir, stat_km_duration_path))
            self.stats_flag = True
        else:
            logger.info('.set_stats(args) must be called after ".set_params(args)"')
            raise Exception('.set_stats(args) must be called after ".set_params(args)"')

    def set_rules(self, rule_key=None, rules_path="rules.yml"):
        """
        A set of rules can be defined and provided in a YAML file. Several customized rules can be allocated in only one file. 
        Hence, the rule_key points to the set of rules that wanted to be used in the current run. If no rule_key is provided, then the tool copy a rules dictionary with default values.
        
        If you want to know all the possibles rules to create your own rules file afterwards, type .initial_conf() and then .rules, e.g. Mobility.rules
        
        Args:
            rule_key (str, optional): e.g. 'user_defined'. Defaults to None.
            rules_path (str, optional): Path to rules file. Defaults to "rules.yml" in config_files folder.

        Raises:
            Exception: Raised if .setStats(args) is not called before. 
        """
        if self.stats_flag:
            if rule_key is None:
                self.user_rules = {}
                logger.info("set_rules: rules not provided, using default rules instead")
            else:
                with open(os.path.join(self.config_dir, rules_path)) as file:
                    rule_dict = yaml.load(file, Loader=yaml.SafeLoader)
                self.user_rules = rule_dict[rule_key]
            self._initial_conf()
            self.rules_flag = True
        else:
            logger.info('.set_rules(args) must be called after ".set_stats(args)"')
            raise Exception('.set_rules(args) must be called after ".set_stats(args)"')

    def _initial_conf(self):
        """
        Create initial configuration for the model.

        Raises:
            Exception: Raised if departure-purpose has different time intervals.
            Exception: Raised if time is not ranged between 0 and 23.99.
            Exception: Raised if original time interval can not be converted, because the quotient is not an integer.
        """
        self.operators = OPERATORS
        self.weeks = self.WEEKS or WEEKS

        self.numb_weeks = int(self.hours / 7 / 24) + 1
        self.sequence = days_week_sequence(self.refdate, self.weeks)
        self.totalrows = self.hours / self.t

        df2_t = self.df2.groupby("days")["time"].diff().dropna().unique()
        if len(df2_t) > 1:
            raise Exception(
                """stat_ntrip variable: departure-purpose table has different time intervals,
                make sure the time interval is unique (time interval is the difference between
                one row and the next one in the table)"""
            )
        self.df2_t = df2_t[0]

        if self.df2["time"].max() >= 24.0:
            raise Exception(
                """replace 24 by 0 in time column of departure time - trip purpose probabilities,
                time should range between 0 and 23.99"""
            )
        df2i = self.df2.copy()
        df2i.loc[df2i[df2i["time"] < 3.0].index, "time"] = (
                df2i[df2i["time"] < 3.0]["time"] + 24.0
        )  # this change enables to create day tours with returning time to "home"
        #  at the most just before 3 am the next day.

        factor = self.df2_t / self.t
        if not factor.is_integer():
            raise Exception(
                f"""Original time interval of departure-purpose table ({self.df2_t})
                can not be converted to the model time_step ({self.t}) because the quotient
                is not an integer ({factor})"""
            )
        time_step = self.t
        if factor > 1:
            df2i = expand_table(df2i, factor, time_step)

        df2ii = (
            df2i.set_index(["days", "time"])
                .stack()
                .reset_index()
                .rename(columns={"level_2": "purpose", 0: "value"})
        )
        self.df2mod = df2ii.copy()

        self.states = self.df2mod["purpose"].unique().tolist()
        self.uniquedays = self.df2mod["days"].unique().tolist()

        npdf2 = self.df2mod.copy()
        npdf2.loc[:, "days"] = npdf2["days"].apply(self.uniquedays.index)
        npdf2.loc[:, "purpose"] = npdf2["purpose"].apply(self.states.index)
        self.time_purp_array = npdf2.to_numpy(dtype=np.float32)

        self.df4 = (
            self.df3.set_index("km")
                .stack()
                .reset_index(name="value")
                .rename(columns={"level_1": "min"})
        )
        self.km_array = self.df4["km"].values.astype(np.float64)
        self.duration_array = self.df4["min"].astype(np.float64).values
        self.km_dur_prob_array = self.df4["value"].values
        self.km_sorted = np.sort(np.unique(self.km_array))
        self.duration_sorted = np.sort(np.unique(self.duration_array))

        self.rules = RULE
        for week, dicts in self.user_rules.items():
            for opt, stdict in dicts.items():
                if isinstance(stdict, list):
                    self.rules[week][opt] = stdict
                elif isinstance(stdict, int):
                    self.rules[week][opt] = stdict
                else:
                    for key, value in stdict.items():
                        self.rules[week][opt][key] = value
        self.name = (
                self.name_prefix
                + "_"
                + "W"
                + str(self.numb_weeks)
                + "_"
                + uuid.uuid4().hex[0:5]
        )

    def _group_trips_week(self):
        """
        This function samples a number of trip for a day (self.Rnd_week_trips). The purpose is to create a day tour.
        This value is obtained based on mobility statistics.
        """
        T = self.df1.copy()
        idx_drop = T[T["trip"].isin(self.cut)].index.tolist()
        T.drop(idx_drop, axis=0, inplace=True)
        T.loc[:, self.weektype] = (
                T[self.weektype] / T[self.weektype].sum()
        )  # once the trip quantity in "cut" is removed from the probability distribution,
        # normalization is implemented for the remaining trip quantities to add up 100% again.
        trips_list = T["trip"].values
        trips_prob = T[self.weektype].values
        self.Rnd_week_trips = rand_choice_nb(
            trips_list, prob=trips_prob
        )  # Numpy function, that takes into account a list of options (trips numb eg:0,1,or 3 ...)
        # and their corresponding probabilities to chose one option (eg: 2 trips for a day tour)

    def _meet_all_conditions(self):
        """
        The rules here are tested to see if the tour created comply with the set of rules.
        """
        self.flag = False
        self.cause = ""
        for condition, op in self.operators.items():
            self.flag1 = False
            for state in self.states:
                if self.rules_[self.weektype][condition][
                    state
                ]:  # for any condition in Rules when False, it continues to the next state
                    if condition == "last_trip_to":
                        if self.tour["purpose"].iloc[-1]:
                            if cmp(self.tour["purpose"].iloc[-1], op, state):
                                pass
                            else:
                                self.flag1 = True
                                self.cause = "last_trip_to " + state
                                break
                    elif condition == "first_trip_to":
                        if self.tour["purpose"].iloc[0]:
                            if cmp(self.tour["purpose"].iloc[0], op, state):
                                pass
                            else:
                                self.flag1 = True
                                self.cause = "first_trip_to " + state
                                break
                    elif condition == "not_last_trip_to":
                        if self.tour["purpose"].iloc[-1]:
                            if cmp(self.tour["purpose"].iloc[-1], op, state):
                                pass
                            else:
                                self.flag1 = True
                                self.cause = "not_last_trip_to " + state
                                break
                    elif (
                            condition == "min_state_duration"
                            or condition == "max_state_duration"
                    ):
                        if not self.tour[self.tour["state"] == state]["duration"].empty:
                            dur_list = self.tour[self.tour["state"] == state][
                                "duration"
                            ].values.tolist()
                            for dur in dur_list:
                                if cmp(
                                        dur,
                                        op,
                                        self.rules_[self.weektype][condition][state],
                                ):
                                    pass
                                else:
                                    self.flag1 = True
                                    self.cause = "min_or_max_state_duration " + state
                                    break
                        else:
                            if self.rules_[self.weektype]["at_least_one_trip"][state]:
                                self.flag1 = True
                                self.cause = "min_or_max_state_duration " + state
                                break
                            else:
                                pass
                    elif condition == "equal_state_and_destination":
                        if (
                                not self.tour[self.tour["state"] == state]["purpose"]
                                        .str.contains(state, case=True, regex=True)
                                        .any()
                        ):
                            pass
                        else:
                            self.flag1 = True
                            self.cause = "equal_state_and_destination " + state
                            break
                    elif (
                            condition == "overall_min_time_at"
                            or condition == "overall_max_time_at"
                    ):
                        if self.tour[self.tour["state"] == state].empty:
                            if self.rules_[self.weektype]["at_least_one_trip"][state]:
                                self.flag1 = True
                                self.cause = "overall_min_or_max_time_at " + state
                                break
                            else:
                                pass
                        elif (
                                self.tour[self.tour["state"] == state]["duration"].sum()
                                != 0
                        ):
                            if cmp(
                                    self.tour[self.tour["state"] == state][
                                        "duration"
                                    ].sum(),
                                    op,
                                    self.rules_[self.weektype][condition][state],
                            ):
                                pass
                            else:
                                self.flag1 = True
                                self.cause = "overall_min_or_max_time_at " + state
                                break
                        else:
                            self.flag1 = True
                            self.cause = "overall_min_or_max_time_at " + state
                            break
            if self.flag1:
                self.flag = True
                break

    def _select_tour(self):
        """
        This function samples the number of trips, the departure time, destination, distance for each trip, duration and tests the rules to generate a successful day tour.
        The tour is created based on the three probability distributions provided at the moment of calling the method .set_stats
        """
        self.day = self.weeks[self.n_day]["day_code"]
        self.weektype = self.weeks[self.n_day]["week"]
        self.cut = self.rules[self.weektype]["n_trip_out"][:]
        self._group_trips_week()
        if self.Rnd_week_trips != 0:
            self.rules_ = copy.deepcopy(self.rules)
            self.no_trip = False
            self.warningB = -1
            self.causes = []
            self.rate = {}
            self.flagiter = False
            while True:
                self.warningB += 1
                unsortedtrips = creation_unsorted_trips_nb(
                    self.Rnd_week_trips,
                    self.km_array,
                    self.duration_array,
                    self.km_dur_prob_array,
                    self.km_sorted,
                    self.duration_sorted,
                    np.array(self.t, dtype=np.float32),
                    self.time_purp_array,
                    self.uniquedays.index(self.day),
                )

                unsortedtrips = np.array(sorted(unsortedtrips, key=lambda x: x[5]))
                prev_dest_code = self.states.index(self.prev_dest)
                tour_array, state_code = create_tour_nb(
                    unsortedtrips,
                    prev_dest_code,
                    self.start,
                    np.array(self.t, dtype=np.float32),
                )
                dftour = tour_array.astype(object)
                dftour[:, 6] = np.array([self.states[int(x)] for x in dftour[:, 6]])
                dftour[:, 7] = np.array([self.states[int(x)] for x in dftour[:, 7]])
                state = self.states[int(state_code)]
                self.tour = pd.DataFrame(
                    dftour,
                    columns=[
                        "trip_duration",
                        "distance",
                        "timesteps",
                        "trips",
                        "departure",
                        "arrival",
                        "purpose",
                        "state",
                        "last_arrival",
                        "duration",
                        "equal",
                    ],
                )

                # self.tour[["departure","arrival"]] = (self.tour[["departure","arrival"]].astype(float))
                # self.tour["departure"] = (self.tour["departure"] / self.t).round().mul(self.t)
                # self.tour["arrival"] = (self.tour["arrival"] / self.t).round().mul(self.t)
                

                self._meet_all_conditions()
                # self._logging_meetcond()
                if self.flag:
                    continue
                if self.flagiter:
                    logger.debug(f"  Tour done: Day {str(self.days)}")
                break

            self.start = self.tour["arrival"].iloc[-1] - 24.0
            self.prev_dst = copy.deepcopy(state)
        else:
            self.no_trip = True
            self.tour = False
            self.start -= 24
            self.prev_dst = copy.deepcopy(self.prev_dest)
            logger.debug(f"No trip day: {str(self.days)}")

    def _fill_rows(self):
        """
        This function convert the tours to a time series data format and adding 'driving' between locations

        .. code-block:: Python

                departure   state   purpose     distance    trip_duration
            1   8.0         home    workplce    10          20
            2   12.0        workplce errands    2           5
            3   13.0        errands workplce    3           7
            4   18.0        workplce Home       9           30

        It is converted to this new data format:

        .. code-block:: Python

                hr   state  distance    trip_duration
            0   7.5  home
            1   8.0  driving
            2   8.5  workplace
            3   9.0  workplace
            4   9.5  workplace
            ...

        """
        self.repeats = [
            "hr",
            "state",
            "weekday",
            "category",
            "distance",
            "trip_duration",
        ]
        self.fixed = []
        self.copied = ["departure", "last_arrival", "purpose", "duration"]
        self.calc = ["dayhrs"]
        self.same = []
        self.db = self.profile.copy()
        self.db.loc[:, "dayhrs"] = self.db["hr"] % 24
        self.timeseries = pd.DataFrame(columns=self.db.columns)
        self.timeseries.loc[:, "hh"] = np.arange(0, self.hours, self.t)

        # Start New version, which works for 1s-based profiles:
        temp_timeseries = [round(num * 3600) for num in self.timeseries["hh"]]
        temp_db = [round(num * 3600) for num in self.db["hr"]]
        temp_intersection_list = list(set(temp_timeseries).intersection(temp_db))

        self.idx = []
        for i in temp_intersection_list:
            self.idx.append(temp_timeseries.index(i))
        self.idx = np.sort(self.idx).tolist() # values might be unsorted -> sort
        # End new version

        # # ───--------- Quantize & align db‑timestamps ───--------------
        # # 1) Snap every profile timestamp to the nearest grid step (self.t),
        # #    then clamp into [0, hours–t] so you never fall off the ends.
        # self.db["hr"] = (
        #     (self.db["hr"] / self.t)
        #     .round()
        #     .mul(self.t)
        #     .clip(lower=0, upper=self.hours - self.t)
        # )
        # self.db = self.db.drop_duplicates(subset="hr", keep="first")

        # # 2) Build seconds‐based lists (now exact multiples of t):
        # secs_ts = [int(h * 3600) for h in self.timeseries["hh"]]
        # secs_db = [int(h * 3600) for h in self.db   ["hr"]]

        # # 3) Find which seconds coincide:
        # intersect = set(secs_ts).intersection(secs_db)

        # # 4) Build the target‐index list:
        # self.idx = sorted(secs_ts.index(s) for s in intersect)

        # # 5) For each “repeat/copy” column, extract exactly matching rows:
        # for r in self.repeats + self.fixed + self.copied:
        #     mask = [sec in intersect for sec in secs_db]
        #     vals = self.db.loc[mask, r].tolist()
        #     self.timeseries.loc[self.idx, r] = vals
        # # -------------------------------------------------------------------------


        self.mixed = self.repeats + self.fixed + self.copied
        for r in self.mixed:
            self.val = self.db[r].values.tolist()
            self.timeseries.loc[self.idx, r] = self.val
        self.timeseries.loc[self.totalrows - 1, "state"] = self.db["state"].iloc[-1]
        self.timeseries.loc[self.totalrows - 1, "hr"] = self.timeseries["hh"][
            self.totalrows - 1
            ]
        self.rp = self.timeseries[::-1].reset_index(drop=True)
        self.rp.loc[:, self.repeats] = self.rp[self.repeats].fillna(method="ffill")
        self.rp.loc[:, self.fixed] = self.rp[self.fixed].fillna(0)
        self.timeseries = self.rp[::-1].reset_index(drop=True)
        for sm in self.same:
            self.timeseries.loc[:, sm] = self.db[sm].values.tolist()[0]
        for cal in self.calc:
            self.timeseries.loc[:, cal] = self.timeseries["hh"].apply(lambda x: x % 24)

        self.timeseries = _add_column_datetime(
            self.timeseries, self.totalrows, self.refdate, self.t
        )
        self.timeseries.loc[:, "count"] = self.timeseries.groupby(["hr", "state"])[
            "distance"
        ].transform("count")
        self.timeseries.loc[:, "distance"] = (
                self.timeseries.loc[:, "distance"] / self.timeseries.loc[:, "count"]
        )
        self.timeseries = self.timeseries[["hh","state", "distance"]]


    def _clean(self, keep_attr=None):
        """
        Deletes all attributes that were not declared in keep_attr. 

        Args:
            keep_attr (list, optional): List of strings containing attributes to be kept. Default is a list of attributes.
        """
        to_rem = list(self.__dict__.keys())[:]
        if keep_attr is None:
            keep_attr = [
                "kind",
                "name_prefix",
                "refdate",
                "hours",
                "t",
                "user_defined",
                "df1",
                "df2",
                "df3",
                "user_rules",
                "states",
                "numb_weeks",
                "totalrows",
                "name",
                "profile",
                "timeseries",
            ]
        for r in keep_attr:
            if r in to_rem:
                to_rem.remove(r)
        for attr in to_rem:
            self.__dict__.pop(attr, None)
        del to_rem

    def run(self):
        """
        This function returns a driving profile. No input possible. Returns nothing. 
        Once it finishes the following attributes can be called:

        - kind
        - name_prefix
        - refdate
        - hours
        - t
        - user_defined
        - df1
        - df2
        - df3
        - user_rules
        - states
        - numb_weeks
        - totalrows
        - name
        - profile
        - timeseries
        """
        if self.rules_flag:
            pass
        else:
            logger.info('.run() must be called after ".setRules(args)"')
            raise Exception('.run() must be called after ".setRules(args)"')
        initial_time = time.time()
        logger.info(f"New profile running: {self.name}")
        logger.debug("###################################################")
        logger.debug("===================================================")
        logger.debug(f"New profile running: {self.name}")
        logger.debug("===================================================")
        logger.debug("###################################################")

        self.logdir = os.path.join("log", self.name)
        self.logdf = pd.DataFrame()
        self.start = -3
        self.flag = True
        self.days = -1
        self.prev_dest = self.initial_state  # it is the first state at the beginning of the profile

        self.total_days = int(self.hours / 24)
        self.profile = pd.DataFrame()
        endflag = False
        lastflag = False
        for _ in range(self.numb_weeks):
            for n_day in self.sequence:
                self.n_day = n_day
                self.days += 1
                # if self.days <= self.total_days:
                #     mobility_progress_bar(self.days, self.total_days)
                self._select_tour()
                self.prev_dest = copy.deepcopy(self.prev_dst)
                if not self.no_trip:
                    for _, row in self.tour.iterrows():
                        self.hr = row["departure"] + self.days * 24.0 - self.t
                        if self.hr == self.hours - self.t:
                            self.profile.loc[self.hr, "hr"] = self.hr
                            self.profile.loc[self.hr, "state"] = row["state"]
                            self.profile.loc[self.hr, "departure"] = row["departure"]
                            self.profile.loc[self.hr, "arrival"] = row["arrival"]
                            self.profile.loc[self.hr, "last_arrival"] = row[
                                "last_arrival"
                            ]
                            self.profile.loc[self.hr, "purpose"] = row["purpose"]
                            self.profile.loc[self.hr, "duration"] = row["duration"]
                            self.profile.loc[self.hr, "weekday"] = self.weeks[
                                self.n_day
                            ]["day"]
                            self.profile.loc[self.hr, "category"] = self.user_defined
                            self.profile.loc[self.hr, "distance"] = 0
                            self.profile.loc[self.hr, "trip_duration"] = 0
                            endflag = True
                            break
                        elif self.hr > self.hours - self.t:
                            endflag = True
                            break
                        elif self.hr < self.hours - self.t:
                            self.profile.loc[self.hr, "hr"] = self.hr
                            self.profile.loc[self.hr, "state"] = row["state"]
                            self.profile.loc[self.hr, "departure"] = row["departure"]
                            self.profile.loc[self.hr, "arrival"] = row["arrival"]
                            self.profile.loc[self.hr, "last_arrival"] = row[
                                "last_arrival"
                            ]
                            self.profile.loc[self.hr, "purpose"] = row["purpose"]
                            self.profile.loc[self.hr, "duration"] = row["duration"]
                            self.profile.loc[self.hr, "weekday"] = self.weeks[
                                self.n_day
                            ]["day"]
                            self.profile.loc[self.hr, "category"] = self.user_defined
                            self.profile.loc[self.hr, "distance"] = 0
                            self.profile.loc[self.hr, "trip_duration"] = 0
                            # add driving in next row
                            self.profile.loc[
                                self.hr + row["timesteps"] * self.t, "hr"
                            ] = (self.hr + row["timesteps"] * self.t)
                            self.profile.loc[
                                self.hr + row["timesteps"] * self.t, "state"
                            ] = "driving"
                            self.profile.loc[
                                self.hr + row["timesteps"] * self.t, "distance"
                            ] = row["distance"]
                            self.profile.loc[
                                self.hr + row["timesteps"] * self.t, "trip_duration"
                            ] = row["trip_duration"]
                if endflag:
                    lastflag = True
                    break
            if lastflag:
                break
        print("")
        if not self.profile.empty:
            if self.profile["hr"].iloc[-1] < self.hours - self.t:
                self.profile.loc[self.hours - self.t, "hr"] = self.hours - self.t
                self.profile.loc[self.hours - self.t, "state"] = self.prev_dest
                self.profile.loc[self.hours - self.t, "distance"] = 0
                self.profile.loc[self.hours - self.t, "trip_duration"] = 0
            elif self.profile["state"].iloc[-1] == "driving":
                self.profile.loc[self.hours - self.t, "hr"] = self.hours - self.t
                self.profile.loc[self.hours - self.t, "state"] = self.profile[
                    "state"
                ].iloc[-2]
                self.profile.loc[self.hours - self.t, "distance"] = 0
                self.profile.loc[self.hours - self.t, "trip_duration"] = 0
            self._fill_rows()
            self._clean()
        else:
            logger.info("""Empty profile. ...   This occurs either because a high probability of zero trips per day or/and selected few hours for the time series (couple of days only). Make sure you change seed value from seed.txt file or add a random int as seed arg to set_seed(seed) in your script to generate different seudo-random values.""")
            raise Exception("""Empty profile. ...   This occurs either because a high probability of zero trips per day or/and selected few hours for the time series (couple of days only). Make sure you change seed value from seed.txt file or add a random int as seed arg to set_seed(seed) in your script to generate different seudo-random values.""")
        final_time = time.time()
        self.total_time = round((final_time - initial_time) / 60, 2)
        logger.info(f"Profile done: {self.name}")
        logger.info(f"Elapsed time (min): {str(self.total_time)}")

    def save_profile(self, folder, description=" "):
        """
        Saves a profile from the current object as a pickle file. 

        Args:
            folder (str): Where the files will be stored. Folder is created in case it does not exist.
            description (str, optional): Profile description. Defaults to " ".
        """
        self.description = description
        info = self.__dict__
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, self.name + ".pickle")
        with gzip.open(filepath, "wb") as file:
            pickle.dump(info, file)
        del info
        logger.debug(f"Profile saved: {filepath}")
        logger.info(" ")
        logger.info("See Log files")
        for handler in logger.handlers:
            if handler.__class__.__name__ == 'FileHandler':
                logger.info(handler.baseFilename)
        try:
            display_all()
        except:
            pass


    def _logging_meetcond(self):
        """
        #TODO DOCSTRING: Find out what is happening here
        """
        if self.cause != "":
            self.causes.append(self.cause)
        if self.flag:
            self.ratesteps = 10
            if self.warningB % self.ratesteps == 0:
                self.counts = Counter(self.causes)
                if self.warningB == 0:
                    pass
                elif self.warningB == self.ratesteps:
                    self.countbefore = self.counts
                else:
                    for k, v in self.counts.items():
                        if k in list(self.countbefore.keys()):
                            self.rate[k] = (v - self.countbefore[k]) / self.ratesteps
                        else:
                            self.rate[k] = v / self.ratesteps
                    self.countbefore = self.counts
                    self.rate["days"] = self.days
                    self.rate["trips"] = self.Rnd_week_trips
                    self.rate["iter"] = self.warningB
                    os.makedirs(self.logdir, exist_ok=True)
                    self.logdf = pd.concat(
                        [self.logdf, pd.DataFrame.from_records([self.rate])], sort=False, ignore_index=True
                    )
                    sortedcol = ["days", "trips", "iter"]
                    cols = sortedcol + [
                        col for col in self.logdf if col not in sortedcol
                    ]
                    self.logdf = self.logdf[cols]
                    self.logdf.to_csv(os.path.join(self.logdir, "log.csv"), index=False)
            if (
                    self.warningB >= 10 * self.ratesteps
                    and self.warningB % 200 == 10 * self.ratesteps
            ):
                if not self.flagiter:
                    self.flagiter = True
                logger.debug(f"    Day {str(self.days)} 'select_tour' method in loop Nr. {self.warningB}. See log file {self.name}")
                self.rank = self.logdf[
                                self.logdf["days"] == self.logdf["days"].iloc[-1]
                                ][-3:][
                    [
                        x
                        for x in self.logdf.columns
                        if x not in ["days", "trips", "iter"]
                    ]
                ].mean()
                log_list = ["       " + x + " uncompliance rate (last 30 iter)" for x in self.rank.nlargest(2).round(2).to_string().split("\n")]
                for str_log in log_list:
                    logger.debug(str_log)
                if self.warningB > 2000:
                    df_strings = self.tour.to_string().replace("\n", "\n\t")
                    logger.debug(f"    Unsuccessful tour sample:\n\n\t {df_strings}\n")
