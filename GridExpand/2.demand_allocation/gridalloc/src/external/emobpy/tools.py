"""
This module contains a tool functions used to create a new project.
"""

from multiprocessing import Lock, Process, Queue, Manager, cpu_count
from numba import jit
import pandas as pd
import numpy as np
import operator
import random
import sys
import os
import time
import datetime
import json
from src.external.emobpy.constants import (TIME_FREQ, DEFAULT_DATA_DIR, USER_PATH)
from src.external.emobpy.messages import (MSG_CONF, MSG_ARGS, MSG_TEXT)
from src.external.emobpy.logger import get_logger

logger = get_logger(__name__)


def _add_column_datetime(df, totalrows, reference_date, t):
    """
    Useful to convert the time series from hours index to datetime index.

    Args:
        df (pd.DataFrame): Table on which datetime column should be added.
        totalrows (int): Number of rows on which datetime column should be added.
        reference_date (str): Starting date for adding. E.g. '01/01/2020'.
        t (float): Float frequency, will be changed to string.

    Returns:
        pd.DataFrame: Table with added datetime column.
    """
    freq = TIME_FREQ[t]['f']
    start_date = pd.to_datetime(reference_date)
    drange = pd.date_range(start_date, periods=totalrows, freq=freq)
    df = pd.DataFrame(df.values, columns=df.columns, index=drange)
    df = df.rename_axis("date").copy()
    return df

def represents_int(s):
    """
    Check if argument is an int value.

    Args:
        s (any type): Value to check.

    Returns:
        bool: True if the argument is an int value.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


@jit(nopython=True)
def is_between(t, time_range):
    """
    Checks if given value is between given time_range.

    Args:
        t (float): Value to check. 
        time_range (list): Time range list. E.g. [1,100].

    Returns:
        bool: Value if t is between time_range.
    """
    if time_range[1] < time_range[0]:
        return t >= time_range[0] or t <= time_range[1]
    return time_range[0] <= t <= time_range[1]

def cmp(arg1, string_operator, arg2):
    """
    Perform comparison operation according to operator module. This function is used on meet_all_conditions().

    Args:
        arg1 (object): First argument.
        string_operator ('<', '<=', '==', '!=', '>=', '>'): String operator which can be used.
        arg2 (object): Second Argument.

    Returns:
        bool: Result of comparison.
    """
    ops = {
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        ">": operator.gt,
    }
    operation = ops.get(string_operator)
    return operation(arg1, arg2)

def mobility_progress_bar(current, total):
    """
    Prints actual progress in format: "Progress: 80% [8 / 10] days".

    Args:
        current (int): Current day.
        total (int): Total number of days.
    """

    progress_message = "Progress: %d%% [%d / %d] days" % (
        current / total * 100,
        current,
        total,
    )
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()

def consumption_progress_bar(current, total):
    """
    Prints message about consumption progress.

    Args:
        current (int): Current index.
        total (int): Total number of loops.
        width (int, optional): Not used. Defaults to 80.
    """
    progress_message = "Progress: %d%% [%d / %d] trips" % (
        current / total * 100,
        current,
        total,
    )
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()
    
def wget_progress_bar(*args):
    """
    Prints actual progress in format: "Downloading: 80% [8 / 10] kilobyte"
    Args:
        current (int): Current download.
        total (int): Total number of downloads.
    """
    current = args[0]
    total = args[1]
    progress_message = "Downloading: %d%% [%d / %d] kilobyte" % (
        current / total * 100,
        current / 1024,
        total / 1024,
    )
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()

class Unit:
    """
    This class represents a unit. In this way a value and a unit can be saved and with class methods the unit
    can easily be changed. The default value used by the programme is always the 1 value in unit_scale.
    """

    def __init__(self, val, unit, description = "No description given."):
        self.val = val
        self.unit = unit
        self.description = description

    def convert_to(self, new_unit, msg=False):

        def change_unit(self, new_unit, msg):
            unit_scale = {'time': {'h': 1 / 3600, 'min': 1 / 60, 's': 1},
                          'velocity': {'km/h': 1, 'km/min': 1 / 60, 'km/s': 1 / 3600, 'm/h': 1000, 'm/min': 1000 / 60,
                                       'm/s': 1000 / 3600},
                          'energy': {'kWh': 1, 'Wh': 1000},
                          'power': {'kW': 1, 'W': 1000},
                          'voltage': {'V': 1, 'kW': 1 / 1000},
                          'distance': {'km': 1/1000, 'm': 1, 'cm': 100},
                          'weight': {'kg': 1, 'g': 1000, 't': 1 / 1000},
                          'volume': {'m3': 1},
                          'area': {'m2': 1}
                          }

            # Get correct scale for unit
            found = False
            for key, value in unit_scale.items():
                if self.unit in value:
                    found = True
                    scale = value
                    break
            if not found:
                return

            # Check if new_unit could not be loaded in scale
            if new_unit not in scale:
                if msg:
                    logger.info('The given unit is not known and can not be converted.')
                return

            conversion_value = scale[self.unit]
            # Change value and unit
            for key, value in scale.items():
                scale[key] = value / conversion_value

            if msg:
                logger.info(f'Unit of value changed: {self.val} {self.unit} -> {self.val * scale[new_unit]} {new_unit}')
            self.val *= scale[new_unit]
            self.unit = new_unit

            return self

        # Check if value is None
        if not self.val:
            if msg:
                logger.info('Unit could not be converted because value is None.')
            return


        change_unit(self, new_unit=new_unit, msg=msg)

        return self

    def convert_to_default_value(self):
        self.convert_to('s')
        self.convert_to('km/h')
        self.convert_to('kWh')
        self.convert_to('kW')
        self.convert_to('V')
        self.convert_to('m')
        self.convert_to('kg')
        self.convert_to('m3')
        self.convert_to('m2')

        return self

    def info(self):
        logger.info(f'val: {self.val}, unit: {self.unit}, description: {self.description}')


def parallelize(function=None, inputdict: dict = None, nr_workers=1, **kargs):
    """
    Parallelize function to run program faster.
    The queue contains tuples of keys and objects, the function must be consistent when getting data from queue.

    Args:
        function (function, optional): Function that is to be parallelized. Defaults to None.
        inputdict (dict, optional): Contains numbered keys and as value any object. Defaults to None.
        nr_workers (int, optional): Number of workers, so their tasks can run parallel. Defaults to 1.

    Returns:
        dict: Dictionary the given functions creates.
    """
    total_cpu = cpu_count()
    logger.debug(f"Workers: {nr_workers} of {total_cpu}")
    if nr_workers > total_cpu:
        nr_workers = total_cpu
        logger.debug(f"Workers: {nr_workers}")
    with Manager() as manager:
        dc = manager.dict()
        queue = Queue()
        for key, item in inputdict.items():
            queue.put((key, item))
        queue_lock = Lock()
        processes = {}
        for i in range(nr_workers):
            if kargs:
                processes[i] = Process(target=parallel_func,
                                       args=(
                                           dc,
                                           queue,
                                           queue_lock,
                                           function,
                                           kargs,
                                       ))
            else:
                processes[i] = Process(target=parallel_func,
                                       args=(
                                           dc,
                                           queue,
                                           queue_lock,
                                           function,
                                       ))
            processes[i].start()
        for i in range(nr_workers):
            processes[i].join()
        outputdict = dict(dc)
    return outputdict


def parallel_func(dc, queue=None, queue_lock=None, function=None, kargs={}):
    """
    #TODO DOCSTRING

    Args:
        dc ([type]): [description]
        queue ([type], optional): [description]. Defaults to None.
        queue_lock ([type], optional): [description]. Defaults to None.
        function ([type], optional): [description]. Defaults to None.
        kargs (dict, optional): [description]. Defaults to {}.

    Returns:
        [type]: [description]
    """

    while True:
        queue_lock.acquire()
        if queue.empty():
            queue_lock.release()
            return None
        key, item = queue.get()
        queue_lock.release()
        obj = function(**item, **kargs)
        dc[key] = obj


def set_seed(seed=None, dir='config_files'):
    """
    Sets seed at the beginning of any python script or jupyter notebook. That allows to repeat the same calculations
    with exactly same results, without any random noise.
    """
    @jit(nopython=True)
    def set_seed_with_numba(seed):
        np.random.seed(seed)
        random.seed(seed)
    if seed is None:
        try:
            with open(os.path.join(dir,'seed.txt')) as f:
                seed = int(f.read().replace('\n', ''))
        except ValueError:
            logger.info('As seed.txt does not contain a number, a random value has been written. This number is useful when you want to replicate the same results in another OS or PC.')
            seed = int(time.time())
            with open(os.path.join(dir,'seed.txt'), 'w') as f:
                f.write(str(seed))
        set_seed_with_numba(seed)
    else:
        if os.path.isfile(os.path.join(dir,'seed.txt')):
            os.remove(os.path.join(dir,'seed.txt'))
            logger.debug(f'seed.txt has been deleted to replace it with the given seed {seed}.')
        with open(os.path.join(dir,'seed.txt'), 'w') as f:
            f.write(str(seed))
            logger.debug(f'Seed {seed} has been written to seed.txt.')
        set_seed_with_numba(seed)


def check_for_new_function_name(attribute_error_name):
    """
    In an earlier update function names have been changed from camelCase to snake_case. To prevent users confusing this
    function raises a specific AttributeError of the user trys to access to old function name, which does not exist
    anymore.
    """
    new_names = {
        'ChooseChargingPoint': '_choose_charging_point',
        'ChooseChargingPointFast': '_choose_charging_point_fast',
        'drawing_soc': '_drawing_soc',
        'fill_rows': '_fill_rows',
        'initial_conf': '_initial_conf',
        'loadSettingDriving': '_load_setting_driving',
        'save_profile': 'save_profile',
        'setBatteryRules': '_set_battery_rules',
        'setScenario': 'set_scenario',
        'setVehicleFeature': '_set_vehicle_feature',
        'soc': '_soc',
        'testing_soc': '_testing_soc',

        'A2BatPoint': '_A2BatPoint',
        'balanced': '_balanced',
        'changeBatteryCapacity': 'x',
        'check_success': '_check_success',
        'immediate': '_immediate',
        'loadScenario': 'load_scenario',
        'setSubScenario': 'set_sub_scenario',

        'load_specs': '_load_specs',

        'cop_and_target_temp': '_cop_and_target_temp',
        'ev_par_test': '_ev_par_test',
        'loadSettingMobility': 'load_setting_mobility',

        'select_driving_cycle_index': '_select_driving_cycle_index',
        'get_index_speed': '_get_index_speed',

        'check': '_check',
        'layers_name': '_layers_name',
        'makearrays': '_makearrays',
        'zones_name': '_zones_name',

        'get_codes': '_get_codes',
        'get_efficiency': '_get_efficiency',
        'load_file': '_load_file',

        'frontal_area': '_frontal_area',
        'PMR': '_pmr',

        'airDensityFromIdealGasLaw': 'air_density_from_ideal_gas_law',
        'calcDewPoint': 'calc_dew_point',
        'calcDryAirPartialPressure': 'calc_dry_air_partial_pressure',
        'calcRelHumidity': 'calc_rel_humidity',
        'calcVaporPressure': 'calc_vapor_pressure',
        'humidairDensity': 'humidair_density',

        'loadfilesBatch': 'loadfiles_batch',
        'clean': '_clean',
        'group_trips_week': '_group_trips_week',
        'logging_meetcond': '_logging_meetcond',
        'MeetAllConditions': '_meet_all_conditions',
        'select_tour': '_select_tour',
        'setParams': 'set_params',
        'setStats': 'set_stats',
        'setRules': 'set_rules',
    }
    if attribute_error_name in new_names.keys():
        raise AttributeError(
            f'{attribute_error_name} does not exist. Note: We changed the attribute names from camelCase '
            f'to snake_case. \n The new attribute name for {attribute_error_name} is {new_names[attribute_error_name]}.')
    else:
        raise AttributeError(
            f'{attribute_error_name} does not exist. Note: We changed the attribute names from camelCase '
            f'to snake_case. \n You may have to adapt your attributes.')

def create_json_messages(file_name, directory, **kwargs):
    """
    Creates a json file with the given name and directory. The file is created in the given directory.
    The file contains the given kwargs as a json file.

    Args:
        file_name (str): Name of the file.
        directory (str): Directory where the file should be created.
        **kwargs: Keyword arguments that should be written to the file.
    """
    with open(os.path.join(directory, file_name + '.json'), 'w') as f:
        json.dump(kwargs, f, indent=4)
        
def check_message_file(file_name, directory):
    """
    Checks if a json file with the given name and directory exists.

    Args:
        file_name (str): Name of the file.
        directory (str): Directory where the file should be created.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    file_exists = os.path.isfile(os.path.join(directory, file_name + '.json'))
    return file_exists

def edit_json_messages(msg_id=0, directory=None, **kwargs):
    """
    Creates a json file with the given name and directory. The file is created in the given directory.
    The file contains the given kwargs as a json file.

    Args:
        file_name (str): Name of the file.
        directory (str): Directory where the file should be created.
        **kwargs: Keyword arguments that should be written to the file.
    """
    user_dir = directory or USER_PATH or DEFAULT_DATA_DIR
    file_name = MSG_CONF[msg_id]['json_file_name']
    if check_message_file(file_name, user_dir):
        with open(os.path.join(user_dir, file_name + '.json'), 'r') as f:
            old_kwargs = json.load(f)
        old_kwargs.update(kwargs)
        with open(os.path.join(user_dir, file_name + '.json'), 'w') as f:
            json.dump(old_kwargs, f, indent=4)
    else:
        create_json_messages(file_name, user_dir, **MSG_ARGS[msg_id])
        edit_json_messages(msg_id, user_dir, **kwargs)
        
def get_json_msg_args(msg_id=0, directory=None):
    """
    Returns the json file with the given name and directory.

    Args:
        file_name (str): Name of the file.
        directory (str): Directory where the file should be created.

    Returns:
        dict: The json file as a dictionary.
    """
    user_dir = directory or USER_PATH or DEFAULT_DATA_DIR
    file_name = MSG_CONF[msg_id]['json_file_name']
    if check_message_file(file_name, user_dir):
        with open(os.path.join(user_dir, file_name + '.json'), 'r') as f:
            return json.load(f)
    else:
        return MSG_ARGS[msg_id]


def msg_disable(msg_id=0):
    """
    Disables message msg_id.
    """
    if msg_id in MSG_CONF.keys():
        edit_json_messages(msg_id, **{'enabled': False})
        logger.info("Message disabled!")
    else:
        raise ValueError(f'Message {msg_id} does not exist.')


def check_msg_arg_conditions(msg_id):
    """
    Check if the message can be displayed based on the conditions in the MSG_CONF.
    """
    # I use 'try' beacuse I do not want to get unexpected behaviour that may lock the program.
    # This is an non relevant function.
    try:
        status = get_json_msg_args(msg_id)
        if status['enabled']:
            actual_day_count = status['day_count']
            actual_total_count = status['total_count']
            status_date = datetime.datetime.strptime(status['date'], "%Y-%m-%d").date()
            formatted_today = datetime.date.today()
            formatted_expiry_date = datetime.datetime.strptime(MSG_CONF[msg_id]['expiry_date'], "%Y-%m-%d").date()
            if formatted_today < formatted_expiry_date:
                if 'max_total_count' in MSG_CONF[msg_id].keys():
                    if actual_total_count < MSG_CONF[msg_id]['max_total_count']:
                        if 'max_day_count' in MSG_CONF[msg_id].keys():
                            if actual_day_count < MSG_CONF[msg_id]['max_day_count']:
                                return True
                            elif status_date < formatted_today:
                                return True
        return False
    except:
        return False
    
def update_msg_arg_conditions(msg_id):
    """
    Updates the message karg status.
    """
    status = get_json_msg_args(msg_id)
    if status['enabled']:
        if set(['day_count','date']).issubset(set(status.keys())):
            today = datetime.date.today()
            status_date = datetime.datetime.strptime(status['date'], "%Y-%m-%d").date()
            if today == status_date:
                status['day_count'] += 1
            else:
                status['day_count'] = 1
                status['date'] = today.strftime("%Y-%m-%d")
        if 'total_count' in status.keys():
            status['total_count'] += 1
        edit_json_messages(msg_id, **status)

def display_text_message(msg_id=0):
    """
    Displays a text message.
    """
    if msg_id in MSG_CONF.keys():
        if check_msg_arg_conditions(msg_id):
            rnd_val = np.random.rand()
            if rnd_val < MSG_CONF[msg_id]['probability']:
                update_msg_arg_conditions(msg_id)
                print("\n=========================================")
                print("ANNOUNCEMENT:\n")
                print(MSG_TEXT[msg_id]['text'], end='\n\n')
                print("=========================================")
                print(f'This message can be disabled:\n   python -c "import emobpy; emobpy.msg_disable({str(int(msg_id))})"\n')
                

            
def display_all():
    """
    Displays all messages.
    """
    for msg_id in MSG_CONF.keys():
        display_text_message(msg_id)