"""
List of some constants used for the calculations.

For more details see the article and cite:

.. code-block:: python

    @article{Gaete-Morales_2021,
    author={Gaete-Morales, Carlos and Kramer, Hendrik and Schill, Wolf-Peter and Zerrahn, Alexander},
    title={An open tool for creating battery-electric vehicle time series from empirical data, emobpy},
    journal={Scientific Data}, year={2021}, month={Jun}, day={11}, volume={8}, number={1}, pages={152},
    issn={2052-4463}, doi={10.1038/s41597-021-00932-9}, url={https://doi.org/10.1038/s41597-021-00932-9}}

"""

import src.external.emobpy as emobpy
from config import config
import os

CWD = os.getcwd()
DEFAULT_DATA_DIR = config.EMOBPY_DATA_PATH# tempfile.gettempdir()
USER_PATH = os.environ.get('EMOBPY_DATA_DIR', config.EMOBPY_DATA_PATH)
USER_DIR = config.EMOBPY_DATA_PATH
MODULE_PATH = emobpy.__path__[0]
DATA_DIR = 'data'
MODULE_DATA_PATH = os.path.join(MODULE_PATH,DATA_DIR)
WEATHER_FILES = {'ERA5-hourly-dew_point_temp.csv':'hourly-dew_point_temp_Kelvin.csv',
                'ERA5-hourly-mslp.csv':'hourly-sea_level_pressure_Pascal.csv',
                'ERA5-hourly-t2m.csv':'hourly-temp_Kelvin.csv'}

WEATHER_OPTIONS = {'dew_point Kelvin':'hourly-dew_point_temp_Kelvin.csv',
                'pressure Pascal':'hourly-sea_level_pressure_Pascal.csv',
                'temp Kelvin':'hourly-temp_Kelvin.csv'}

EVSPECS_FILE = 'evspecs.json'
MG_EFFICIENCY_FILE = f"motor_efficiency.csv"
DC_FILE = 'driving_cycles.yaml'
COUNTRY_CODE_ZONES_FILE = 'country_code_zones.json'

VEHICLE_NEEDED_PARAMETERS = ['battery_charging_eff','battery_discharging_eff',
                        'transmission_eff','auxiliary_power', 'cabin_volume',
                        'hvac_cop_heating','hvac_cop_cooling']

TARGET_TEMP = {'heating':18,'cooling': 20} # degC
GRAVITY = 9.81 # m/s2

LAYER_NAMES = ['metal','glassLaminated','glassTempered', 'polyurethane foam', 'polyester+fiberglass', 'fiberglass']
ZONE_NAMES = ['lateral_windows','windshields', 'backlite', 'rest']
ZONE_LAYERS = {'lateral_windows': {'metal': 0, 'glassLaminated': 0,'glassTempered':1, 'polyurethane foam': 0, 'polyester+fiberglass': 0, 'fiberglass':0},
               'windshields':     {'metal': 0, 'glassLaminated': 1,'glassTempered':0, 'polyurethane foam': 0, 'polyester+fiberglass': 0, 'fiberglass':0},
               'backlite':        {'metal': 0, 'glassLaminated': 0,'glassTempered':1, 'polyurethane foam': 0, 'polyester+fiberglass': 0, 'fiberglass':0},
               'rest':            {'metal': 1, 'glassLaminated': 0,'glassTempered':0, 'polyurethane foam': 1, 'polyester+fiberglass': 1, 'fiberglass':1}}
ZONE_SURFACE = {'lateral_windows': 1.5, 'windshields': 1.7, 'backlite': 1.4, 'rest': 9.9}  # m2
LAYER_CONDUCTIVITY = {'metal': 60, 'glassLaminated': 0.6, 'glassTempered': 1.38, 'polyurethane foam': 0.022, 'polyester+fiberglass': 0.64, 'fiberglass':2}  # W/(mK) ref: 10.1016/j.applthermaleng.2014.07.044
LAYER_THICKNESS = {'metal': 0.0009, 'glassLaminated': 0.0045, 'glassTempered': 0.0035, 'polyurethane foam': 0.058, 'polyester+fiberglass': 0.002, 'fiberglass':0.001}  # meter

GROUPID = {'freetime':1,'commuter':2}

VARIABLES_DB = {
    'driving': [
        'name', 'kind', 'profile', 't', 'totalrows', 'hours', 'f', 'refdate',
        'states', 'timeseries'
    ],
    'consumption': [
        'name', 'input', 'kind', 'profile', 't', 'totalrows', 'hours', 'f',
        'refdate', 'states', 'vehicle', 'timeseries'
    ],
    'availability': [
        'name', 'input', 'kind', 'profile', 'timeseries', 'chargingdata',
        'battery_capacity', 'charging_eff', 'discharging_eff', 'soc_init',
        'soc_min', 't', 'totalrows', 'refdate', 'success', 'notation', 'description', 'soc_end',
    ],
    'charging': ['name', 'kind', 'input', 'option', 'timeseries', 'success']
}

OPERATORS = {'last_trip_to': "==",
             'first_trip_to': "==",
             'not_last_trip_to': "!=",
             'overall_min_time_at': ">=",
             'overall_max_time_at': "<=",
             'min_state_duration': ">=",
             'max_state_duration': "<=",
             'equal_state_and_destination': "=="
             }

WEEKS = {0: {'day': 'Sunday', 'week': 'weekend', 'day_code': 'sunday'},
         1: {'day': 'Monday', 'week': 'weekday', 'day_code': 'weekdays'},
         2: {'day': 'Tuesday', 'week': 'weekday', 'day_code': 'weekdays'},
         3: {'day': 'Wednesday', 'week': 'weekday', 'day_code': 'weekdays'},
         4: {'day': 'Thursday', 'week': 'weekday', 'day_code': 'weekdays'},
         5: {'day': 'Friday', 'week': 'weekday', 'day_code': 'weekdays'},
         6: {'day': 'Saturday', 'week': 'weekend', 'day_code': 'saturday'}
         }

TIME_FREQ = {1.0: {'f': 'H'}, 0.5: {'f': '30min'}, 0.25: {'f': '15min'}, 0.125: {'f': '450s'}, 1/60:{'f':'60s'}, 1/3600: {'f': '1s'}}

RULE = {'weekday':
            {'n_trip_out': [1],
             'max_same_trip_times': False,
             'last_trip_to':{'home':False,'errands':False,'escort':False,'shopping':False,'leisure':False,'workplace':False,'driving':False},
             'first_trip_to':{'home':False,'errands':False,'escort':False,'shopping':False,'leisure':False,'workplace':False,'driving':False},
             'not_last_trip_to':{'home':False,'errands':False,'escort':False,'shopping':False,'leisure':False,'workplace':False,'driving':False},
             'at_least_one_trip':{'home':False,'errands':False,'escort':False,'shopping':False,'leisure':False,'workplace':False,'driving':False},
             'overall_min_time_at':{'home':False,'errands':False,'escort':False,'shopping':False,'leisure':False,'workplace':False,'driving':False},
             'overall_max_time_at':{'home':False,'errands':False,'escort':False,'shopping':False,'leisure':False,'workplace':False,'driving':False},
             'min_state_duration':{'home':0.5,'errands':0.5,'escort':0.5,'shopping':0.5,'leisure':0.5,'workplace':0.5,'driving':False},
             'max_state_duration':{'home':False,'errands':False,'escort':False,'shopping':False,'leisure':False,'workplace':False,'driving':False},
             'equal_state_and_destination':{'home':True,'errands':True,'escort':True,'shopping':True,'leisure':True,'workplace':True,'driving':True}
            },
        'weekend':
            {'n_trip_out': [1],
             'max_same_trip_times': False,
             'last_trip_to':{'home':False,'errands':False,'escort':False,'shopping':False,'leisure':False,'workplace':False,'driving':False},
             'first_trip_to':{'home':False,'errands':False,'escort':False,'shopping':False,'leisure':False,'workplace':False,'driving':False},
             'not_last_trip_to':{'home':False,'errands':False,'escort':False,'shopping':False,'leisure':False,'workplace':False,'driving':False},
             'at_least_one_trip':{'home':False,'errands':False,'escort':False,'shopping':False,'leisure':False,'workplace':False,'driving':False},
             'overall_min_time_at':{'home':False,'errands':False,'escort':False,'shopping':False,'leisure':False,'workplace':False,'driving':False},
             'overall_max_time_at':{'home':False,'errands':False,'escort':False,'shopping':False,'leisure':False,'workplace':False,'driving':False},
             'min_state_duration':{'home':0.5,'errands':0.5,'escort':0.5,'shopping':0.5,'leisure':0.5,'workplace':0.5,'driving':False},
             'max_state_duration':{'home':False,'errands':False,'escort':False,'shopping':False,'leisure':False,'workplace':False,'driving':False},
             'equal_state_and_destination':{'home':True,'errands':True,'escort':True,'shopping':True,'leisure':True,'workplace':True,'driving':True}
            }
       }


AIR_SPECIFIC_HEAT = {-50: 1002.6145045340269,
                    -49: 1002.6265326256818,
                    -48: 1002.6388632461362,
                    -47: 1002.6515013571064,
                    -46: 1002.6644519310647,
                    -45: 1002.6777199501787,
                    -44: 1002.6913104052605,
                    -43: 1002.7052282947265,
                    -42: 1002.7194786235676,
                    -41: 1002.7340664023328,
                    -40: 1002.7489966461204,
                    -39: 1002.7642743735844,
                    -38: 1002.7799046059481,
                    -37: 1002.7958923660334,
                    -36: 1002.812242677299,
                    -35: 1002.8289605628901,
                    -34: 1002.8460510447015,
                    -33: 1002.8635191424484,
                    -32: 1002.8813698727547,
                    -31: 1002.8996082482483,
                    -30: 1002.9182392766676,
                    -29: 1002.9372679599825,
                    -28: 1002.9566992935255,
                    -27: 1002.9765382651323,
                    -26: 1002.9967898542995,
                    -25: 1003.0174590313453,
                    -24: 1003.0385507565879,
                    -23: 1003.0600699795325,
                    -22: 1003.0820216380705,
                    -21: 1003.1044106576894,
                    -20: 1003.1272419506906,
                    -19: 1003.1505204154249,
                    -18: 1003.1742509355313,
                    -17: 1003.1984383791925,
                    -16: 1003.223087598396,
                    -15: 1003.2482034282104,
                    -14: 1003.2737906860681,
                    -13: 1003.2998541710605,
                    -12: 1003.3263986632443,
                    -11: 1003.353428922955,
                    -10: 1003.3809496901337,
                    -9: 1003.4089656836603,
                    -8: 1003.4374816007012,
                    -7: 1003.4665021160616,
                    -6: 1003.4960318815537,
                    -5: 1003.5260755253667,
                    -4: 1003.5566376514543,
                    -3: 1003.5877228389253,
                    -2: 1003.6193356414492,
                    -1: 1003.6514805866657,
                    0: 1003.6841621756065,
                    1: 1003.7173848821271,
                    2: 1003.7511531523458,
                    3: 1003.78547140409,
                    4: 1003.8203440263577,
                    5: 1003.8557753787805,
                    6: 1003.8917697911013,
                    7: 1003.9283315626558,
                    8: 1003.9654649618677,
                    9: 1004.0031742257493,
                    10: 1004.0414635594101,
                    11: 1004.0803371355768,
                    12: 1004.1197990941209,
                    13: 1004.1598535415919,
                    14: 1004.2005045507633,
                    15: 1004.2417561601842,
                    16: 1004.2836123737395,
                    17: 1004.3260771602194,
                    18: 1004.3691544528953,
                    19: 1004.4128481491074,
                    20: 1004.4571621098555,
                    21: 1004.5021001594022,
                    22: 1004.5476660848831,
                    23: 1004.5938636359228,
                    24: 1004.6406965242622,
                    25: 1004.6881684233927,
                    26: 1004.7362829681956,
                    27: 1004.7850437545954,
                    28: 1004.8344543392153,
                    29: 1004.884518239043,
                    30: 1004.9352389311056,
                    31: 1004.9866198521497,
                    32: 1005.0386643983317,
                    33: 1005.091375924914,
                    34: 1005.1447577459705,
                    35: 1005.1988131341002,
                    36: 1005.2535453201463,
                    37: 1005.3089574929251,
                    38: 1005.3650527989613,
                    39: 1005.4218343422342,
                    40: 1005.4793051839256,
                    41: 1005.5374683421813,
                    42: 1005.5963267918768,
                    43: 1005.6558834643926,
                    44: 1005.7161412473946,
                    45: 1005.7771029846256,
                    46: 1005.8387714757008,
                    47: 1005.9011494759131,
                    48: 1005.9642396960461,
                    49: 1006.028044802192}
