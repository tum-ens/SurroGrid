"""
Based on the vehicle mobility time series, the driving electricity consumption (ii) time series is derived. 
This requires further input data, such as information on nominal motor power, curb weight, drag coefficient, and dimensions, 
which the tool includes for several current BEV models. Ambient temperature is also a significant parameter that affects the 
consumption of BEV. For that reason, emobpy is endowed with a database of hourly temperature for European countries with a registry 
of the last 17 years. Additionally, the vehicle cabin insulation characteristics are required; this data is not widely available 
and thus assumed independently of the BEV models database. Driving cycles are also important input parameters that are used to 
simulate every individual trip. The model includes two driving cycles, Worldwide Harmonized Light Vehicles Test Cycle (WLTC) 
and Environmental Protection Agency (EPA). This input data is already provided within the tool, and the user can select a particular 
BEV model, country weather, and driving cycle. Alternatively, emobpy also allows providing user-defined custom data.

For more details see the article and cite:

.. code-block:: python

    @article{Gaete-Morales_2021,
    author={Gaete-Morales, Carlos and Kramer, Hendrik and Schill, Wolf-Peter and Zerrahn, Alexander},
    title={An open tool for creating battery-electric vehicle time series from empirical data, emobpy},
    journal={Scientific Data}, year={2021}, month={Jun}, day={11}, volume={8}, number={1}, pages={152},
    issn={2052-4463}, doi={10.1038/s41597-021-00932-9}, url={https://doi.org/10.1038/s41597-021-00932-9}}

See also the examples in the documentation https://diw-evu.gitlab.io/emobpy/emobpy

"""

# import zenodo_get
from datetime import datetime
import pandas as pd
import numpy as np
from numba import jit
import yaml
import os
import uuid
import gzip
import pickle
import json

from src.external.emobpy.constants import (
    TIME_FREQ,
    DEFAULT_DATA_DIR,
    USER_PATH,
    MODULE_DATA_PATH,
    WEATHER_OPTIONS,
    EVSPECS_FILE,
    MG_EFFICIENCY_FILE,
    DC_FILE,
    COUNTRY_CODE_ZONES_FILE,
    LAYER_NAMES,
    ZONE_NAMES,
    ZONE_LAYERS,
    ZONE_SURFACE,
    LAYER_CONDUCTIVITY,
    LAYER_THICKNESS,
    TARGET_TEMP,
    GRAVITY,
    VEHICLE_NEEDED_PARAMETERS,
)

from src.external.emobpy.functions import (
    inertial_mass,
    include_weather,
    rolling_resistance_coeff,
    vehicle_mass,
    prollingresistance,
    pairdrag,
    p_gravity,
    pinertia,
    p_wheel,
    p_motorout,
    EFFICIENCYregenerative_braking,
    p_generatorin,
    p_motorin,
    p_generatorout,
    qhvac
)
from src.external.emobpy.tools import (Unit, check_for_new_function_name, _add_column_datetime, consumption_progress_bar, wget_progress_bar, display_all)
from src.external.emobpy.init import copy_to_user_data_dir
from src.external.emobpy.logger import get_logger

logger = get_logger(__name__)


######################################################################
# These functions are for electricity consumption profile creation ###
######################################################################


class Weather:
    def __init__(self):
        pass

    def __getattr__(self, item):
        check_for_new_function_name(item)
        # if the return value is not callable, we get TypeError:

    def temp(self, country_code, year):
        """
        Loads selected temperature data in Kelvin into object.

        Args:
            country_code (str): E.g. 'DE'.
            year (int): E.g. 2016.

        Returns:
            list: Temperature data.
        """
        return self._load_data(country_code, year) - 273.15

    def pressure(self, country_code, year):
        """
        Loads selected pressure data in Pascal into object.

        Args:
            country_code (str): E.g. 'DE'.
            year (int): E.g. 2016.

        Returns:
            list: Pressure data.
        """
        return self._load_data(country_code, year, option="pressure Pascal", location=None) / 100

    def dewpoint(self, country_code, year):
        """
        Loads selected dew point data in Kelvin into object.

        Args:
            country_code (str): E.g. 'DE'.
            year (int): E.g. 2016.

        Returns:
            list: Dew point data.
        """
        return self._load_data(country_code, year, option="dew_point Kelvin") - 273.15

    # @staticmethod
    # def download_weather_data(location=None):
    #     """
    #     Download weather data from zenodo.

    #     Args:
    #         location (str, optional): Path to user path. Defaults to None.

    #     Returns:
    #         list: Weather data.
    #     """
    #     user_dir = location or USER_PATH or DEFAULT_DATA_DIR + "/user_files"
    #     os.makedirs(user_dir, exist_ok=True)
    #     os.chdir(user_dir)
    #     zenodo_get.zenodo_get(["10.5281/zenodo.1489915", "-wurls.txt"])
    #     os.chdir(CWD)
    #     time.sleep(2)
    #     fh = open(os.path.join(user_dir, "urls.txt"))
    #     text_list = []
    #     for line in fh:
    #         text_list.append(line)
    #     fh.close()
    #     dest_list = []
    #     for url in text_list:
    #         for file in WEATHER_FILES.keys():
    #             if file in url:
    #                 filename = os.path.join(user_dir, WEATHER_FILES[file])
    #                 if os.path.exists(filename):
    #                     os.remove(filename)
    #                 print(f"Downloading file... {url.strip()}")
    #                 dest = wget.download(url.strip(), filename, bar=wget_progress_bar)
    #                 print("")
    #                 dest_list.append(dest)
    #     for dfp in dest_list:
    #         logger.info(dfp)
    #     return dest_list

    @staticmethod
    def _load_data(country_code="DE", year=2016, option="temp Kelvin", location=None):
        """
        Load data from csv files and configure it in a DataFrame.

        Args:
            country_code (str, optional): Defaults to "DE".
            year (int, optional): Defaults to 2016.
            option (str, optional): Defaults to "temp Kelvin".
            location (str, optional): Path where data file is stored. Defaults to None.

        Returns:
            pd.DataFrame: Loaded data.
        """

        user_dir = location or USER_PATH or DEFAULT_DATA_DIR
        filename = os.path.join(user_dir, WEATHER_OPTIONS[option])
        country_timezones_file = os.path.join(user_dir, COUNTRY_CODE_ZONES_FILE)
        with open(country_timezones_file) as file:
            country_timezones = json.load(file)
        if country_code not in country_timezones:
            raise Exception(f"Country code {country_code} not in file: {country_timezones_file}. Select another country code or include it in the file.")
        dateparse = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S %Z")
        df = pd.read_csv(filename, parse_dates=["date"], date_parser=dateparse, usecols=["date", country_code])
        timezones = country_timezones[country_code]
        df.date = pd.DatetimeIndex(df.date).tz_localize("GMT").tz_convert(timezones[0])
        data = df[df.date.dt.year.isin([year])]
        logger.info(" ".join([option, country_code, str(year), "Timezone:", timezones[0]]))
        return data.set_index("date").reset_index(drop=True)[country_code][0:8760].values

    @staticmethod
    def calc_vapor_pressure(t):
        """
        Calculate vapor pressure.

        Args:
            t (array): Dew point or air temperature in degree Celsius.

        Returns:
            array: Vapor pressure array
        """
        T = t  # dew point or air temp degC
        E = 6.11 * np.power(
            10, ((7.5 * T) / (237.3 + T))
        )  # saturated  vapor pressure (mb) if t is dewpoint
        return E

    def calc_rel_humidity(self, Dp, T):
        """
        Calculate humidity.

        Args:
            Dp (array): Dew point temperature in degree Celsius.
            T (array): Temperature in degree Celsius.

        Returns:
            array: Relative humidity in percent.
        """
        H = (
                100 * self.calc_vapor_pressure(Dp) / self.calc_vapor_pressure(T)
        )  # Relative humidity (percentage)
        return H

    @staticmethod
    def calc_dew_point(t, h):
        """
        Calculate dew point.
        Args:
            t (float): air temperature in degree Celsius.
            h (float): Relative humidity in percent.

        Returns:
            array:
        """
        T = t
        H = h
        Td = (
                243.04
                * (np.log(H / 100) + ((17.625 * T) / (243.04 + T)))
                / (17.625 - np.log(H / 100) - ((17.625 * T) / (243.04 + T)))
        )
        return Td

    @staticmethod
    def calc_dry_air_partial_pressure(P, pv):
        """
        Calculate dry air partial pressure.

        Args:
            P (float): [description]
            pv (float): [description]

        Returns:
            array: Dry air partial pressure.
        """
        pair = P - pv
        return pair

    @staticmethod
    def air_density_from_ideal_gas_law(t, p):
        """
        Calculate Air density from ideal gas law.

        Args:
            t (float): Temperature in degree Celsius.
            p (float): Pressure in mega bar.

        Returns:
            array: Air density
        """
        T = t + 273.15  # convert degC => degK
        P = 100 * p  # convert mb   => Pascals
        N = 287.05  # specific gas constant , J/(kg*degK) = 287.05 for dry air
        return P / (N * T)

    def humidair_density(self, t, p, dp=None, h=None):
        """
        Calculate humid air density.

        Args:
            t (array): Temperature in degree Celsius.
            p (array): Pressure in mbar.
            dp (array, optional): Dew point temperature in degree Celsius. Defaults to None.
            h (array, optional): Humidity in percent. Defaults to None.

        Raises:
            Exception: Dp or h is missing.

        Returns:
            array: Humid air density.
        """

        if dp is None and h is None:
            raise Exception("One value is required, either dp or h")
        if dp is not None:
            pv = self.calc_vapor_pressure(dp)  # mbar
        if h is not None:
            pv = self.calc_vapor_pressure(self.calc_dew_point(t, h))  # mbar

        pd = self.calc_dry_air_partial_pressure(p, pv)  # mbar

        Pv = 100 * pv  # convert mb   => Pascals
        Pd = 100 * pd  # convert mb   => Pascals

        Rd = 287.05  # specific gas constant for dry air [J/(kgK)]
        Rv = 461.495  # specific gas constant for water vapour [J/(kgK)]
        T = t + 273.15  # convert degC => degK

        AD = Pd / (Rd * T) + Pv / (Rv * T)  # density [kg/m3]
        return AD


class BEVspecs:
    
    def __init__(self, filename=None):
        self.filename = filename
        self.data = []
        self.parameters = [
            "acc_0_100_kmh",
            "axle_ratio",
            "battery_cap",
            "curb_weight",
            "drag_coeff",
            "motor_type",
            "height",
            "length",
            "market",
            "num_cells",
            "num_modules",
            "power",
            "reg_braking",
            "top_speed",
            "torque",
            "trunk_volume",
            "battery_type",
            "voltage",
            "weight",
            "width",
        ]
        self._load_specs(filename=self.filename)

    def __getattr__(self, item):
        check_for_new_function_name(item)
        # if the return value is not callable, we get TypeError:

    def _load_specs(self, filename=None):
        """
        Load specification based on specification file.

        Args:
            filename (str, optional): Name of specification file to be loaded. Defaults to None.
        """

        if filename is None:
            user_dir = USER_PATH or DEFAULT_DATA_DIR
            self.filename = os.path.join(user_dir, EVSPECS_FILE)
        else:
            self.filename = filename

        with open(self.filename) as file:
            self.data = json.load(file)

    def search_by_parameter(self, parameter='power', first_x=10, brand_filter=[], model_filter=[],
                            year_filter=[]):
        """
        Searching for vehicles sorted in descending order of given parameter. It returns a Pandas DataFrame.

        Args:
            parameter (str): Vehicle parameter to compare. Defaults to 'power'
            first_x (int): Number of vehicles to show. Defaults to 10.
            brand_filter (int): Filter for brands. Defaults to [].
            model_filter (int): Filter for models. Defaults to [].
            year_filter (int): Filter for years. Defaults to [].

        Returns:
            data (pd.DataFrame)
        """

        print_dict = json.loads(json.dumps(self.data))
        print_dict.pop('fallback_parameters')

        df = pd.DataFrame(columns=['brand', 'model', 'year', 'value', 'unit'])

        for brand_name, brand_values in self.data.items():
            if brand_name == 'fallback_parameters':
                continue
            if brand_filter and brand_name not in brand_filter:
                continue
            for model_name, model_values in brand_values.items():
                if model_filter and model_name not in model_filter:
                    continue
                for year_name, year_values in model_values.items():
                    if year_filter and year_name not in year_filter:
                        continue
                    for para_name, para_value in year_values.items():
                        if para_name == parameter:
                            df = df.append(
                                {'brand': brand_name, 'model': model_name, 'year': year_name, 'value': para_value["value"], 'unit': para_value['unit']},
                                ignore_index=True)
        logger.info(f'Parameter: {parameter}')
        data = df.sort_values(by='value', ascending=False).reset_index(drop=True)
        logger.info(data.head(first_x))
        data['parameter'] = parameter
        return data

    def show_models(self, brand='', model='', year=''):
        """
        Shows a list of all cars from the database. Can be filtered by brand, model and year.

        Args:
            brand (str, optional): Show only cars that match the brand. Defaults to ''.
            model (str, optional): Show only cars that match the model. Defaults to ''.
            year (str, optional): Show only cars that match the year. Defaults to ''.
        """

        def pretty(d, indent=0):
            for key, value in d.items():
                if value:
                    logger.info('\t' * indent + str(key))
                    if isinstance(value, dict) and indent < 2:
                        pretty(value, indent + 1)

        print_dict = json.loads(json.dumps(self.data))
        print_dict.pop('fallback_parameters')

        for brand_name, brand_values in self.data.items():
            if brand_name == 'fallback_parameters':
                continue
            if brand and brand != brand_name:
                print_dict.pop(brand_name)
            for model_name, model_values in brand_values.items():
                if model and model != model_name:
                    print_dict[brand_name].pop(model_name)
                for year_name, year_values in model_values.items():
                    if year and year != year_name:
                        print_dict[brand_name][model_name].pop(year_name)
        pretty(print_dict)

    def get(self, brand, model, year, parameter):
        """
        Search for specific information in the vehicle database and returns it.

        Args:
            brand (str): E.g 'Volkswagen'
            model (str): E.g. 'ID.3'
            year (int): E.g. 2020
            parameter (str): E.g. 'acc_0_100_kmh'

        Returns:
            [float]: Requested value. None if nothing was found.
        """
        if parameter == 'Brand':
            return brand
        if parameter == 'EV Model':
            return model
        if parameter == 'Model year':
            return year

        if parameter in self.parameters:
            try:
                unit_data = self.data[model][parameter]
                param = Unit(val=unit_data['value'], unit=unit_data['unit'], description=unit_data['description'])
                param.convert_to_default_value()
                return param.val
            except KeyError:
                logger.info("No value found! verify there is no typo")
        else:
            logger.info("No value found!")


    def get_fallback_parameter(self, parameter):
        """
        Get data for a given parameter if it is missing.

        Args:
            parameter (str): Parameter to get data value for.

        Returns:
            float: Fallback data value for given parameter.
        """
        try:
            unit_data = self.data['fallback_parameters'][parameter]
            param = Unit(val=unit_data['value'], unit=unit_data['unit'], description=unit_data['description'])
            param.convert_to_default_value()

            return param.val
        except KeyError:
            return None

    def dropna_model(self, parameter):
        """
        Delete all na in self.data for given parameter.

        Args:
            parameter (str): Parameter from which to delete.
        """
        for brand, brand_values in self.data.items():
            for model, model_values in brand_values.items():
                for year, year_values in model_values.items():
                    for param, param_value in year_values.items():
                        if param_value.__class__.__name__ in ["float", "int"] and param_value is not None:
                            logger.info(f"Delete {brand} {model} {year}")
                            self.data[brand][model].pop(year)

    def replacena_model(self, parameter, default):
        """
        Replace all na in self.data for given parameter with default value.

        Args:
            parameter (str): Parameter from which to delete.
            default (float): Value to be used instead.
        """
        for brand, brand_values in self.data.items():
            for model, model_values in brand_values.items():
                for year, year_values in model_values.items():
                    for param, param_value in year_values.items():
                        if param_value.__class__.__name__ in ["float", "int"] and param_value is None:
                            logger.info(f"Added {default} to {brand} {model} {year}")
                            self.data[brand][model][year][parameter] = default

    def maximum(self, parameter):
        """
        Returns maximum of specific parameter for the object.

        Args:
            parameter (str): Parameter of which the maximum is required.

        Returns:
            float: Maximum of the parameter.
        """
        value = []
        for brand, brand_values in self.data.items():
            for model, model_values in brand_values.items():
                for year, year_values in model_values.items():
                    for param, param_value in year_values.items():
                        if param == parameter and \
                                param_value.__class__.__name__ in ["float", "int"] and \
                                param_value is not None:
                            value.append(param_value)
        return max(value)

    def average(self, parameter):
        """
        Returns average of a given parameter from self.data.

        Args:
            parameter (str): Parameter of which the average is required.

        Returns:
            float: Average of the parameter.
        """
        value = []
        for brand, brand_values in self.data.items():
            for model, model_values in brand_values.items():
                for year, year_values in model_values.items():
                    for param, param_value in year_values.items():
                        if param == parameter and \
                                param_value.__class__.__name__ in ["float", "int"] and \
                                param_value is not None:
                            value.append(param_value)
        return round(sum(value) / len(value), 3)

    def model(self, model, use_fallback=True, msg=True):
        """
        Initializes ModelSpecs object, adds parameters and checks them.

        Args:
            model (tuple): Data of model. E.g ('Volkswagen', 'ID.3', 2020)
            use_fallback:
            msg (bool, optional): Flag, whether to inform about missing parameters. Defaults to True.

        Returns:
            ModelSpecs object: Initialized object.
        """
        M = ModelSpecs(model, self)
        M.add_parameters()
        M.add_calculated_param()
        if use_fallback:
            M.add_fallback_data()
        self._ev_par_test(M, msg=msg)
        return M

    def _ev_par_test(self, Model, msg=True):
        """
        Checks whether all relevant parameters have been saved.

        Args:
            Model (Model object): Vehicle model to be checked.
            msg (bool, optional): Flag, whether missing messages should be printed. Defaults to True.
        """
        param_missing = []
        for wanted in VEHICLE_NEEDED_PARAMETERS:
            flag = False
            for parameter in Model.parameters:
                if wanted == parameter:
                    if not isinstance(Model.parameters[parameter], (int, float)):
                        flag = False
                        break
                    flag = True
                    break
            if not flag:
                param_missing.append(wanted)
        if len(param_missing) != 0:
            if msg:
                logger.info("Missing relevant parameters:")
                for name in param_missing:
                    logger.info(f"   {name}")
                logger.info('Please, add these parameters to the model instance < model_instance.add({"parameter":value}) >')

    def save(self):
        """
        Save self.data into .yml file.
        """
        with open(self.filename, "w") as file:
            json.dump(self.data, file, indent=4)
        logger.info(f"File saved: {self.filename}")
        logger.info(f"File saved: {self.filename}")


class ModelSpecs:
    
    def __init__(self, model, BEVspecs_instance):
        self.name = model
        self.parameters = {}
        self.db = BEVspecs_instance
        for parameter in self.db.parameters:
            self.parameters[parameter] = None
        self.parameters["Brand"] = model[0]
        self.parameters["EV Model"] = model[1]
        self.parameters["Model year"] = model[2]

    def __getattr__(self, item):
        check_for_new_function_name(item)
        # if the return value is not callable, we get TypeError:

    def add_parameters(self):
        """
        Adds a value from the database to all parameters in self.db.parameters.
        """
        for parameter in self.db.parameters:
            value = self.db.get(*self.name, parameter)
            self.parameters[parameter] = value

    def add_calculated_param(self):
        """
        Calculate all parameters that can be calculated from existing information.
        """
        if (
                "power" in self.parameters
                and "curb_weight" in self.parameters
        ):
            self.parameters["pmr"] = self._pmr(
                self.parameters["power"],
                self.parameters["curb_weight"],
            )
        else:
            self.parameters["pmr"] = None
        if (
                "curb_weight" in self.parameters
                and "axle_ratio" in self.parameters
        ):
            self.parameters["inertial_mass"] = inertial_mass(
                self.parameters["curb_weight"],
                self.parameters["axle_ratio"],
            )
        else:
            self.parameters["inertial_mass"] = None
        if (
                "width" in self.parameters
                and "height" in self.parameters
        ):
            self.parameters["front_area"] = self._frontal_area(
                self.parameters["height"],
                self.parameters["width"],
            )
        else:
            self.parameters["front_area"] = None

    def add_fallback_data(self):
        for parameter in VEHICLE_NEEDED_PARAMETERS:
            if parameter not in self.parameters:
                fallback = self.db.get_fallback_parameter(parameter)
                self.parameters[parameter] = fallback
                logger.info(f"Fallback value {fallback} added for missing {parameter} parameter.")

    def _pmr(self, power, curb_weight):
        """Calculates PMR.

        Args:
            power (int): Power of the vehicle.
            curb_weight (int): Empty weight of the vehicle.

        Returns:
            int: PMR value for the vehicle
        """
        return power * 1000 / curb_weight  # W/kg

    def _frontal_area(self, height, width):
        """
        Calculate and returns size of frontal area.

        Args:
            height (float): Height of the vehicle.
            width (float): Width of the vehicle.

        Returns:
            float: Frontal area of the vehicle in square.
        """
        return height * width

    def addtodb(self):
        """
        Adds parameters, which are not None, to database.

        Returns:
            None, List of None parameters: Returns list of None parameters, which can not be added to database.
        """
        nones = []
        flag = False
        for param in self.parameters:
            if self.parameters[param] is None:
                flag = True
                nones.append(param)
        if flag:
            logger.info("The model can not be added to the database. It has parameters with None as value")
            return nones
        self.add_calculated_param()
        self.db.data.append(self.parameters)
        logger.info("Model added to the BEVspecs instance")
        return None

    def add(self, parameters_dict, msg=True):
        """
        Adds parameter and associated value to the object.

        Args:
            parameters_dict (dict): Contains the name of the parameters and the corresponding value.
            msg (bool, optional): Flag, whether to inform about added parameters. Defaults to True.
        """
        for k, v in parameters_dict.items():
            self.parameters[k] = v
            if k in VEHICLE_NEEDED_PARAMETERS:
                if msg:
                    logger.info(f"{k} has been added to the model object. Required OK")


class MGefficiency:
    
    def __init__(self, filename=None):
        self.data = None
        self.filename = filename
        self._load_file(filename=self.filename)
        self._get_codes()

    def __getattr__(self, item):
        check_for_new_function_name(item)
        # if the return value is not callable, we get TypeError:

    def _load_file(self, filename=None):
        """
        Loads bev efficiency from csv file.
        Args:
            filename (str, optional): Csv file with bev efficiency data. Defaults to None.
        """
        if filename is None:
            user_dir = USER_PATH or DEFAULT_DATA_DIR
            self.filename = os.path.join(user_dir, MG_EFFICIENCY_FILE)
        else:
            self.filename = filename
        with open(self.filename) as file:
            self.data = pd.read_csv(file)

    def _get_codes(self):
        """
        Loads code data into self.load_fraction, self.motor and self.generator.
        """
        self.load_fraction = self.data.load_fraction.values
        self.motor = self.data.motor.values
        self.generator = self.data.generator.values

    def get_efficiency(self, load_fraction, g_m_code):
        """
        g_m_code: 1 -> motor, -1 -> generator
        #TODO DOCSTRING
        Args:
            load_fraction ([type]):
            g_m_code (-1, 1): 1 for motor, -1 for generator

        Raises:
            Exception: Raised if g_m_code is not 1 or -1.

        Returns:
            type: [description]
        """
        if g_m_code not in [1, -1]:
            raise Exception(f"g_m_code is {g_m_code}. It should be 1 or -1")
        if g_m_code == 1:
            return self._get_efficiency(load_fraction, self.load_fraction, self.motor)
        elif g_m_code == -1:
            return self._get_efficiency(
                np.abs(load_fraction), self.load_fraction, self.generator
            )

    @staticmethod
    @jit(nopython=True)
    def _get_efficiency(load_fraction, load_fraction_values, column_values):
        """
        #TODO DOCSTRING
        Gets a one-dimensional linear interpolation of given arguments.

        Args:
            load_fraction ([type]): [description]
            load_fraction_values ([type]): [description]
            column_values ([type]): [description]

        Returns:
            float: Efficiency.
        """
        return np.interp(load_fraction, load_fraction_values, column_values)


@jit(nopython=True)
def acceleration(V0, V2):  # V0, V2 km/h
    """
    Calculate and returns acceleration.

    Args:
        V0 (float): Old speed.
        V2 (float): New speed.

    Returns:
        float: Acceleration.
    """
    acc = (V2 - V0) / 2 / 3.6  # acc m/s**2
    return acc


@jit(nopython=True)
def acceleration_array(speed_array):
    """
    Calculates and returns acceleration array from speed_array.
    The acceleration of the adjoining values is calculated.

    Args:
        speed_array (ndarray): Array with speed values.

    Returns:
        ndarray: Array with acceleration values.
    """
    acc = np.zeros((speed_array.shape[0],))
    acc[0] = acceleration(0, speed_array[1])
    i = 0
    for a, b in zip(speed_array[0:-2], speed_array[2:]):
        i += 1
        acc[i] = acceleration(a, b)
    return acc


class DrivingCycle:
    
    def __init__(self):
        self.data = []
        self.index_speed = None
        user_dir = USER_PATH or DEFAULT_DATA_DIR
        self.datafile = os.path.join(user_dir, DC_FILE)
        # self.load_data()

    def __getattr__(self, item):
        check_for_new_function_name(item)
        # if the return value is not callable, we get TypeError:

    def get_csv(self, csv_path=os.path.join(MODULE_DATA_PATH, "driving_cycles.csv")):
        """
        Load csv as dataframe into self.dc_df

        Args:
            csv_path (str, optional): Path of file. Defaults to os.path.join(MODULE_DATA_PATH, "driving_cycles.csv").
        """
        self.csv_path = csv_path
        self.dc_df = pd.read_csv(self.csv_path, index_col="Seconds")

    def create_data(self):
        """
        Create self.data from self.dc_df.
        """
        self.data = []
        for i, dc_name in enumerate(self.dc_df.columns):
            dc = {}
            dc["name"] = dc_name
            dc["type"] = dc_name.split("_")[0]
            dc["id"] = i
            dc["speed"] = {}
            dc["speed"]["value"] = self.dc_df[dc_name].dropna().values
            dc["speed"]["unit"] = "km/h"
            dc["mean_speed"] = {}
            dc["mean_speed"]["value"] = round(float(np.mean(dc["speed"]["value"])), 1)
            dc["mean_speed"]["unit"] = "km/h"
            dc["time"] = {}
            dc["time"]["value"] = len(dc["speed"]["value"])
            dc["time"]["unit"] = "s"
            dc["distance"] = {}
            dc["distance"]["value"] = round(float(sum(dc["speed"]["value"] / 3600)), 1)
            dc["distance"]["unit"] = "km"
            dc["normalized"] = {}
            dc["normalized"]["value"] = np.round(
                dc["speed"]["value"] / dc["mean_speed"]["value"], 4
            )
            dc["normalized"]["unit"] = ""
            dc["acceleration"] = {}
            dc["acceleration"]["value"] = np.round(
                acceleration_array(dc["speed"]["value"]), 4
            )
            dc["acceleration"]["unit"] = "m/s**2"
            dc["max_acceleration"] = {}
            dc["max_acceleration"]["value"] = float(max(dc["acceleration"]["value"]))
            dc["max_acceleration"]["unit"] = "m/s**2"
            self.data.append(dc)
        self._get_index_speed()

    def save_data(self):
        """
        Save self.tmpdata to file.
        """
        self.tmpdata = []
        i = -1
        for dc in self.data:
            i += 1
            self.tmpdata.append(dc)
            for key in dc.keys():
                if isinstance(self.tmpdata[i][key], dict):
                    if isinstance(self.tmpdata[i][key]["value"], np.ndarray):
                        self.tmpdata[i][key]["value"] = self.tmpdata[i][key][
                            "value"
                        ].tolist()

        with open(self.datafile, "w") as file:
            yaml.dump(self.tmpdata, file)
        logger.debug(f"File saved: {self.datafile}")

    def load_data(self):
        """
        Load data from self.datafile to self.data.
        """
        if os.path.isfile(self.datafile):
            with open(self.datafile) as file:
                self.tmpdata = yaml.load(file, Loader=yaml.FullLoader)
            i = -1
            for dc in self.tmpdata:
                i += 1
                self.data.append(dc)
                for key in dc.keys():
                    if isinstance(self.data[i][key], dict):
                        if isinstance(self.data[i][key]["value"], list):
                            self.data[i][key]["value"] = np.array(
                                self.data[i][key]["value"]
                            )
            self._get_index_speed()
        else:
            logger.info(f'File "{self.datafile}" does not exist!. You can create it from .csv file')

    def _get_index_speed(self):
        """
        Loads index speed in self.index_speed.
        """
        self.index_speed = {}
        classes = ["WLTC_3b", "WLTC_2"]
        dc_types = list(set([t["type"] for t in self.data]))

        dc_class = []
        dc_full = []
        for t in self.data:
            flag = False
            for entry in classes:
                if entry in t["name"]:
                    flag = True
                    dc_class.append(entry)
                    if "full" in t["name"]:
                        dc_full.append(True)
                    else:
                        dc_full.append(False)
                    break
            if not flag:
                dc_class.append("none")
                if "full" in t["name"]:
                    dc_full.append(True)
                else:
                    dc_full.append(False)

        for type_ in dc_types:
            self.index_speed[type_] = {}
            if "WLTC" == type_:
                for cl in classes:
                    self.index_speed[type_][cl] = {"partial": {}, "full": {}}
            else:
                self.index_speed[type_]["none"] = {"partial": {}, "full": {}}

        for s in self.data:
            if dc_full[s["id"]]:
                key = "full"
            else:
                key = "partial"
            self.index_speed[s["type"]][dc_class[s["id"]]][key][s["id"]] = {
                "value": s["mean_speed"]["value"],
                "unit": s["mean_speed"]["unit"],
            }

    def _select_driving_cycle_index(self, driving_cycle_type, speed, PMR, full_driving_cycle):
        """
        #TODO DOCSTRING

        Args:
            driving_cycle_type (str): Type of driving cycle. E.g. 'WLTC'.
            speed (dict): Speed value. E.g. {'value': 11.11, 'unit': 'm/s'}
            PMR (float):
            full_driving_cycle (fool): [description]

        Returns:
            [type]: [description]
        """

        WLTC_class = "none"
        if driving_cycle_type == "WLTC":
            if PMR is not None:
                if PMR > 34:
                    WLTC_class = "WLTC_3b"
                elif 22 < PMR <= 34:
                    WLTC_class = "WLTC_2"

        idx = []
        spd = []
        if not full_driving_cycle:
            for k, v in self.index_speed[driving_cycle_type][WLTC_class][
                "partial"
            ].items():
                idx.append(k)
                driving_cycle_type_average_speed = Unit(v["value"], v["unit"]).convert_to("m/s").val
                trip_average_speed = Unit(speed["value"], speed["unit"]).convert_to("m/s").val

                spd.append(-abs(driving_cycle_type_average_speed - trip_average_speed))
            return idx[np.argmax(spd)]
        else:
            return list(
                self.index_speed[driving_cycle_type][WLTC_class]["full"].keys()
            )[0]

    def driving_cycle(self, trip, model, full_driving_cycle=False):
        """
        Calculates driving cycle from Trip and ModelSpecs object.

        Args:
            trip (Trip): Trip for the driving cycle.
            model (ModelSpecs): Vehicle Model for the driving cycle.
            full_driving_cycle (bool, optional): [description]. Defaults to False.
        """
        trip.index = self._select_driving_cycle_index(
            trip.driving_cycle_type,
            trip._mean_speed,
            model.parameters["pmr"],
            full_driving_cycle,
        )

        if full_driving_cycle:
            temp = Unit(self.data[trip.index]["time"]["value"], self.data[trip.index]["time"]["unit"])
            temp.convert_to("s")
            trip.duration["value"] = temp.val

            temp = Unit(self.data[trip.index]["distance"]["value"], self.data[trip.index]["distance"]["unit"])
            temp.convert_to("m")
            trip.distance["value"] = temp.val

            trip.get_mean_speed()

        # Todo use Unit class for vector calculations
        trip.time["value"] = np.ceil(Unit(trip._duration["value"], trip._duration["unit"]).convert_to("s").val)
        trip.time["unit"] = "s"

        scale = (trip.time["value"]
                // Unit(self.data[trip.index]["time"]["value"], self.data[trip.index]["time"]["unit"]).convert_to("s").val
        )
        slide = (
                trip.time["value"]
                % Unit(self.data[trip.index]["time"]["value"], self.data[trip.index]["time"]["unit"]).convert_to("s").val
        )
        normalized = self.data[trip.index]["normalized"]["value"]
        normalized_array = np.array(list(normalized) * int(scale) + list(normalized)[0: int(slide)])
        speed_array = (
                normalized_array
                * Unit(trip._mean_speed["value"], trip._mean_speed["unit"]).convert_to("m/s").val
        )
        i = 0
        for last_secs in range(-20, 0):
            i += 1
            calc = (
                    speed_array[last_secs - 1] - speed_array[last_secs - 1] * (i / 100) * 2
            )
            speed_array[last_secs] = max(0, calc)

        trip.speed["value"] = speed_array
        trip.speed["unit"] = "m/s"
        trip.acceleration["value"] = acceleration_array(speed_array)
        trip.acceleration["unit"] = "m/s**2"

        trip.driving_cycle_name = self.data[trip.index]["name"]


class Trips:
    
    def __init__(self):
        self.quantity = 0
        self.trips = []

    def __getattr__(self, item):
        check_for_new_function_name(item)
        # if the return value is not callable, we get TypeError:

    def get_code(self):
        """
        Returns trip code based on quantity of trip plus 1.

        Returns:
            int: Trip code.
        """
        code = self.quantity
        self.quantity += 1
        return code

    def add_trip(self, trip):
        """
        Adds single trip to trips object.[summary]

        Args:
            trip (Trip): Object to be added to the trips collection.
        """
        self.trips.append(trip)

    def get_trip(self, code):
        """
        Returns specific trip based on code.

        Args:
            code (int): Code of the requested trip.

        Returns:
            Trip: Requested trip object.
        """
        return self.trips[code]


class Trip:
    
    def __init__(self, trips):
        self.code = trips.get_code()
        self.origin = None
        self.destination = None
        self.distance = {"value": None, "unit": None}
        self._distance = {"value": None, "unit": None}
        self.duration = {"value": None, "unit": None}
        self._duration = {"value": None, "unit": None}
        self.start_trip_time = {"value": None, "unit": None}
        self.end_trip_time = {"value": None, "unit": None}
        self.mean_speed = {"value": None, "unit": None}
        self._mean_speed = {"value": None, "unit": None}
        self.driving_cycle_type = None
        self.driving_cycle_name = None
        self.acceleration = {"value": None, "unit": None}
        self.speed = {"value": None, "unit": None}
        self.time = {"value": None, "unit": None}
        self.index = None
        self.balance = None
        self.results = {}
        self.rate = {"value": None, "unit": None}
        self.consumption = {"value": None, "unit": None}
        trips.add_trip(self)

    def __getattr__(self, item):
        check_for_new_function_name(item)
        # if the return value is not callable, we get TypeError:

    def add_distance_duration(self, distance, duration):
        """
        Sets loaded distance and duration to self.distance and self.duration and saves mean_speed.

        Args:
            distance (dict): Dictionary containing value and unit. E.g. {'value': 10.0, 'unit': 'km'}
            duration (dict): Dictionary containing value and unit. E.g. {'value': 15.0, 'unit': 'min'}
        """
        self.distance["value"] = distance["value"]
        self.distance["unit"] = distance["unit"]
        self.duration["value"] = duration["value"]
        self.duration["unit"] = duration["unit"]
        self.get_mean_speed()

    def get_mean_speed(self):
        """
        Calculates mean_speed ans saves it in object attribute.
        """

        self._distance["value"] = Unit(self.distance["value"], self.distance["unit"]).convert_to("m").val
        self._distance["unit"] = "m"
        self._duration["value"] = Unit(self.duration["value"], self.duration["unit"]).convert_to("s").val
        self._duration["unit"] = "s"

        self.mean_speed = {
            "value": self.distance["value"] / self.duration["value"],
            "unit": self.distance["unit"] + "/" + self.duration["unit"],
        }
        self._mean_speed = {
            "value": self._distance["value"] / self._duration["value"],
            "unit": self._distance["unit"] + "/" + self._duration["unit"],
        }


class HeatInsulation:
    
    def __init__(self, default=False):
        self.flag = True
        self.zone_layers = {}
        self.zone_surface = {}
        self.layer_conductivity = {}
        self.layer_thickness = {}
        self.layer_names = None
        self.zone_names = None
        if default:
            self._layers_name(LAYER_NAMES)
            self._zones_name(ZONE_NAMES)
            self.zone_layers = ZONE_LAYERS
            self.zone_surface = ZONE_SURFACE
            self.layer_conductivity = LAYER_CONDUCTIVITY
            self.layer_thickness = LAYER_THICKNESS
            self.flag = False
            self._makearrays()

    def __getattr__(self, item):
        check_for_new_function_name(item)
        # if the return value is not callable, we get TypeError:

    def _layers_name(self, name_list):
        """
        Sets heat insulation layers and default value for each material in name_list.
        E.g. 

        Args:
            name_list (list): Contains names of heat insulation materials. 
        """
        self.layer_names = name_list
        for nm in name_list:
            self.layer_conductivity[nm] = None
            self.layer_thickness[nm] = None

    def _zones_name(self, name_list):
        """
        Sets the vehicle zones.
        E.g. 'lateral_windows', 'windshields' .. 
        Args:
            name_list (list): Contains names of zones.
        """
        self.zone_names = name_list
        for nm in name_list:
            self.zone_layers[nm] = None
            self.zone_surface[nm] = None

    def compile(self):
        """
        Set all zones from self.zone_names to self.zone_layers. 
        """
        for zone in self.zone_names:
            layer = {}
            for lyr in self.layer_names:
                layer[lyr] = None
            self.zone_layers[zone] = layer
        self._check()

    def show(self):
        """
        Prints Heat Insulation attributes.
        """
        logger.info("zone_layers:")
        logger.info(self.zone_layers)
        logger.info("zone_surface:")
        logger.info(self.zone_surface)
        logger.info("layer_conductivity:")
        logger.info(self.layer_conductivity)
        logger.info("layer_thickness:")
        logger.info(self.layer_thickness)
        self._check()

    def _check(self):
        """
        Checks for None values in self.zone_names and self.layer_names.
        """
        flag = False
        for zone in self.zone_names:
            for lyr in self.layer_names:
                if self.zone_layers[zone][lyr] is None:
                    logger.info(f"{self.__class__.__name__}.zone_layers['{zone}']['{lyr}'] = None")
                    flag = True

        for zone in self.zone_names:
            if self.zone_surface[zone] is None:
                logger.info(f"{self.__class__.__name__}.zone_surface['{zone}'] = None")
                flag = True

        for lyr in self.layer_names:
            if self.layer_conductivity[lyr] is None:
                logger.info(f"{self.__class__.__name__}.layer_conductivity['{lyr}'] = None")
                flag = True

        for lyr in self.layer_names:
            if self.layer_thickness[lyr] is None:
                logger.info(f"{self.__class__.__name__}.layer_thickness['{lyr}'] = None")
                flag = True
        if flag:
            self.flag = True
            logger.info('There are still "None" fields. All fields must contain values.')
        else:
            self.flag = False

    def _makearrays(self):
        """
        Create np.array for some object attributes. 
        Attributes: zone_layers, zone_surface, layer_conductivity, layer_thickness
        """
        if not self.flag:
            z_l = []
            for zone in self.zone_layers.keys():
                z_l.append(list(self.zone_layers[zone].values()))
            self.zone_layers_ = np.array(z_l)
            self.zone_surface_ = np.array(list(self.zone_surface.values()))
            self.layer_conductivity_ = np.array(list(self.layer_conductivity.values()))
            self.layer_thickness_ = np.array(list(self.layer_thickness.values()))


class Consumption:
    
    def __init__(self, inpt, ev_model):
        copy_to_user_data_dir()
        self.profile = None
        self.kind = "consumption"
        self.input = inpt
        self.vehicle = ev_model
        self._ev_par_test()
        self.brand = "_".join(self.vehicle.parameters["Brand"].split(" "))
        self.model = "_".join(self.vehicle.parameters["EV Model"].split(" "))
        self.year = str(int(self.vehicle.parameters["Model year"]))
        self.name = (
                self.input
                + "_"
                + self.brand
                + "_"
                + self.model
                + "_"
                + self.year
                + "_"
                + uuid.uuid4().hex[0:5]
        )
        self.COP = {
            "heating": self.vehicle.parameters["hvac_cop_heating"],
            "cooling": self.vehicle.parameters["hvac_cop_cooling"],
        }
        self.TARGET_TEMP = TARGET_TEMP

    def __getattr__(self, item):
        check_for_new_function_name(item)
        # if the return value is not callable, we get TypeError:

    def _ev_par_test(self):
        """
        Checks whether all required parameters have been saved in the object. 

        Raises:
            Exception: Raised if parameter is missing. 
        """
        param_names = VEHICLE_NEEDED_PARAMETERS
        param_missing = []
        for wanted in param_names:
            flag = False
            for parameter in self.vehicle.parameters:
                if wanted == parameter:
                    if not isinstance(self.vehicle.parameters[parameter], (int, float)):
                        flag = False
                        break
                    flag = True
                    break
            if not flag:
                param_missing.append(wanted)
        if len(param_missing) == 0:
            pass
        else:
            logger.info("Missing parameters in vehicle object:")
            for name in param_missing:
                logger.info(name)
            raise Exception('Parameters missing, add these parameters to the ev_model object < '
                            'ev_model.add({"parameter_name":value}) >')

    def _cop_and_target_temp(self, T_out):
        """
        # TODO DOCSTRING

        Args:
            T_out ([type]): [description]

        Returns:
            [type]: [description]
        """
        if T_out < self.TARGET_TEMP["heating"]:
            T_targ = self.TARGET_TEMP["heating"]
            cop = self.COP["heating"]
            flag = 1
        elif T_out > self.TARGET_TEMP["cooling"]:
            T_targ = self.TARGET_TEMP["cooling"]
            cop = self.COP["cooling"]
            flag = -1
        else:
            T_targ = None
            cop = 1
            flag = 0
        return T_targ, cop, flag

    def load_setting_mobility(self, DataBase):
        """
        Load certain attributes of the object with data from the transferred database.

        Attributes:

        - self.df
        - self.t
        - self.totalrows
        - self.hours
        - self.freq
        - self.refdate
        - self.energy_consumption
        - self.states

        Args:
            DataBase (DataBase()): Database from which the data is to be loaded. 

        Raises:
            ValueError: A driving profile can not be found in the database.
        """
        if DataBase.db[self.input]:
            if DataBase.db[self.input]["kind"] == "driving":
                self.profile = DataBase.db[self.input]["profile"].copy()
                self.t = DataBase.db[self.input]["t"]
                self.totalrows = DataBase.db[self.input]["totalrows"]
                self.hours = DataBase.db[self.input]["hours"]
                self.freq = TIME_FREQ[self.t]["f"]
                self.refdate = DataBase.db[self.input]["refdate"]
                self.states = DataBase.db[self.input]["states"]
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

    def run(
            self,
            heat_insulation,
            weather,
            passenger_mass=75,
            passenger_sensible_heat=70,
            passenger_nr=1.5,
            air_cabin_heat_transfer_coef=10,
            air_flow=0.01,
            driving_cycle_type="WLTC",
            road_type=0,
            wind_speed=0,
            road_slope=0,

    ):
        """
        #TODO Docstring

        Args:
            heat_insulation (object): [description]
            weather_country (str, optional: [description]. Defaults to "DE".
            weather_year (int, optional): [description]. Defaults to 2016.
            passenger_mass (int, optional): Passenger mass in kg. Defaults to 75.
            passenger_sensible_heat (int, optional): Passenger sensible heat in W. Defaults to 70.
            air_cabin_heat_transfer_coef (int, optional): Coefficient in W/(m2K). Defaults to 10.
            air_flow (float, optional): Ranges from  0.02 (high ventilation) to 0.001 (minimum ventilation) in me/s.
            Defaults to 0.01.
            driving_cycle_type (str, optional): [desc]. Defaults to "WLTC".
            road_type (int, optional): See function rolling_resistance_coef(method='M1') if an integer then all trips
            have the same value, if list must fit the length of the profile. Defaults to 0.
            wind_speed (int, optional):  m/s if an integer then all trips have the same value, if list must fit the
            length of the profile. Defaults to 0.
            road_slope (int, optional):  Radians if an integer then all trips have the same value, if list must fit the
            length of the profile. Defaults to 0.

        Raises:
            Exception: [description]
        """

        self.heat_insulation = heat_insulation
        self.passenger_mass = passenger_mass
        self.passenger_sensible_heat = passenger_sensible_heat
        self.passenger_nr = passenger_nr
        self.air_flow = air_flow
        self.driving_cycle_type = driving_cycle_type
        self.road_type = road_type
        self.wind_speed = wind_speed
        self.road_slope = road_slope
        self.transmission_eff = self.vehicle.parameters["transmission_eff"]
        self.battery_discharge_eff = self.vehicle.parameters[
            "battery_discharging_eff"
        ]
        self.battery_charge_eff = self.vehicle.parameters["battery_charging_eff"]
        self.air_cabin_heat_transfer_coef = air_cabin_heat_transfer_coef
        self.auxiliary_power = self.vehicle.parameters["auxiliary_power"] * 1000
        self.cabin_volume = self.vehicle.parameters["cabin_volume"]

        if heat_insulation.flag:
            raise Exception("heat_insulation object is not complete. Test the object with the method check() before "
                            "including it as argument")

        logger.info("New profile running: " + self.name)
        logger.debug("###################################################")
        logger.debug("===================================================")
        logger.debug("New profile running: " + self.name)
        logger.debug("===================================================")
        logger.debug("###################################################")

        self.profile = self.profile[["hr", "state", "distance", "trip_duration"]].copy()
        self.profile.loc[:, "datetime"] = pd.to_datetime(self.refdate) + (
                self.profile.hr * 60
        ).astype("timedelta64[m]")
        self.profile = self.profile.set_index("datetime")
        self.profile = self.profile.sort_index().reset_index()
        self.profile.loc[:, "speed km/h"] = (
                self.profile["distance"] / self.profile["trip_duration"] * 60
        )
        self.profile.loc[:, "wind_m/s"] = wind_speed
        self.profile.loc[:, "slope_rad"] = road_slope
        self.profile.loc[:, "road_type"] = road_type

        wt = Weather()
        D = wt.humidair_density
        # temp_arr = wt.temp(weather_country, weather_year)
        # pres_arr = wt.pressure(weather_country, weather_year)
        # dp_arr = wt.dewpoint(weather_country, weather_year)
        # hum_arr = wt.calc_rel_humidity(dp_arr, temp_arr)
        temp_arr = weather["temp_air"]
        pres_arr = weather["pressure"]
        dp_arr = weather["dew_point"]
        hum_arr = weather["relative_humidity"]
        
        r_ha = wt.humidair_density(temp_arr, pres_arr, h=hum_arr)

        self.profile = include_weather(
            self.profile, self.refdate, temp_arr, pres_arr, dp_arr, hum_arr, r_ha
        )
        self.η = MGefficiency()
        self.Trips = Trips()
        dc = DrivingCycle()
        dc.load_data()
        total = len(self.profile[self.profile["state"] == "driving"])
        current = 1

        for i, row in self.profile.iterrows():
            if row["state"] == "driving":
                # consumption_progress_bar(current, total)
                current += 1
                trip = Trip(self.Trips)
                trip.driving_cycle_type = driving_cycle_type
                trip.add_distance_duration(
                    distance={"value": row["distance"], "unit": "km"},
                    duration={"value": row["trip_duration"], "unit": "min"},
                )
                dc.driving_cycle(trip, self.vehicle, full_driving_cycle=False)
                v = trip.speed["value"]  # m/s
                acc = trip.acceleration["value"]  # m/s2
                targ_temp, cop, ret = self._cop_and_target_temp(row["temp_degC"])
                frontal_area = self.vehicle.parameters["front_area"]
                P_max = (
                        self.vehicle.parameters["power"] * 1000
                )  # kW to W
                f_d = self.vehicle.parameters["drag_coeff"]
                f_r = rolling_resistance_coeff(
                    method="M1",
                    temp=row["temp_degC"],
                    v=v * 3.6,
                    road_type=row["road_type"],
                )
                # f_r = rolling_resistance_coeff(method='M2', v=v*3.6, tire_type=0, road_type=4)
                m_i = self.vehicle.parameters["inertial_mass"]
                m_c = self.vehicle.parameters["curb_weight"]
                m_v = vehicle_mass(m_c, passenger_mass * passenger_nr)
                P_rol = prollingresistance(f_r, m_v, GRAVITY, v)
                P_air = pairdrag(
                    row["air_density_kg/m3"], frontal_area, f_d, v, row["wind_m/s"]
                )  # last arg is wind speed
                P_g = p_gravity(
                    m_v, GRAVITY, v, row["slope_rad"]
                )  # last arg is road slope
                P_ine = pinertia(m_i, m_v, acc, v)
                P_wheel = p_wheel(P_rol, P_air, P_g, P_ine)
                P_m_o = p_motorout(P_wheel, self.transmission_eff)
                n_rb = EFFICIENCYregenerative_braking(acc)
                P_gen_in = p_generatorin(P_wheel, self.transmission_eff, n_rb)
                Load_p_m = P_m_o / P_max
                Load_p_g = P_gen_in / P_max
                n_mot = self.η.get_efficiency(Load_p_m, 1)
                n_gen = self.η.get_efficiency(Load_p_g, -1)
                P_m_in = p_motorin(P_m_o, n_mot)
                P_g_out = p_generatorout(P_gen_in, n_gen)
                P_aux = np.array([self.auxiliary_power] * len(v))
                Q_hvac, Tcabin = qhvac(
                    D,
                    row["temp_degC"],
                    targ_temp,
                    self.cabin_volume,
                    air_flow,
                    heat_insulation.zone_layers_,
                    heat_insulation.zone_surface_,
                    heat_insulation.layer_conductivity_,
                    heat_insulation.layer_thickness_,
                    v,
                    Q_sensible=passenger_sensible_heat,
                    persons=passenger_nr,
                    air_cabin_heat_transfer_coef=air_cabin_heat_transfer_coef,
                )
                P_hvac = np.abs(Q_hvac[:, 0]) / cop
                P_gen_bat_charg = P_g_out * self.battery_charge_eff * -1
                P_bat = (P_m_in + P_aux + P_hvac) / self.battery_discharge_eff
                # section to calculate consumption
                P_all = P_m_in + P_aux + P_hvac + P_g_out
                P_all_negative = P_all.copy()
                P_all_negative[P_all_negative > 0.0] = 0.0
                P_all_positive = P_all.copy()
                P_all_positive[P_all_positive < 0.0] = 0.0
                P_bat_chg = P_all_negative * self.battery_charge_eff
                P_bat_dischg = P_all_positive / self.battery_discharge_eff
                P_bat_actual = np.add(P_bat_dischg, P_bat_chg)  # W
                consumption = P_bat_actual.sum() / 1000 / 3600  # kWh
                rate = consumption / v.sum() * 100000  # kWh/100 km

                # Add variables to trip object: International units (power in W)
                trip.results["targ_temp"] = targ_temp
                trip.results["cop"] = cop
                trip.results["ret"] = ret
                trip.results["frontal_area"] = frontal_area
                trip.results["P_max"] = P_max
                trip.results["Drag_coeff"] = f_d
                trip.results["roll_res_coeff"] = f_r
                trip.results["m_i"] = m_i
                trip.results["m_c"] = m_c
                trip.results["m_v"] = m_v
                trip.results["P_rol"] = P_rol
                trip.results["P_air"] = P_air
                trip.results["P_g"] = P_g
                trip.results["P_ine"] = P_ine
                trip.results["P_wheel"] = P_wheel
                trip.results["P_gen_in"] = P_gen_in
                trip.results["Load_p_m"] = Load_p_m
                trip.results["Load_p_g"] = Load_p_g
                trip.results["n_mot"] = n_mot
                trip.results["n_gen"] = n_gen
                trip.results["P_m_in"] = P_m_in
                trip.results["P_g_out"] = P_g_out
                trip.results["P_aux"] = P_aux
                trip.results["Q_hvac"] = Q_hvac
                trip.results["Tcabin"] = Tcabin
                trip.results["Tout"] = row["temp_degC"]
                trip.results["P_hvac"] = P_hvac

                trip.results["P_gen_bat_charg"] = P_gen_bat_charg
                trip.results["P_bat"] = P_bat  # only all positive loads
                trip.results[
                    "P_bat_actual"
                ] = P_bat_actual  # positive load after generation subtraction and negative load (generation) after
                # positive loads subtraction

                # Variable for the balance

                P_wheel_pos = P_wheel[P_wheel > 0].sum()  # Ws
                P_wheel_neg = P_wheel[P_wheel < 0].sum() * -1  # Ws
                P_m_o_t = P_m_o.sum()  # Ws
                P_gen_in_t = P_gen_in.sum() * -1  # Ws
                P_m_in_t = P_m_in.sum()  # Ws
                P_g_out_t = P_g_out.sum() * -1  # Ws
                P_aux_t = P_aux.sum()  # Ws
                P_hvac_t = P_hvac.sum()  # Ws
                heat_source = np.abs(Q_hvac[:, 0]).sum() - P_hvac_t  # Ws
                P_gen_bat_charg_t = P_gen_bat_charg.sum()  # Ws
                P_gen_bat_dischg_t = (
                        P_gen_bat_charg_t * self.battery_discharge_eff
                )  # Ws
                P_bat_t = P_bat.sum()  # Ws

                trip.consumption[
                    "value"
                ] = consumption  # the only option this value be to small or negative is the ev goes downhill most of
                # the trip
                trip.consumption["unit"] = "kWh"
                trip.rate["value"] = rate
                trip.rate["unit"] = "kWh/100 km"

                loss_gen = P_gen_in_t - P_g_out_t
                loss_trans_m = P_m_o_t - P_wheel_pos
                loss_trans_g = P_wheel_neg - P_gen_in_t
                loss_motor = P_m_in_t - P_m_o_t
                loss_gen_bat_charg = P_gen_bat_charg_t * (1 - self.battery_charge_eff)
                loss_gen_bat_dischg = P_gen_bat_charg_t * (
                        1 - self.battery_discharge_eff
                )
                loss_bat = P_bat_t * (1 - self.battery_discharge_eff)

                if ret == 1:
                    cooling = 0
                    heating = P_hvac_t + heat_source
                elif ret == -1:
                    cooling = P_hvac_t + heat_source
                    heating = 0
                elif ret == 0:
                    cooling = 0
                    heating = 0

                # data for sankey diagram
                j = np.zeros((v.shape[0], 7))
                j[:, 0] = P_rol
                j[:, 1] = P_air
                j[:, 2] = P_g
                j[:, 3] = P_ine
                j[:, 4] = np.sum(j[:, 0:4], axis=1)
                j[:, 5] = j[:, 4]
                j[np.where(j[:, 5] > 0.0), 5] = 0
                j[:, 6] = j[:, 4]
                j[np.where(j[:, 6] < 0.0), 6] = 0

                ig = np.zeros((v.shape[0], 4))
                ig[np.where(j[:, 5] < 0.0), 0:4] = j[np.where(j[:, 5] < 0.0), 0:4]
                ig[np.where(ig[:, 0] > 0.0), 0] = 0
                ig[np.where(ig[:, 1] > 0.0), 1] = 0
                ig[np.where(ig[:, 2] > 0.0), 2] = 0
                ig[np.where(ig[:, 3] > 0.0), 3] = 0
                xg = np.true_divide(
                    ig,
                    ig.sum(axis=1, keepdims=True),
                    out=np.zeros_like(ig),
                    where=ig.sum(axis=1, keepdims=True) != 0,
                )
                yg = (xg.T * j[:, 5]).T * -1
                zg = yg.sum(axis=0)

                ip = np.zeros((v.shape[0], 4))
                ip[np.where(j[:, 6] > 0.0), 0:4] = j[np.where(j[:, 6] > 0.0), 0:4]
                ip[np.where(ip[:, 0] < 0.0), 0] = 0
                ip[np.where(ip[:, 1] < 0.0), 1] = 0
                ip[np.where(ip[:, 2] < 0.0), 2] = 0
                ip[np.where(ip[:, 3] < 0.0), 3] = 0
                xp = np.true_divide(
                    ip,
                    ip.sum(axis=1, keepdims=True),
                    out=np.zeros_like(ip),
                    where=ip.sum(axis=1, keepdims=True) != 0,
                )
                yp = (xp.T * j[:, 6]).T
                zp = yp.sum(axis=0)
                gra_neg = zg[2]
                acc_neg = zg[3]

                rol_pos = zp[0]
                air_pos = zp[1]
                gra_pos = zp[2]
                acc_pos = zp[3]

                self.profile.loc[i, "consumption kWh/100 km"] = rate
                self.profile.loc[i, "consumption kWh"] = consumption
                self.profile.loc[i, "battery discharge kWh"] = P_bat_t / 3600 / 1000
                self.profile.loc[i, "regeneration kWh"] = (
                        P_gen_bat_dischg_t / 3600 / 1000
                )
                self.profile.loc[i, "auxiliary kWh"] = P_aux_t / 3600 / 1000
                self.profile.loc[i, "hvac kWh"] = P_hvac_t / 3600 / 1000
                self.profile.loc[i, "motor in kWh"] = P_m_in_t / 3600 / 1000
                self.profile.loc[i, "transmission in kWh"] = P_m_o_t / 3600 / 1000
                self.profile.loc[i, "wheel kWh"] = P_wheel_pos / 3600 / 1000
                self.profile.loc[i, "rolling res kWh"] = rol_pos / 3600 / 1000
                self.profile.loc[i, "air res kWh"] = air_pos / 3600 / 1000
                self.profile.loc[i, "gravity kWh"] = gra_pos / 3600 / 1000
                self.profile.loc[i, "acceleration kWh"] = acc_pos / 3600 / 1000
                self.profile.loc[i, "trip code"] = trip.code

                stv = [
                    ["Heat source", "HVAC", heat_source / 3600 / 1000],
                    ["Potential energy", "Gravity force", gra_neg / 3600 / 1000],
                    [
                        "Battery",
                        "Discharge",
                        (P_bat_t - P_gen_bat_charg_t) / 3600 / 1000,
                    ],
                    [
                        "Discharge",
                        "Losses",
                        (loss_bat + loss_gen_bat_dischg) / 3600 / 1000,
                    ],
                    ["Discharge", "HVAC", P_hvac_t / 3600 / 1000],
                    ["Discharge", "Auxiliary", P_aux_t / 3600 / 1000],
                    ["Discharge", "Motor", P_m_in_t / 3600 / 1000],
                    [
                        "reg_braking",
                        "Discharge",
                        P_gen_bat_dischg_t / 3600 / 1000,
                    ],
                    [
                        "reg_braking",
                        "Losses",
                        loss_gen_bat_charg / 3600 / 1000,
                    ],
                    ["HVAC", "Cooling", cooling / 3600 / 1000],
                    ["HVAC", "Heating", heating / 3600 / 1000],
                    ["Motor", "Transmission of traction", P_m_o_t / 3600 / 1000],
                    ["Motor", "Losses", loss_motor / 3600 / 1000],
                    ["Transmission of traction", "Wheel", P_wheel_pos / 3600 / 1000],
                    ["Transmission of traction", "Losses", loss_trans_m / 3600 / 1000],
                    ["Wheel", "Rolling resistance", rol_pos / 3600 / 1000],
                    ["Wheel", "Air resistance", air_pos / 3600 / 1000],
                    ["Wheel", "Gravity force", gra_pos / 3600 / 1000],
                    ["Wheel", "Acceleration force", acc_pos / 3600 / 1000],
                    ["Rolling resistance", "Losses", rol_pos / 3600 / 1000],
                    ["Air resistance", "Losses", air_pos / 3600 / 1000],
                    ["Gravity force", "Kinetic energy", gra_neg / 3600 / 1000],
                    ["Gravity force", "Losses", (gra_pos - gra_neg) / 3600 / 1000],
                    ["Acceleration force", "Kinetic energy", acc_neg / 3600 / 1000],
                    ["Acceleration force", "Losses", (acc_pos - acc_neg) / 3600 / 1000],
                    [
                        "Kinetic energy",
                        "Transmission of regenerative",
                        (acc_neg + gra_neg) / 3600 / 1000,
                    ],
                    [
                        "Transmission of regenerative",
                        "Generator",
                        P_gen_in_t / 3600 / 1000,
                    ],
                    [
                        "Transmission of regenerative",
                        "Losses",
                        loss_trans_g / 3600 / 1000,
                    ],
                    ["Generator", "reg_braking", P_g_out_t / 3600 / 1000],
                    ["Generator", "Losses", loss_gen / 3600 / 1000],
                    ["Cooling", "Losses", cooling / 3600 / 1000],
                    ["Heating", "Losses", heating / 3600 / 1000],
                    ["Auxiliary", "Losses", P_aux_t / 3600 / 1000],
                ]

                link_label = []
                for lk in stv:
                    llk = [lk[0], lk[1], str(round(lk[2], 1))]
                    link_label.append("->".join(llk))

                sort = np.array(stv, dtype=object)
                s = sort.T[0].tolist()
                t = sort.T[1].tolist()
                v = sort.T[2]

                balance = {}
                balance["label"] = [
                    "Heat source",
                    "Potential energy",
                    "Battery",
                    "Discharge",
                    "reg_braking",
                    "HVAC",
                    "Motor",
                    "Generator",
                    "Transmission of traction",
                    "Wheel",
                    "Kinetic energy",
                    "Cooling",
                    "Heating",
                    "Auxiliary",
                    "Gravity force",
                    "Acceleration force",
                    "Rolling resistance",
                    "Air resistance",
                    "Losses",
                    "Transmission of regenerative",
                ]
                balance["source"] = [balance["label"].index(i) for i in s]
                balance["target"] = [balance["label"].index(i) for i in t]
                balance["value"] = v
                balance["link_label"] = link_label
                balance["data"] = stv
                trip.balance = balance
        print("")
        self._fill_rows()

    def _fill_rows(self):
        """
        Sets data for many attributes and is executed in self.run.
        """
        repeats = [
            "hr",
            "state",
            "distance",
            "consumption",
            # "trip code",
        ]
        fixed = ["consumption kWh","trip_duration"]
        variable_consumption = ["instant consumption in W","average power in W"]
        calcu = ["dayhrs"]
        mixed = repeats + fixed + variable_consumption

        df = self.profile.copy(deep=True)
        df['consumption'] = df["consumption kWh"]
        df['instant consumption in W'] = df["consumption kWh"]
        df['average power in W'] = df["consumption kWh"]
        df = df[mixed]

        self.timeseries = pd.DataFrame(columns=mixed)
        self.timeseries.loc[:, "hh"] = np.arange(0, self.hours, self.t)

        # Old version, which does not work for 1s-based profiles:
        # idxx_original = self.timeseries[self.timeseries["hh"].isin(df["hr"].tolist())].index.tolist()

        trip_data = self.Trips.trips

        # Start New version, which works for 1s-based profiles:
        temp_timeseries = [round(num*3600) for num in self.timeseries["hh"]]
        temp_df = [round(num*3600) for num in df["hr"]]
        temp_intersection_list = list(set(temp_timeseries).intersection(temp_df))

        idxx = []
        for i in temp_intersection_list:
            idxx.append(temp_timeseries.index(i))
        idxx = np.sort(idxx).tolist()
        # End new version

        for r in mixed:
            vall = df[r].values.tolist()
            self.timeseries.loc[idxx, r] = vall

        self.timeseries.loc[self.totalrows - 1, "state"] = df["state"].iloc[-1]
        self.timeseries.loc[self.totalrows - 1, "hr"] = self.timeseries["hh"][self.totalrows - 1]
        rp = self.timeseries[::-1].reset_index(drop=True)
        rp.loc[:, repeats] = rp[repeats].fillna(method="ffill")
        rp.loc[:, fixed] = rp[fixed].fillna(0)
        self.timeseries = rp[::-1].reset_index(drop=True)
        for cal in calcu:
            self.timeseries.loc[:, cal] = self.timeseries["hh"].apply(lambda x: x % 24)

        # Find idxs of starts of trips:
        temp_idx = self.timeseries.loc[idxx, "state"] == "driving"
        temp_idx = temp_idx.shift(-1) # all idx before driving start trip
        idx_trip_start = (temp_idx[temp_idx == 1].index +1).tolist()
        # TODO: Get the distance at every time step, not only at the end of the trip. trip.speed["value"]
        # Add real consumption:
        self.timeseries.loc[:, "instant consumption in W"] = 0
        self.timeseries.loc[:, "average power in W"] = 0
        # Only if resolution is 1s:
        if self.freq == '1s':
            for trip in trip_data:
                trip_start = idx_trip_start[trip.code]
                trip_length = trip.time["value"]
                trip_values = trip.results["P_bat_actual"]
                self.timeseries.loc[trip_start:trip_start+trip_length-1, "instant consumption in W"] = trip_values
            self.timeseries.loc[:, "average power in W"] = self.timeseries.loc[:, "instant consumption in W"]
        else:
            for trip in trip_data:
                trip_start = idx_trip_start[trip.code]
                trip_length = int(trip.time["value"])
                step_secs = int(self.t*3600)
                if trip_length < step_secs:
                    trip_values = trip.results["P_bat_actual"].sum()/step_secs
                    self.timeseries.loc[trip_start, "average power in W"] = trip_values
                else:
                    j = 0
                    for i in range(0, trip_length, step_secs):
                        trip_section = trip_start + j
                        last_second = i+step_secs
                        if last_second > trip_length:
                            last_second = trip_length
                            trip_values = trip.results["P_bat_actual"][i:last_second].sum()/step_secs
                            self.timeseries.loc[trip_section, "average power in W"] = trip_values
                        else:
                            trip_values = trip.results["P_bat_actual"][i:last_second].mean()
                            self.timeseries.loc[trip_section, "average power in W"] = trip_values
                        j+=1

        self.timeseries = _add_column_datetime(
            self.timeseries, self.totalrows, self.refdate, self.t
        )
        self.timeseries.loc[:, "count"] = self.timeseries.groupby(["hr", "state"])["state"].transform("count")

        self.timeseries['consumption'] = (self.timeseries['consumption'] * self.timeseries['distance']).fillna(0) / (
            self.timeseries['distance'].replace(0,
                                                1))  # this is an artifact to make zero the consumption when distance is zero
        self.timeseries.loc[:, "consumption"] = (
                self.timeseries.loc[:, "consumption"] / self.timeseries.loc[:, "count"]
        )
        self.timeseries.loc[:, "distance"] = (
                self.timeseries.loc[:, "distance"] / self.timeseries.loc[:, "count"]
        )
        self.timeseries = self.timeseries[["hh", "state", "distance", "consumption", "instant consumption in W","average power in W"]]


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
        logger.debug("=== profile saved === : " + filepath)
        logger.info(" ")
        logger.info("See Log files")
        for handler in logger.handlers:
            if handler.__class__.__name__ == 'FileHandler':
                logger.info(handler.baseFilename)
        try:
            display_all()
        except:
            pass
