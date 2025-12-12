"""
Fuctions used for Consumption class
"""


import pandas as pd
import numpy as np
import numba
from src.external.emobpy.constants import AIR_SPECIFIC_HEAT

T_RNG = np.array(list(AIR_SPECIFIC_HEAT.keys()))
CP_RNG = np.array(list(AIR_SPECIFIC_HEAT.values()))

@numba.jit(nopython=True)
def inertial_mass(curb_weight, gear_ratio):
    """
    Calculates and returns inertial mass.

    Args:
        curb_weight (float): Curb weight of the car.
        gear_ratio (float): Gear ratio of the car.

    Returns:
        float: Inertial mass of the car.
    """
    return curb_weight * (0.04 + 0.0025 * gear_ratio ** 2)

def include_weather(pf, refdate, temp_arr, pres_arr, dp_arr, H, r_ha):
    """
    Adds weather data to given DataFrame.

    Args:
        pf (pd.DataFrame): DataFrame where weather data should be added.
        refdate (str): E.g. '01/01/2020'.
        temp_arr (ndarray): Temperature in degree Celsius.
        pres_arr (ndarray): Pressure in mbar.
        dp_arr (ndarray): Dewpoint data in degree Celsius.
        H (ndarray): Humidity data.
        r_ha (ndarray): Air density in kg/m3.

    Returns:
        pd.DataFrame: [description]
    """
    year = pd.to_datetime(refdate).year
    start_date = pd.to_datetime(year, format="%Y")
    drange = pd.date_range(start_date, periods=len(r_ha) * 2, freq="H")
    df = pd.DataFrame(
        data={
            "temp_degC": np.concatenate([temp_arr, temp_arr]),
            "pressure_mbar": np.concatenate([pres_arr, pres_arr]),
            "dewpoint_degC": np.concatenate([dp_arr, dp_arr]),
            "humidity": np.concatenate([H, H]),
            "air_density_kg/m3": np.concatenate([r_ha, r_ha]),
        },
        index=drange,
    )
    df = df.rename_axis("datetime")
    df = df.sort_index().reset_index()
    df["weather_time"] = df["datetime"]
    return pd.merge_asof(
        pf, df, on="datetime", tolerance=pd.Timedelta("3600s"), direction="nearest"
    ).set_index("hr", drop=False)

def rolling_resistance_coeff(method='M1', **kwargs):
    """
    Returns calculated rolling resistance coeff depending on method.

    M1 depends on v:velocity (km/h), temp: degC and road_type: int  -> Wang et al.
    M2 depends on v:velocity (km/h), tire_type: int, road_type: int -> Rahka et al.

    M1 options:
        v: km/h
        temp: degC
        road_type
            0: ordinary car tires on concrete, new asphalt, cobbles small new, coeff: 0.01 - 0.015
            1: car tires on gravel - rolled new, on tar or asphalt, coeff: 0.02
            2: car tires on cobbles  - large worn, coeff: 0.03
            3: car tire on solid sand, gravel loose worn, soil medium hard, coeff: 0.04 - 0.08
            4: car tire on loose sand, coeff: 0.2 - 0.4
    M2 options:
        v: km/h
        tire_type
            0: Radial,    c2:0.0328, c3: 4.575
            1: Bias ply,  c2:0.0438, c3: 6.100
        road_type
            Concrete: excellent 0: 1.00, good 1: 1.50, poor 2: 2.00
            Asphalt: good 3: 1.25, fair 4: 1.75, poor 5: 2.25
            Macadam: good 6: 1.50, fair 7: 2.25, poor 8: 3.75
            Cobbles: ordinary 9: 5.50, poor 10: 8.50
            Snow: 2 inch 11: 2.50, 4 inch 12: 3.75
            Dirt: Smooth 13: 2.50, sandy 14: 3.75
            Sand not implemented range 6.00 - 30.00

    Args:
        method (str, optional): [description]. Defaults to 'M1'.

    Raises:
        Exception: Raised if Method is not M1 or M2.

    Returns:
        [type]: [description]
    """
    if method == 'M1':
        return rolling_resistance_coeff_M1(**kwargs)
    elif method == 'M2':
        return rolling_resistance_coeff_M2(**kwargs)
    else:
        raise Exception('Method must be M1 or M2')


@numba.jit(nopython=True)
def rolling_resistance_coeff_M1(temp, v, road_type=0):
    """
    Returns calculated rolling resistance coeff for M1.

    Args:
        temp (float): Temperature ein degree celsius.
        v (float): Speed in km/h.
        road_type (int, optional): 
                0: ordinary car tires on concrete, new asphalt, cobbles small new, coeff: 0.01 - 0.015 (Default)
                1: car tires on gravel - rolled new, on tar or asphalt, coeff: 0.02
                2: car tires on cobbles  - large worn, coeff: 0.03
                3: car tire on solid sand, gravel loose worn, soil medium hard, coeff: 0.04 - 0.08
                4: car tire on loose sand, coeff: 0.2 - 0.4
            reference: Wang, J.; Besselink, I.; Nijmeijer, H. Electric Vehicle Energy Consumption Modelling and
            Prediction Based on Road Information.
            World Electr. Veh. J. 2015, 7, 447-458. https://doi.org/10.3390/wevj7030447
    
    Returns:
        int: Rolling resistance coefficient
    """
    factor = [1, 1.5, 2.2, 4, 20]
    return (1.9e-6 * temp ** 2 - 2.1e-4 * temp + 0.013 +
            5.4e-5 * v) * factor[road_type]


@numba.jit(nopython=True)
def rolling_resistance_coeff_M2(v, tire_type=0, road_type=4):
    """
    Returns calculated rolling resistance coeff for M2.

    Args:
        v (float): Speed in km/h.
        tire_type (int, optional): 
                0: Radial,    c2:0.0328, c3: 4.575
                1: Bias ply,  c2:0.0438, c3: 6.100 (Default)
        road_type (int, optional): [description]. Defaults to 4.
                Concrete: excellent 0: 1.00, good 1: 1.50, poor 2: 2.00
                Asphalt: good 3: 1.25, fair 4: 1.75, poor 5: 2.25
                Macadam: good 6: 1.50, fair 7: 2.25, poor 8: 3.75
                Cobbles: ordinary 9: 5.50, poor 10: 8.50
                Snow: 2 inch 11: 2.50, 4 inch 12: 3.75
                Dirt: Smooth 13: 2.50, sandy 14: 3.75
                Sand not implemented range 6.00 - 30.00
                reference: Rahka et al. 2001. Vehicle Dynamics Model for Predicting Maximum Truck Acceleration Levels.
                https://doi.org/10.1061/(ASCE)0733-947X(2001)127:5(418)

    Returns:
        int: Rolling resistance coefficient
    """

    road = {
        0: {'Cr': 1.0},
        1: {'Cr': 1.5},
        2: {'Cr': 2.0},
        3: {'Cr': 1.25},
        4: {'Cr': 1.75},
        5: {'Cr': 2.25},
        6: {'Cr': 1.5},
        7: {'Cr': 2.25},
        8: {'Cr': 3.75},
        9: {'Cr': 5.5},
        10: {'Cr': 8.5},
        11: {'Cr': 2.5},
        12: {'Cr': 3.75},
        13: {'Cr': 2.5},
        14: {'Cr': 3.75}
    }
    tire = {0: {'c2': 0.0328, 'c3': 4.575}, 1: {'c2': 0.0438, 'c3': 6.100}}

    Cr = road[road_type]['Cr']
    c2 = tire[tire_type]['c2']
    c3 = tire[tire_type]['c3']

    return Cr * (c2 * v + c3) / 1000


@numba.jit(nopython=True)
def vehicle_mass(curb_weight, passengers_weight):
    """
    Calculates and returns vehicle mass.

    Args:
        curb_weight (float): Curb weight of the vehicle.
        passengers_weight (float): Passengers weight.

    Returns:
        float: Vehicle mass.
    """
    return curb_weight + passengers_weight


@numba.jit(nopython=True)
def prollingresistance(rolling_resistance_coeff,
                       vehicle_mass,
                       g,
                       v,
                       slop_angle=0):
    """
    Calculates and returns polling resistance.

    #TODO DOCSTRING
    Args:
        rolling_resistance_coeff ([type]): [description]
        vehicle_mass ([type]): [description]
        g ([type]): [description]
        v ([type]): [description]
        slop_angle (int, optional): [description]. Defaults to 0.

    Returns:
        float: Polling resistance.
    """
    return rolling_resistance_coeff * vehicle_mass * g * np.cos(
        np.deg2rad(slop_angle)) * v


@numba.jit(nopython=True)
def pairdrag(air_density, frontal_area, drag_coeff, v, wind_speed=0):
    """
    #TODO DOCSTRING
    Reference: Wang, J.; Besselink, I.; Nijmeijer, H. Electric Vehicle Energy Consumption Modelling and Prediction
    Based on Road Information.
    World Electr. Veh. J. 2015, 7, 447-458. https://doi.org/10.3390/wevj7030447

    Args:
        air_density ([type]): [description]
        frontal_area ([type]): [description]
        drag_coeff ([type]): [description]
        v ([type]): [description]
        wind_speed (int, optional): Wind speed in direction of the vehicle.. Defaults to 0.

    Returns:
        float: [description]
    """
    return 1 / 2 * air_density * frontal_area * drag_coeff * (
            v - wind_speed) ** 2 * v


@numba.jit(nopython=True)
def p_gravity(vehicle_mass, g, v, slop_angle=0):
    """
            #TODO DOCSTRING

    Args:
        vehicle_mass ([type]): [description]
        g ([type]): [description]
        v ([type]): [description]
        slop_angle (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    return vehicle_mass * g * np.sin(np.deg2rad(slop_angle)) * v


@numba.jit(nopython=True)
def pinertia(inertial_mass, vehicle_mass, acceleration, v):
    """
            #TODO DOCSTRING

    Args:
        inertial_mass ([type]): [description]
        vehicle_mass ([type]): [description]
        acceleration ([type]): [description]
        v ([type]): [description]

    Returns:
        [type]: [description]
    """
    return (vehicle_mass + inertial_mass) * acceleration * v


@numba.jit(nopython=True)
def p_wheel(p_rollingresistance, p_airdrag, p_gravity, p_inertia):
    """        #TODO DOCSTRING

    Args:
        p_rollingresistance ([type]): [description]
        p_airdrag ([type]): [description]
        p_gravity ([type]): [description]
        p_inertia ([type]): [description]

    Returns:
        [type]: [description]
    """
    return p_rollingresistance + p_airdrag + p_gravity + p_inertia


@numba.jit(nopython=True)
def p_motorout(p_wheel, transmission_eff):
    """        #TODO DOCSTRING

    Args:
        p_wheel ([type]): [description]
        transmission_eff ([type]): [description]

    Returns:
        [type]: [description]
    """
    only_positive = p_wheel.copy()
    only_positive[only_positive < 0.0] = 0.0
    result = only_positive / transmission_eff
    mask = np.isnan(result)
    result[mask] = 0
    return result


@numba.jit(nopython=True)
def p_generatorin(p_wheel, transmission_eff, regenerative_braking_eff):
    """        #TODO DOCSTRING

    Args:
        p_wheel ([type]): [description]
        transmission_eff ([type]): [description]
        regenerative_braking_eff ([type]): [description]

    Returns:
        [type]: [description]
    """
    only_negative = p_wheel.copy()
    only_negative[only_negative > 0.0] = 0.0
    result = only_negative * transmission_eff * regenerative_braking_eff
    mask = np.isnan(result)
    result[mask] = 0
    return result


@numba.jit(nopython=True)
def EFFICIENCYregenerative_braking(acceleration):
    """        #TODO DOCSTRING

    Args:
        acceleration ([type]): [description]

    Returns:
        [type]: [description]
    """
    neg_acceleration = acceleration.copy()
    neg_acceleration[neg_acceleration > 0.0] = 0.0
    result = (np.exp(0.0411 / np.abs(neg_acceleration))) ** (-1)
    mask = np.isnan(result)
    result[mask] = 0
    return result


@numba.jit(nopython=True)
def p_motorin(p_motor_out, motor_eff):
    """        #TODO DOCSTRING

    Args:
        p_motor_out ([type]): [description]
        motor_eff ([type]): [description]

    Returns:
        [type]: [description]
    """
    result = p_motor_out / motor_eff
    mask = np.isnan(result)
    result[mask] = 0
    return result


@numba.jit(nopython=True)
def p_generatorout(p_generator_in, generator_eff):
    """        #TODO DOCSTRING

    Args:
        p_generator_in ([type]): [description]
        generator_eff ([type]): [description]

    Returns:
        [type]: [description]
    """
    result = p_generator_in * generator_eff
    mask = np.isnan(result)
    result[mask] = 0
    return result


# Heat transfer
@numba.jit(nopython=True)
def q_person(q_sensible, persons=1):
    """        #TODO DOCSTRING

    Args:
        q_sensible ([type]): [description]
        persons (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    return q_sensible * persons


@numba.jit(nopython=True)
def q_ventilation(density_air, flow_air, Cp_air, temp_air):
    """        #TODO DOCSTRING

    Density_air: kg/m3, Flow_air: m3/s, Cp_air: J/(kg*K), Temp_air: degC
    Args:
        density_air ([type]): [description]
        flow_air ([type]): [description]
        Cp_air ([type]): [description]
        temp_air ([type]): [description]

    Returns:
        [type]: [description]
    """
    temp_kelvin = temp_air + 273.15
    return density_air * flow_air * Cp_air * temp_kelvin


@numba.jit(nopython=True)
def q_transfer(zone_layer,
               zone_area,
               layer_conductivity,
               layer_thickness,
               t_air_cabin,
               t_air_out,
               vehicle_speed,
               air_cabin_heat_transfer_coef=10):
    """        #TODO DOCSTRING

    Args:
        zone_layer ([type]): [description]
        zone_area ([type]): [description]
        layer_conductivity ([type]): [description]
        layer_thickness ([type]): [description]
        t_air_cabin ([type]): [description]
        t_air_out ([type]): [description]
        vehicle_speed ([type]): [description]
        air_cabin_heat_transfer_coef (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    t_air_cabin_K = t_air_cabin + 273.15
    t_air_out_K = t_air_out + 273.15
    R = resistances(zone_layer, zone_area, layer_conductivity, layer_thickness,
                    vehicle_speed, air_cabin_heat_transfer_coef)
    return (t_air_cabin_K - t_air_out_K) * R


@numba.jit(nopython=True)
def htc_air_out(vehicle_speed, limit=5):
    """        #TODO DOCSTRING

    Args:
        vehicle_speed ([type]): [description]
        limit (int, optional): [description]. Defaults to 5.

    Returns:
        [type]: [description]
    """
    h = 6.14 * np.power(vehicle_speed, 0.78)
    if vehicle_speed < limit:
        h = 6.14 * np.power(limit, 0.78)
    return h


@numba.jit(nopython=True)
def resistances(zone_layer, zone_area, layer_conductivity, layer_thickness,
                vehicle_speed, air_cabin_heat_transfer_coef):
    """[summary]        #TODO DOCSTRING

    Args:
        zone_layer ([type]): [description]
        zone_area ([type]): [description]
        layer_conductivity ([type]): [description]
        layer_thickness ([type]): [description]
        vehicle_speed ([type]): [description]
        air_cabin_heat_transfer_coef ([type]): [description]

    Returns:
        [type]: [description]
    """
    x_z = zone_layer * layer_thickness
    R_c = x_z / layer_conductivity
    h_i = air_cabin_heat_transfer_coef
    h_o = htc_air_out(vehicle_speed)
    S_rc = R_c.sum(axis=1)
    R_hz = 1 / h_i + S_rc + 1 / h_o
    R_z = zone_area / R_hz
    return R_z.sum()


# @numba.jit(nopython=True)
def qhvac(D,
          T_out,
          T_targ,
          cabin_volume,
          flow_air,
          zone_layer,
          zone_area,
          layer_conductivity,
          layer_thickness,
          vehicle_speed,
          Q_sensible=70,
          persons=1,
          P_out=1013.25,
          h_out=60,
          air_cabin_heat_transfer_coef=10):
    """
    #TODO DOCUMENTATION
    Q indexes 0: Qtotal, 1: Q_in_per, 2: Q_in_vent, 3: Q_out_vent, 4: Q_tr

    Args:
        D (method): [description]
        T_out (float): [description]
        T_targ (int): [description]
        cabin_volume (float): [description]
        flow_air (float): [description]
        zone_layer (ndarray): [description]
        zone_area (ndarray): [description]
        layer_conductivity (ndarray): [description]
        layer_thickness (ndarray): [description]
        vehicle_speed (ndarray): [description]
        Q_sensible (int, optional): [description]. Defaults to 70.
        persons (int, optional): [description]. Defaults to 1.
        P_out (float, optional): [description]. Defaults to 1013.25.
        h_out (int, optional): [description]. Defaults to 60.
        air_cabin_heat_transfer_coef (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """

    mass_flow_in = flow_air * D(T_out, P_out, h=h_out)

    T = np.zeros((vehicle_speed.shape[0],))
    Q = np.zeros((vehicle_speed.shape[0], 8))
    if T_targ is None:
        return Q, T
    t_diff = T_out - T_targ  # positive if cooling, negative if heating
    if t_diff > 0:
        plus = -0.05
        sign = -1  # cooling
    else:
        plus = 0.05
        sign = 1  # heating

    for tm in range(vehicle_speed.shape[0]):
        if tm == 0:
            t_1 = T_out
            t = T_out + plus
        else:
            t_1 = T[tm - 1]
            if sign == -1:
                if np.round(t, 2) > T_targ:
                    t += plus
                else:
                    t = T_targ
            else:
                if np.round(t, 2) < T_targ:
                    t += plus
                else:
                    t = T_targ

        Q_in_per = q_person(Q_sensible, persons)
        Q[tm][1] = Q_in_per
        Q_in_vent = q_ventilation(D(T_out, P_out, h=h_out), flow_air,
                                  cp(T_out), T_out)
        Q[tm][2] = Q_in_vent
        Q_out_vent = q_ventilation(D(t, P_out, h=h_out),
                                   mass_flow_in / D(t, P_out, h=h_out), cp(t),
                                   t)
        Q[tm][3] = Q_out_vent
        Q_tr = q_transfer(zone_layer, zone_area, layer_conductivity,
                          layer_thickness, t, T_out, vehicle_speed[tm],
                          air_cabin_heat_transfer_coef)
        Q[tm][4] = Q_tr
        Q[tm][0] = cabin_volume * D(t, P_out, h=h_out) * cp(t) * (
                t - t_1) - Q_in_per - Q_in_vent + Q_out_vent + Q_tr
        T[tm] = t

        # more info for debugging
        Q[tm][5] = D(T_out, P_out, h=h_out)
        Q[tm][6] = D(t, P_out, h=h_out)
        Q[tm][7] = resistances(zone_layer, zone_area, layer_conductivity,
                               layer_thickness, vehicle_speed[tm],
                               air_cabin_heat_transfer_coef)
    return Q, T


@numba.jit(nopython=True)
def cp(T):
    """        #TODO DOCSTRING

    Args:
        T ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.interp(T, T_RNG, CP_RNG)


def plot_multi(data, cols=None, spacing=.1, **kwargs):
    """        #TODO DOCSTRING

    Args:
        data ([type]): [description]
        cols ([type], optional): [description]. Defaults to None.
        spacing (float, optional): [description]. Defaults to .1.

    Returns:
        [type]: [description]
    """

    from pandas import plotting

    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    colors = getattr(
        getattr(plotting, '_matplotlib').style,
        '_get_standard_colors')(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(ax=ax_new,
                                  label=cols[n],
                                  color=colors[n % len(colors)],
                                  **kwargs)
        ax_new.set_ylabel(ylabel=cols[n])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    return ax


# TODO: This function should go in a reporting module like report.py
def balance(db, tscode, include=None):
    """        #TODO DOCSTRING

    Args:
        db ([type]): [description]
        tscode ([type]): [description]
        include ([type], optional): [description]. Defaults to None.

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """
    if db.db[tscode]["kind"] != "consumption":
        raise Exception(
            "code '{}' does not correspond to a consumption profile".format(
                tscode))
    if include is None:
        # all trips
        flag = True
        cons = 0
        distance = 0
        for trip in db.db[tscode]["Trips"].trips:
            if flag:
                value = np.zeros(trip.balance["value"].shape)
                flag = False
            value = value + trip.balance["value"]
            cons = cons + trip.consumption['value']  # kWh
            distance = distance + trip.distance['value']  # km
        label = trip.balance["label"]
        source = trip.balance["source"]
        target = trip.balance["target"]
        consumption = cons
        rate = consumption * 100 / distance
    elif isinstance(include, int):
        value = db.db[tscode]["Trips"].trips[include].balance["value"]
        label = db.db[tscode]["Trips"].trips[include].balance["label"]
        source = db.db[tscode]["Trips"].trips[include].balance["source"]
        target = db.db[tscode]["Trips"].trips[include].balance["target"]
        distance = db.db[tscode]["Trips"].trips[include].distance["value"]
        consumption = db.db[tscode]["Trips"].trips[include].consumption[
            "value"]
        rate = consumption * 100 / distance  # kWh/100 km
    elif isinstance(include, list):
        flag = True
        cons = 0
        distance = 0
        count = -1
        for trip in db.db[tscode]["Trips"].trips:
            count += 1
            if include[0] <= count < include[1]:
                if flag:
                    value = np.zeros(trip.balance["value"].shape)
                    flag = False
                value = value + trip.balance["value"]
                cons = cons + trip.consumption["value"]
                distance = distance + trip.distance["value"]
        label = trip.balance["label"]
        source = trip.balance["source"]
        target = trip.balance["target"]
        consumption = cons
        rate = consumption * 100 / distance  # kWh/100 km
    return distance, consumption, rate, label, source, target, value
