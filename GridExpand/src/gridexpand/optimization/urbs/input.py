"""Read the Step 2 urbs input HDF5 and prepare the data of one pyomo model."""

import copy

import numpy as np
import pandas as pd
import pyomo.environ as pyomo

from .features.modelhelper import invcost_factor
from .identify import identify_mode

EV_SESSION_KEYS = ("ev_sessions", "ev_session_hours")
TIME_SERIES_KEYS = ("buy_sell_price", "demand", "eff_factor", "supim", "weather", "building_thermal_timeseries")

# The single support timeframe of every table (a relic of intertemporal urbs). It
# labels the `stf` index level of all urbs_out results. It used to be the
# calendar year of the run (date.today().year); it is fixed to the year the
# existing results were produced with, so reruns do not depend on the calendar.
SUPPORT_TIMEFRAME = 2026


def read_input_h5(input_path):
    """Read all ``/urbs_in/*`` tables of a Step 2 file into the urbs input dict.

    Time series get an all-zero initialization row at ``t = 0`` (storage start
    state), every table except the EV session tables gets a leading
    ``support_timeframe`` index level, and ``type_period`` and an empty
    ``global_prop`` are added.
    """
    ### Read out data
    with pd.HDFStore(input_path, mode="r") as store:
        input_keys = [key for key in store.keys() if key.startswith("/urbs_in/")]
        print("Available input datasets:", input_keys)

    # Read only input data into a dictionary
    raw_data_dict = {}
    for key in input_keys:
        raw_data_dict[key.replace("/urbs_in/", "")] = pd.read_hdf(input_path, key=key)

    # Extract site data
    sites = pd.read_hdf(path_or_buf=input_path, key="/urbs_in/demand").columns.get_level_values(0).unique()
    raw_data_dict["site"] = pd.DataFrame(index=list(sites))
    raw_data_dict["site"].index.name = "Name"

    # Add an initialization row to timeseries data:
    for key in raw_data_dict.keys():
        if key in TIME_SERIES_KEYS:
            columns = raw_data_dict[key].columns
            zero_row = raw_data_dict[key].iloc[[0]].copy() if key == "building_thermal_timeseries" else pd.DataFrame([np.zeros(len(columns))], columns=columns)
            raw_data_dict[key] = pd.concat([zero_row, raw_data_dict[key]], ignore_index=True)

    ### Convert columns to multiindex:
    # A table can legitimately be empty: retiring the virtual mobility storage
    # leaves buildings without a stationary battery with no storage rows at all.
    for key, index_columns in (
        ("commodity", ['Site', 'Commodity', 'Type']),
        ("process", ['Site', 'Process']),
        ("process_commodity", ['Process', "Commodity", "Direction"]),
        ("storage", ['Site', "Storage", "Commodity"]),
    ):
        table = raw_data_dict[key]
        if table.empty and not set(index_columns).issubset(table.columns):
            table = pd.DataFrame(columns=list(index_columns))
        raw_data_dict[key] = table.set_index(index_columns)

    ### Add support_timeframe to Multiindex
    support_timeframe = SUPPORT_TIMEFRAME
    for key in raw_data_dict.keys():
        # The EV session tables are flat relational tables keyed by session_id;
        # they carry their own model-hour column and must not be reindexed.
        if key in EV_SESSION_KEYS:
            continue
        if key in TIME_SERIES_KEYS:
            raw_data_dict[key] = pd.concat([raw_data_dict[key]], keys=[support_timeframe], names=['support_timeframe', 't'])
        else:
            raw_data_dict[key] = pd.concat([raw_data_dict[key]], keys=[support_timeframe], names=['support_timeframe'])

    ### Add remaining input columns
    typeperiod = raw_data_dict["demand"][[raw_data_dict["demand"].columns[0]]].copy()  # copies the first column as a DataFrame
    typeperiod.iloc[:, 0] = np.nan                   # sets all values to NaN
    typeperiod.columns = ['weight_typeperiod']
    raw_data_dict["type_period"] = typeperiod

    raw_data_dict["global_prop"] = pd.DataFrame()

    return raw_data_dict


def pyomo_model_prep(data, timesteps):
    """Create the ConcreteModel shell and the parameter dicts of one cluster.

    Adds ``support_timeframe``, ``cost_factor`` and ``invcost-factor`` columns to
    the cluster's commodity, process, site and storage tables (in place).

    Args:
        data: urbs input dict of one cluster.
        timesteps: range of modeled timesteps (including the t = 0 row).

    Returns:
        A pyomo ConcreteModel without components.
    """
    m = pyomo.ConcreteModel()

    ###### Assign basic properties
    m.global_prop = data['global_prop']
    m.mode = identify_mode(data)
    m.stf_list = m.global_prop.index.levels[0].tolist()  # create list with all support timeframe values
    m.timesteps = timesteps

    m.cost_type_list = ['Invest', 'Fixed', 'Variable']
    if m.mode['bsp']:
        m.cost_type_list.extend(['Revenue', 'Purchase'])

    ###### Extract relevant quantities from data
    commodity = data['commodity']
    process = data['process']
    site = data['site']

    ##### Assign quantitites to model
    m.demand_dict = data['demand'].to_dict() # P demands
    m.supim_dict = data['supim'].to_dict()   # PV

    if m.mode['sto']:
        storage = data['storage'].dropna(axis=0, how='all')  # drop all fully empty rows
    if m.mode['bsp']:
        m.buy_sell_price_dict = data["buy_sell_price"].dropna(axis=0, how='all').to_dict()
    if m.mode['tve']:
        m.eff_factor_dict = data["eff_factor"].dropna(axis=0, how='all').to_dict()
    if m.mode['tdy']:
        m.typeperiod = data['type_period'].dropna(axis=0, how='all').to_dict()
    else:
        # if mode 'typeperiod' is not active, create a dict with ones
        temp = pd.DataFrame(index=data['demand'].dropna(axis=0, how='all').index)
        temp['weight_typeperiod']=1
        m.typeperiod = temp.to_dict()

    # Create columns of support timeframe values
    commodity['support_timeframe'] = (commodity.index.get_level_values('support_timeframe'))
    process['support_timeframe'] = (process.index.get_level_values('support_timeframe'))
    site['support_timeframe'] = (site.index.get_level_values('support_timeframe'))
    if m.mode['sto']:
        storage['support_timeframe'] = (storage.index.get_level_values('support_timeframe'))

    # process input/output ratios
    m.r_in_dict = (data['process_commodity'].xs('In', level='Direction')['ratio'].to_dict())
    m.r_out_dict = (data['process_commodity'].xs('Out', level='Direction')['ratio'].to_dict())

    pro_inv_cost_fix = data["process"]['inv-cost-fix']
    pro_inv_cost_fix = pro_inv_cost_fix[pro_inv_cost_fix > 0]
    m.pro_inv_cost_fix_dict = pro_inv_cost_fix.to_dict()

    if m.mode['sto']:
        try:
            # storages with fixed energy-to-power ratio
            # IMPLEMENT: there also applied e2p ratio for thermal storage, is that wanted?
            sto_ep_ratio = storage['ep-ratio']
            m.sto_ep_ratio_dict = sto_ep_ratio[sto_ep_ratio >= 0].to_dict()
        except KeyError:
            m.sto_ep_ratio_dict = {}

    # derive annuity factors from WACC and depreciation duration (one year problem)
    # (no fallback: a failing annuity must not silently become the full capex)
    process['invcost-factor'] = (
        process.apply(
        lambda x: invcost_factor(
            x['depreciation'],
            x['wacc']),
        axis=1))

    # cost factor will be set to 1 for non intertemporal problems
    commodity['cost_factor'] = 1
    process['cost_factor'] = 1
    site['cost_factor'] = 1
    if m.mode['sto']:
        storage['invcost-factor'] = (
            storage.apply(lambda x:
                          invcost_factor(x['depreciation'],
                                         x['wacc']),
                          axis=1))
        storage['cost_factor'] = 1

    # Converting Data frames to dictionaries
    m.commodity_dict = commodity.to_dict()
    m.process_dict = process.to_dict()
    m.site_dict = site.to_dict()

    if m.mode['sto']:
        m.storage_dict = storage.to_dict()
        m.sto_linked_capacity_dict = {}
        linked_process_column = 'linked-process'
        linked_ratio_column = 'max-energy-per-process-capacity'
        if linked_process_column in storage and linked_ratio_column in storage:
            linked_rows = storage[
                storage[linked_process_column].notna()
                | storage[linked_ratio_column].notna()
            ]
            for storage_index, row in linked_rows.iterrows():
                linked_process = row[linked_process_column]
                linked_ratio = row[linked_ratio_column]
                if pd.isna(linked_process) or pd.isna(linked_ratio):
                    raise ValueError(
                        'Storage capacity linkage requires both linked-process and '
                        'max-energy-per-process-capacity.'
                    )
                linked_ratio = float(linked_ratio)
                if linked_ratio <= 0:
                    raise ValueError(
                        'max-energy-per-process-capacity must be positive.'
                    )
                stf, sit, _, _ = storage_index
                process_index = (stf, sit, str(linked_process))
                if process_index not in process.index:
                    raise ValueError(
                        f'Storage {storage_index!r} references missing process '
                        f'{process_index!r}.'
                    )
                m.sto_linked_capacity_dict[storage_index] = (
                    str(linked_process), linked_ratio
                )
    return m


def get_cluster_data(data, cluster):
    """Return the input dict restricted to the buildings (sites) in ``cluster``.

    Site-indexed tables, site columns and EV sessions are filtered (the results
    are copies); all other tables (global properties, process-commodity ratios,
    prices, weather, type periods) are deep-copied unchanged. Key order is kept.
    """
    sites = set(cluster)
    filter_sessions = 'ev_sessions' in data and not data['ev_sessions'].empty
    if filter_sessions:
        sessions = data['ev_sessions'][data['ev_sessions']['site'].isin(cluster)]
        retained = set(sessions['session_id'])
    cluster_data = {}
    for key, value in data.items():
        if key == 'building_thermal_parameters':
            cluster_data[key] = value[value['Site'].isin(sites)].copy()
        elif key == 'building_thermal_timeseries':
            retained_buildings = set(data['building_thermal_parameters'].loc[data['building_thermal_parameters']['Site'].isin(sites), 'building_objectid'].astype(str))
            cluster_data[key] = value.loc[:, value.columns.get_level_values(0).isin(retained_buildings)].copy()
        elif key in ('commodity', 'process', 'site', 'storage'):
            cluster_data[key] = value[value.index.get_level_values(1).isin(cluster)]
        elif key in ('demand', 'supim', 'eff_factor'):
            cluster_data[key] = value[[column for column in value.columns if column[0] in sites]]
        elif filter_sessions and key == 'ev_sessions':
            cluster_data[key] = sessions
        elif filter_sessions and key == 'ev_session_hours':
            cluster_data[key] = value[value['session_id'].isin(retained)]
        else:
            cluster_data[key] = copy.deepcopy(value)
    return cluster_data
