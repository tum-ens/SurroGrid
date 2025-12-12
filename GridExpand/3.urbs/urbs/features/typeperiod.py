import pyomo.core as pyomo
import pandas as pd
import tsam.timeseriesaggregation as tsam
from datetime import datetime, timedelta
import numpy as np
from urbs.identify import *
import re

### Apply timeseries aggregation
def run_tsam(data, noTypicalPeriods, hoursPerPeriod):
    """ Included time series for period selection:
    - Demand: water_heat, space_heat, mobilities, electricity, electricity-reactive
    - SupIm: solar
    - BSP: electricity_import, electricity_feed_in
    - TimeVarEff: heatpump_air, charging stations
    - Availability: import"""

    ################# Input data manipulations + weight assignment  ##############################
    ### Extract all relevant columns for type period selection
    filtered_dfs = []                                   # container list for tsam input columns
    extract_from = [data["weather"]]                    # List of data sheets from which columns are extracted for tsam input
    extract_if_startwith = ["Tamb", "Irradiation"]      # List of column names to extract
    for df in extract_from:
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels >= 2:
            mask = df.columns.get_level_values(1).str.startswith(tuple(extract_if_startwith))
            filtered_dfs.append(df.loc[:, mask])
        else: raise ValueError("Check dataframe validity and column multiindex naming convention!")
    time_series_data = pd.concat(filtered_dfs, axis=1)

    ### Assign datetime index which is required by tsam method
    date_hour = np.arange(datetime(2022, 1, 1), datetime(2023, 1, 1), timedelta(hours=1)).astype(datetime).T
    time_series_data = time_series_data.iloc[1:, :]           # drop initialization timestep (which is 0 and just used for storage setup later)
    time_series_data['DateTime'] = date_hour                  # add column with datetime
    time_series_data = time_series_data.set_index('DateTime') # set required index for tsam method

    ### Generate weights for each time series
    # normalize weights for seasonal time series to their occurence in nodes:
    weights={}
    for col in time_series_data.columns:
        if col[1] == 'Tamb':
            weights[col] = 1.
        if col[1] == 'Irradiation': 
            weights[col] = 1.

    ################### TSAM settings: extreme periods and clustering method #########################
    ### Add extreme periods
    addMeanMin_cols = None  # forms the mean of the regarded typeperiod (e.g. temperature for each week), then takes period with minimum mean
    addMeanMax_cols = None  # forms the max of the regarded typeperiod (e.g. temperature for each week), then takes period with minimum mean
    addMeanMin_cols = [col for col in time_series_data.columns if col[1] == 'Tamb']
    addMeanMax_cols = [col for col in time_series_data.columns if col[1] == 'Irradiation']


    ### Clustering settings
    extremePeriodMethod = "replace_cluster_center"    # Alternative: 'new_cluster_center'
    clusterMethod = 'hierarchical'                    # good, because deterministic
    representationMethod = "medoidRepresentation"     # final cluster representatives are chosen from actual typeperiod members (alternative: cluster mean representation)
    segmentation = False                              # No segmentation of final typeperiods into representative segments
    rescaleClusterPeriods = False                     # No rescaling of cluster data, as we will extract typeperiods manually from input data
    if extremePeriodMethod=="new_cluster_center":     # Reduce number of typical periods by how many extreme periods will be added (s.t. final typical weeks still same)
        noTypicalPeriods -= sum([len(addMeanMin_cols) if addMeanMin_cols is not None else 0, len(addMeanMax_cols) if addMeanMax_cols is not None else 0])
        if noTypicalPeriods < 1: raise ValueError("Number extreme periods cannot be higher than number of typical periods")


    ################################# Conduct TSAM #######################################
    ### Apply tsam method described in: https://tsam.readthedocs.io/en/latest/index.html
    aggregation = tsam.TimeSeriesAggregation(time_series_data,                                # timeseries to which to apply tsam
                                             noTypicalPeriods=noTypicalPeriods,               # how many typical periods
                                             hoursPerPeriod=hoursPerPeriod,                   # how many time steps in a period
                                             clusterMethod=clusterMethod,                     # hierarchical (which is deterministic)
                                             weightDict=weights,                              # weights for each time series (distance are scaled with weights, thus higher contribution in overall type week difference)
                                             extremePeriodMethod = extremePeriodMethod,       # how to include extreme period (here: use as new cluster center) (alternatively: replace cluster center in which extreme period lies)
                                             addMeanMin = addMeanMin_cols,                    # columns for which the mean minimum value type period should be added as extreme period cluster center
                                             addMeanMax = addMeanMax_cols,                    # same for mean maximum
                                             representationMethod=representationMethod,       # represent type period with actual sample weeks instead of cluster means
                                             segmentation = segmentation,                     # No segmentation of final typeperiods into representative segments
                                             rescaleClusterPeriods = rescaleClusterPeriods)   # Do not rescale data to original as we extract that from input data
    aggregation.createTypicalPeriods()       # Conduct tsam


    ##################### Extract and return relevant data ###############################
    ### Extract relevant tsam data
    # What typeweek was a particular week replaced by, [0, 1, 1, 1, 2, 0, 5, ...]
    orderPeriods = aggregation.clusterOrder
    # tsam data to be written to a separate h5 file
    tsam_data = {}
    tsam_data['clusterCenterIndices'] = pd.DataFrame(aggregation.clusterCenterIndices)  # e.g. [37, 42, 18, 2, 46]
    tsam_data['clusterOrder'] = pd.DataFrame(aggregation.clusterOrder)                  # e.g. [0, 0, 4, 3, 1, 1, 2, 0, ...]
    tsam_data['extremePeriodMethod'] = pd.Series(aggregation.extremePeriodMethod)       # e.g. replace_cluster_center 
    tsam_data['clusterMethod'] = pd.Series(aggregation.clusterMethod)                   # e.g. hierarchical
    tsam_data['noTypicalPeriods'] = pd.DataFrame({aggregation.noTypicalPeriods})        # e.g. 6
    tsam_data['hoursPerPeriod'] = pd.DataFrame({aggregation.hoursPerPeriod})            # e.g. 168
    tsam_data["extremePeriodIndices"] = pd.Series([value["stepNo"] for value in aggregation.extremePeriods.values()], dtype='int64') # additionally added extremePeriod week indices (if not already a cluster center) e.g. [50] taken from extremePeriods with key column_name = (columnindex1, renamed columnindex2) = ("ambient", "Tamb"+"daily min."); value = {"stepNo": week with minimum, "profile": all time series data of this week (also other columns), "column":column_multiindex before rename}
   

    ### Create weight vector to store the respective weight per time step (with 0 weight for initial row)
    # Calculate weights by occurence (account for last week not being complete)
    last_period_nr = orderPeriods[-1]   # what typePeriod was last week replaced by (e.g. last week in year replaced by week 1)
    occurence_weight_factor = np.bincount(orderPeriods).astype(dtype="float64")   # how often does each period occur => list of weights
    occurence_weight_factor[last_period_nr] = occurence_weight_factor[last_period_nr] - 1 + (8760 % hoursPerPeriod)/hoursPerPeriod    # last period has to be weighted differently (as not full week subtract overassigned weight)

    # Sort weeks and weights (important s.t. later correct weights are assigned to timestep when slicing input data)
    sorted_weeks_weights = sorted(zip(aggregation.clusterCenterIndices, occurence_weight_factor), key=lambda x: x[0])
    weeks_to_keep, week_weights = zip(*sorted_weeks_weights)
    print(f"The total number of typeperiods is: {len(weeks_to_keep):.0f}")

    # Create weight vector
    weight_vector = pd.Series([0])          # create weight vector series (initialized with weight 0 for row 0)
    for i in range(len(weeks_to_keep)):
        new_weights = pd.Series(np.ones(hoursPerPeriod) * week_weights[i])             # create vector with length of typeperiod and weight of type period
        weight_vector = pd.concat([weight_vector, new_weights], ignore_index=True)     # concat all weight vectors with previously single 0 vector

    # Assign new type period weights
    data["type_period"].loc[data["type_period"].index[:len(weight_vector)], "weight_typeperiod"] = weight_vector.values
    data["type_period"] = data["type_period"].dropna(axis=0)
    
    # fix error for week 52 and reinidexing
    ### Extract type periods from input data
    # Compute the indices to keep (0 for first row, then all indices within the weeks of clusterCenterIndices)
    indices_to_keep = [0]+[i for i in range(len(time_series_data)+1) if ((i-1) // hoursPerPeriod) in weeks_to_keep]
    indices_to_keep_extra = []
    if len(indices_to_keep)<len(weight_vector): # only the case if week 52 is in weeks_to_keep then add the remaining hours from the beginning of the year
        indices_to_keep_extra=[i for i in range(1, len(weight_vector)-len(indices_to_keep)+1)]
    tsam_data['kept_timesteps'] = pd.Series(indices_to_keep + indices_to_keep_extra)
    new_timesteps = range(0, len(weight_vector))


    # Create a new MultiIndex where the 'year' stays the same and the new 'hour' is a range starting at 0.
    def reset_hour_counter(group):
        new_hour = range(len(group))
        group.index = pd.MultiIndex.from_arrays(
            [group.index.get_level_values('support_timeframe'), new_hour],
            names=['support_timeframe', 't'])
        return group

    # Select obtained tsam periods from input data
    # data_to_adjust = ["demand", "supim", "buy_sell_price", "eff_factor", "weather"]
    # for data_name in data_to_adjust:
    #     data[data_name] = data[data_name].reindex(indices_to_keep, level='t')
    #     data[data_name] = data[data_name].groupby(level='support_timeframe', group_keys=False).apply(reset_hour_counter)

    data_to_adjust = ["demand", "supim", "buy_sell_price", "eff_factor", "weather"]
    for data_name in data_to_adjust:
        data_base = data[data_name].reindex(indices_to_keep, axis=0, level='t')
        if indices_to_keep_extra != []:
            data_extra = data[data_name].reindex(indices_to_keep_extra, axis=0, level='t')
            data[data_name] = pd.concat([data_base, data_extra])
        else: data[data_name] = data_base
        data[data_name] = data[data_name].groupby(level='support_timeframe', group_keys=False).apply(reset_hour_counter)


    ############ Limit BEV charging demand ##############
    # Battery charging availability might become infeasible after tsam split, adjust timeseries (e.g. all of demand is placed in first hour of week, but cannot be met as charging cap too low)
    df_ava = data["eff_factor"]
    df_sto = data["storage"]
    df_pro = data["process"]
    df_demand = data["demand"]

    # Extract all mobility relevant demand columns
    filtered_columns = [col for col in df_ava.columns if col[1].startswith('charging_station')]
    df_ava = df_ava[filtered_columns]

    # Mark each home stretch by different integers so that they can be grouped by
    home_stretch_id = (df_ava != df_ava.shift()).cumsum()

    # Loop over each electric vehicle
    for column in filtered_columns:
        # Extract associated BEV battery capacity:
        number = re.search(r'\d+$', column[1]).group() if re.search(r'\d+$', column[1]) else None  # Extract number at end of charging station string
        battery_cap = df_sto.xs((column[0], f"mobility_storage{number}"), level=[1,2])["cap-up-c"].values[0]

        # Extract associated BEV charging station capacity:
        charging_cap = df_pro.xs((column[0], f"charging_station{number}"), level=[1,2])["cap-up"].values[0]

        # Loop over each stretch and check whether charging demand reasonable
        home_stretches = pd.DataFrame(home_stretch_id[column][df_ava[column]==1])
        for stretch_id, stretch_df in home_stretches.groupby(column):
            # Determine maximum charge limitation per stretch by charging station capacity 
            indices = stretch_df.index
            max_charge_allowed = len(indices)*charging_cap

            # Determine total demand in home stretch and if it violates conditions, reduce demand 
            total_stretch_demand = df_demand.loc[indices, (column[0], f"mobility{number}")].sum()
            passable = min(total_stretch_demand, max_charge_allowed-0.1, battery_cap-0.1)           # Include slack for feasibility 
            if total_stretch_demand != passable:
                df_demand.loc[indices, (column[0], f"mobility{number}")]*=(passable/total_stretch_demand)

    ##################### Exit ##########################
    # returns: data of selected type weeks; new range of timesteps; typeperiod weights (excluding 0 row); dict of most relvant aggregation data
    return data, new_timesteps, tsam_data


def select_predefined_timesteps(data, timesteps):
    data['demand'] = data['demand'][data['demand'].index.get_level_values(1).isin(timesteps)]
    data['supim'] = data['supim'][data['supim'].index.get_level_values(1).isin(timesteps)]
    data['eff_factor'] = data['eff_factor'][data['eff_factor'].index.get_level_values(1).isin(timesteps)]
    # data['availability'] = data['availability'][data['availability'].index.get_level_values(1).isin(timesteps)]
    data['buy_sell_price'] = data['buy_sell_price'][data['buy_sell_price'].index.get_level_values(1).isin(timesteps)]
    data['weather'] = data['weather'][data['weather'].index.get_level_values(1).isin(timesteps)]

    tsam_data = {}
    tsam_data['kept_timesteps'] = pd.Series(timesteps)

    return data, tsam_data


def add_typeperiod(m, hoursPerPeriod):
    ### change weight parameter to 1, since the whole year is representated by weight_typeperiod
    m.del_component(m.weight)
    m.weight = pyomo.Param(
        initialize=1,
        doc='Pre-factor for variable costs and emissions for annual result for type period = 1')
    ### create list with all period ends
    t_endofperiod_list = [i * hoursPerPeriod * m.dt for i in list(range(1,1+int(len(m.timesteps) / m.dt / hoursPerPeriod)))]

    # if m.mode['tsam'] and m.mode['tsam_season']:
    #     ### prepare time lists for set tuples
    #     start_end_typeperiods_list = []
    #     t_startofperiod_list = []
    #     for hour in t_endofperiod_list:
    #         start_end_typeperiods_list.append((hour + 1 - m.hoursPerPeriod, hour))
    #         t_startofperiod_list.append(hour + 1 - m.hoursPerPeriod)
    #     subsequent_typeperiods_list = []
    #     t_endofperiod_list_without_last = t_endofperiod_list[0:-1]
    #     for hour in t_endofperiod_list_without_last:
    #         subsequent_typeperiods_list.append((hour,hour+1))

    #     ### allocate weights to the specific period with a dict
    #     m.typeperiod_weights = dict(zip(t_endofperiod_list, m.weighting_order))

    #     ### define timeperiod sets
    #     m.t_startofperiod = pyomo.Set(
    #         within=m.t,
    #         initialize=t_startofperiod_list,
    #         ordered=True,
    #         doc='timestep at the start of each timeperiod')
    #     m.t_endofperiod = pyomo.Set(
    #         within=m.t,
    #         initialize=t_endofperiod_list,
    #         ordered=True,
    #         doc='timestep at the end of each timeperiod')
    #     m.subsequent_typeperiods = pyomo.Set(
    #         within=m.t * m.t,
    #         initialize=subsequent_typeperiods_list,
    #         ordered=True,
    #         doc='subsequent timesteps between two typeperiods')
    #     m.start_end_typeperiods = pyomo.Set(
    #         within=m.t * m.t,
    #         initialize=start_end_typeperiods_list,
    #         ordered=True,
    #         doc='start and end of each modeled typeperiod as tuple')

    #     ### enable seasonal storage with SOC variable and two constraints
    #     ### SOC variable
    #     if m.mode['sto']:
    #         m.deltaSOC = pyomo.Var(
    #             m.t_endofperiod, m.sto_tuples,
    #             within=pyomo.Reals,
    #             doc='Variable to describe the delta of a storage within each period')
    #         ### constraint to describe the SOC difference of a storage within a repeating period A
    #         m.res_delta_SOC = pyomo.Constraint(
    #             m.start_end_typeperiods, m.sto_tuples,
    #             rule=res_delta_SOC,
    #             doc='delta_SOC_A = weight * (SOC_A_tN - SOC_A_t0)')
    #         ### SOC constraint for two consecutive typeperiods A and B
    #         m.res_typeperiod_delta_SOC = pyomo.Constraint(
    #             m.subsequent_typeperiods, m.sto_tuples,
    #             rule=res_typeperiod_deltaSOC_rule,
    #             doc='SOC_B_t0 = SOC_A_t0 + delta_SOC_A')

    #         ### delete old ciclycity rule to enable typeperiod simulation
    #         del m.res_storage_state_cyclicity

    #         ### new ciclycity constraint for typeperiods
    #         m.res_storage_state_cyclicity_typeperiod = pyomo.Constraint(
    #             m.sto_tuples,
    #             rule=res_storage_state_cyclicity_rule_typeperiod,
    #             doc='storage content end >= storage content start - deltaSOC[last_typeperiod]')
    if False: pass
    else:
        # t_startofperiod_list = []
        # for hour in t_endofperiod_list:
        #     t_startofperiod_list.append(hour + 1 - m.hoursPerPeriod)
        # ### if tsam is not active classical
        # ### original timeset for cyclicity rule
        # m.t_startofperiod = pyomo.Set(
        #     within=m.t,
        #     initialize=t_startofperiod_list,
        #     ordered=True,
        #     doc='timestep at the start of each timeperiod')
        m.t_endofperiod = pyomo.Set(
            within=m.t,
            initialize=t_endofperiod_list,
            ordered=True,
            doc='timestep at the end of each timeperiod')
        # if not m.grid_plan_model:
        if not False:
            ### cyclicity contraint
            if m.mode['sto']:
                m.res_storage_state_cyclicity_typeperiod = pyomo.Constraint(
                    m.t_endofperiod, m.sto_tuples,
                    rule=res_storage_state_cyclicity_typeperiod_rule,
                doc='storage content initial == storage content at the end of each timeperiod')
    return m

### cyclicity rule without tsam
def res_storage_state_cyclicity_typeperiod_rule(m, t, stf, sit, sto, com):
    return (m.e_sto_con[m.t.at(1), stf, sit, sto, com] ==      # Indexing in pyomo starts at 1 not 0!
            m.e_sto_con[t, stf, sit, sto, com])

### SOC rule for each repeating typeperiod
def res_delta_SOC(m, t_0, t_end, stf, sit, sto, com):
    return ( m.deltaSOC[t_end, stf, sit, sto, com] ==
             (m.typeperiod_weights[t_end] - 1) * (m.e_sto_con[t_end, stf, sit, sto, com] - m.e_sto_con[t_0, stf, sit, sto, com]))

### new storage rule using tsam considering the delta SOC per repeating typeperiod
def res_typeperiod_deltaSOC_rule(m, t_A, t_B, stf, sit, sto, com):
    return (m.e_sto_con[t_B, stf, sit, sto, com] ==
            m.e_sto_con[t_A, stf, sit, sto, com] + m.deltaSOC[t_A, stf, sit, sto, com])

### new ciclycity rule for typeperiods
def res_storage_state_cyclicity_rule_typeperiod(m, stf, sit, sto, com):
    return (m.e_sto_con[m.t[len(m.t)], stf, sit, sto, com] >=
            m.e_sto_con[m.t[1], stf, sit, sto, com] - m.deltaSOC[m.t[len(m.t)], stf, sit, sto, com])  # Indexing in pyomo starts at 1 not 0!