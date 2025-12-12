from config import config

import pandas as pd
import numpy as np
import random


##############################################################
############### Sampling building occupancy ##################
##############################################################
def _compute_f(k, t, p, allowed_x, cache, tol):
    """
    Recursively compute f(k, t): an approximate “density” (or weight)
    of achieving a leftover sum t using k independent draws from p,
    allowing a tolerance tol in the base case.
    
    Args:
        k (int): number of variables.
        t (float): leftover sum.
        p (dict): mapping allowed x -> probability.
        allowed_x (list of float): list of allowed values.
        cache (dict): memoization dictionary.
        tol (float): tolerance for zero sum in the base case.
    
    Returns:
        float: weight of achieving leftover t with k draws.
    """
    # Round t to mitigate floating-point issues.
    key = (k, round(t, 9))
    if key in cache:
        return cache[key]
    if k == 0:
        # Accept t as "zero" if within tol.
        return 1.0 if abs(t) < tol else 0.0
    total = 0.0
    for x in allowed_x:
        total += p[x] * _compute_f(k-1, t - x, p, allowed_x, cache, tol)
    cache[key] = total
    return total

def _closest_allowed(val, allowed_x):
        """
        Returns the allowed value closest to val.
        """
        return min(allowed_x, key=lambda a: abs(a - val))

def _sample_sequence_with_tolerance(n, s, p, allowed_x, tol=0.99):
    """
    Samples a sequence [x_1, ..., x_n] from the distribution p(x) with the constraint
    that the total sum is approximately s (i.e. within tolerance tol). The first n-1 samples
    are drawn sequentially using conditional weights, and the final leftover is snapped to
    the closest allowed value.
    
    Args:
        n (int): total number of sampled variables.
        s (float): target sum.
        p (dict): mapping allowed x -> probability.
        allowed_x (list of float): list of allowed values.
        tol (float): tolerance for matching the target sum.
    
    Returns:
        list: a list of n samples whose sum is approximately s.
    """
    cache = {}
    sequence = []
    remaining_sum = s
    remaining_vars = n

    for i in range(n - 1):
        # Denom is f(remaining_vars, remaining_sum) i.e. weight for the remaining sum.
        denom = _compute_f(remaining_vars, remaining_sum, p, allowed_x, cache, tol)
        if denom == 0:
            raise ValueError("No valid continuation found; consider increasing tol.")
        choices = []
        for x in allowed_x:
            # Weight if we choose x is p(x) times the weight for achieving the remainder.
            f_val = _compute_f(remaining_vars - 1, remaining_sum - x, p, allowed_x, cache, tol)
            weight = p[x] * f_val
            if weight > 0:
                choices.append((x, weight))
        
        # Now randomly sample according to weight
        total_weight = sum(weight for (_, weight) in choices)
        r = random.uniform(0, total_weight)
        cumulative = 0.0
        for x, weight in choices:
            cumulative += weight
            if r <= cumulative:
                chosen = x
                break

        # Update remaining sums and variables after sampling an x
        sequence.append(chosen)
        remaining_sum -= chosen
        remaining_vars -= 1

    # For the final variable, we have a leftover which might not equal one of allowed_x.
    # Snap to the closest allowed value.

    final_value = _closest_allowed(remaining_sum, allowed_x)
    sequence.append(final_value)
    return sequence

def _get_occupancy_distribution(prob:dict, n_hh:int, n_occ:int)->list:
    """
    Samples a likely occupancy distribution over all households of a building, given that number of occupants adds up to n_occ.
    
    Parameters:
    - n_hh: number of households in building
    - n_occ: number of occupants in building

    Returns: 
    - list of number of occupants in each household of a building 
    """
    if pd.isna(n_hh) or pd.isna(n_occ): 
        return []
    else:
        # To check whether occupancy can even be fulfilled by statistics:
        min_dist_member = min(prob.keys())          # minimum household size in statistics
        max_dist_member = max(prob.keys())          # maximum household size in statistics

        if max_dist_member*n_hh < n_occ:            # statistics do not allow for filling up building, simply assing max occupants to each household 
            return [max_dist_member]*n_hh
        elif min_dist_member*n_hh > n_occ:          # fewer occupants in building than covered by stats, simply assign min occupants to each household
            return [min_dist_member]*n_hh
        else:
            allowed_x = prob.keys()
            try:
                return _sample_sequence_with_tolerance(n_hh, n_occ, prob, allowed_x, tol=0.95)
            except: 
                return [_closest_allowed(allowed_x, n_occ/n_hh)]*n_hh

def _assign_household_occupancy(df_buildings):
    if len(df_buildings)==0:
        df_buildings["occ_list"]=pd.NA
        return df_buildings
    else:
        df_prob = config.HH_SIZE_DISTRIBUTION
        prob = dict(zip(df_prob["size"], df_prob["probability"]))          # retrieve polynomial encoding household size probabilities
        df_buildings['occ_list'] = df_buildings.apply(lambda row: _get_occupancy_distribution(prob, row['houses_per_building'], row['occupants']), axis=1)
        return df_buildings
    

##############################################################
################ Sampling building demands ###################
##############################################################
def _get_total_demands(cdf, occ_list):
    if len(occ_list)==0:
        return []
    else:
        demand_list = []

        u = np.random.rand(len(occ_list))
        columns = cdf.columns.get_level_values(0).unique()

        # for hh in cdf.columns.get_level_values(0).unique():
        #     sampled_x = np.interp(u, cdf[hh, "Y"], cdf[hh, "X"])

        for i, n_occ in enumerate(occ_list):
            if n_occ > 3: n_occ = 4
            n_occ = int(n_occ)
            demand_list.append(np.interp(u[i], cdf[columns[n_occ-1], "Y"], cdf[columns[n_occ-1], "X"]))
        return demand_list

def _assign_total_elec_demands(df_buildings):
    if len(df_buildings)==0:
        df_buildings["demand_tot_list"]=pd.NA
        return df_buildings
    else:
        df_cdfs = config.ELEC_BY_HHSIZE_CDFS_NOHEAT
        df_buildings['demand_tot_list'] = df_buildings.apply(lambda row: _get_total_demands(df_cdfs, row['occ_list']), axis=1)
        return df_buildings


##############################################################
################## Sampling building use #####################
##############################################################
def _get_use_type(dist, type, use):
    if use == "Residential":  # Use pre-assigned type
        return type
    if use == "Public":       # Sample from distribution
        return np.random.choice(dist["type"], p=dist['public_prob'])
    if use == "Commercial":   # Sample from distribution
        return np.random.choice(dist["type"], p=dist['commercial_prob'])

def _assign_use_type(df_buildings):
    df_type_dist = config.TYPE_GHD_DISTRIBUTION
    df_buildings["type"] = df_buildings.apply(lambda row: _get_use_type(df_type_dist, row['type'], row["use"]), axis=1)
    return df_buildings

def _get_single_building_elec_timeseries_res(yearly_demand_list, df_normalized_lps, lps_total_demands):
    ts_list=[]
    for demand in yearly_demand_list:
        closest_id = (np.abs(lps_total_demands["kWh"] - demand)).idxmin()
        closest_device = lps_total_demands.loc[closest_id, "devicenumber"]
        
        # Rescale the chosen timeseries so its total equals the current step.
        scaled_series = df_normalized_lps[closest_device] * demand
        ts_list.append(scaled_series)

    # Combine all scaled timeseries into a DataFrame: each column corresponds to a step.
    df_ts = pd.concat(ts_list, axis=1)
    total_ts = df_ts.sum(axis=1)
    return total_ts

def _get_single_building_elec_timeseries_ghd(type, area, floors, df_normalized_lps_ghd):
    return df_normalized_lps_ghd[type]*area*floors


##############################################################
############## Generation, Publicly Callable #################
##############################################################
def sample_statistics(df_buildings):
    df_buildings = _assign_household_occupancy(df_buildings)
    df_buildings = _assign_total_elec_demands(df_buildings)
    df_buildings = _assign_use_type(df_buildings)
    return df_buildings

def get_elec_demand(df_buildings):
    df_normalized_lps_res = pd.read_hdf(config.ELEC_LPS_PATH, key="df_normalized_scaled")
    lps_res_total_demand = pd.read_hdf(config.ELEC_LPS_PATH, key="df_sums")
    df_normalized_lps_ghd = pd.read_csv(config.ELEC_GHD_PATH, skiprows=1, header=[0])

    # Apply function and create a new DataFrame
    data_dict_res = {row["bus"]: _get_single_building_elec_timeseries_res(row['demand_tot_list'], df_normalized_lps_res, lps_res_total_demand) for idx, row in df_buildings.iterrows() if row["use"]=="Residential"}
    data_dict_ghd = {row["bus"]: _get_single_building_elec_timeseries_ghd(row['type'], row["area"], row["floors"], df_normalized_lps_ghd) for idx, row in df_buildings.iterrows() if row["use"]!="Residential"}

    # Convert to DataFrame
    df_elec_demand_res = pd.DataFrame(data_dict_res).reset_index(drop=True)
    df_elec_demand_ghd = pd.DataFrame(data_dict_ghd).reset_index(drop=True)
    df_elec_demand = pd.concat([df_elec_demand_res, df_elec_demand_ghd], axis=1)
    df_elec_demand.columns = pd.MultiIndex.from_product([df_elec_demand.columns, ["electricity"]])

    # Assign correct total demand to non-res
    non_res_mask = df_buildings["use"] != "Residential"
    df_buildings.loc[non_res_mask, "demand_tot_list"] = df_buildings.loc[non_res_mask, "bus"].map(
        lambda bus: df_elec_demand[bus].sum().tolist()
    )

    return df_buildings, df_elec_demand

# def get_elec_react_demand(df_elec_demand):
#     conversion_factor = math.tan(math.acos(config.ELEC_REACT_PF))
#     df_elec_react_demand = df_elec_demand.copy()
#     df_elec_react_demand*=conversion_factor
#     df_elec_react_demand.columns = df_elec_react_demand.columns.set_levels(["electricity-reactive"]*len(df_elec_react_demand.columns.levels[1]), level=1)
#     return df_elec_react_demand

def create_pro_elec(consumer_bus_list):
    df_pro_base = pd.DataFrame(consumer_bus_list, columns=['Site'])
    df_pro_base[["Process","inst-cap","cap-up","inv-cost-fix","inv-cost","fix-cost","var-cost","wacc","depreciation","pf-min"]] = (
        "import", config.IMP_INST_CAP, config.IMP_CAP_UP, config.IMP_INV_COST_FIX, config.IMP_INV_COST, 
        config.IMP_FIX_COST, config.IMP_VAR_COST, config.IMP_WACC, config.IMP_DEPRECIATION, config.IMP_PF_MIN)

    df_pro_feed = df_pro_base.copy()
    df_pro_feed["Process"] = "feed_in"

    # df_pro_Q = df_pro_base.copy()
    # df_pro_Q["Process"] = "Q_feeder_central"

    # df_pro = pd.concat([df_pro_base, df_pro_feed, df_pro_Q], axis=0)
    df_pro = pd.concat([df_pro_base, df_pro_feed], axis=0)
    return df_pro.reset_index(drop=True)

def create_com_elec(consumer_bus_list):
    df_com_base = pd.DataFrame(consumer_bus_list, columns=['Site'])
    df_com_base[["Commodity","Type","price"]] = ("electricity", "Demand", np.nan)

    # df_com_Q = df_com_base.copy()
    # df_com_Q["Commodity"] = "electricity-reactive"

    df_com_imp = df_com_base.copy()
    df_com_imp[["Commodity","Type","price"]] = ("electricity_import", "Buy", 1)

    df_com_feed = df_com_base.copy()
    df_com_feed[["Commodity","Type","price"]] = ("electricity_feed_in", "Sell", 1)

    # df_com = pd.concat([df_com_base, df_com_Q, df_com_imp, df_com_feed], axis=0)
    df_com = pd.concat([df_com_base, df_com_imp, df_com_feed], axis=0)
    return df_com.reset_index(drop=True)

def create_pro_com_elec():
    # df_pro_com = pd.DataFrame({
    #     'Process':   ["import", "import", "feed_in", "feed_in", "Q_feeder_central"],
    #     'Commodity': ["electricity_import", "electricity", "electricity", "electricity_feed_in", "electricity-reactive"],
    #     'Direction': ["In", "Out", "In", "Out", "Out"],
    #     'ratio':     [1, 1, 1, 1, 1]
    # })
    df_pro_com = pd.DataFrame({
        'Process':   ["import",             "import",       "feed_in",     "feed_in"],
        'Commodity': ["electricity_import", "electricity",  "electricity", "electricity_feed_in"],
        'Direction': ["In",                 "Out",          "In",          "Out"],
        'ratio':     [1,                    1,              1,              1]
    })
    return df_pro_com.reset_index(drop=True)

def create_sto_elec(consumer_bus_list):
    df_sto = pd.DataFrame(consumer_bus_list, columns=['Site'])
    df_sto[["Storage","Commodity","inst-cap-c","cap-up-c","inst-cap-p","cap-up-p","eff-in","eff-out","discharge","ep-ratio",
            "inv-cost-p","inv-cost-c","fix-cost-p","fix-cost-c","var-cost-p","wacc","depreciation"]] = (
            "battery_private", "electricity", config.BS_INST_CAP_C, config.BS_CAP_UP_C, config.BS_INST_CAP_P, 
            config.BS_CAP_UP_P, config.BS_EFF_IN, config.BS_EFF_OUT, config.BS_DISCHARGE, config.BS_EP_RATIO,
            config.BS_INV_COST_P, config.BS_INV_COST_C, config.BS_FIX_COST_P, config.BS_FIX_COST_C,
            config.BS_VAR_COST_P, config.BS_WACC, config.BS_DEPRECIATION)
    return df_sto.reset_index(drop=True)