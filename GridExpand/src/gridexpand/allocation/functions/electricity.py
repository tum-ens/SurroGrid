from gridexpand.allocation.config import config

import functools
import random

import numpy as np
import pandas as pd

from gridexpand.common.reproducibility import frame_fingerprint, physical_building_id, stable_seed


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

def _sample_sequence_with_tolerance(n, s, p, allowed_x, tol=0.99, rng=None):
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
        r = (rng or random).uniform(0, total_weight)
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

def _get_occupancy_distribution(prob:dict, n_hh:int, n_occ:int, rng=None)->list:
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
        n_hh = int(round(n_hh))
        n_occ = float(n_occ)
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
                return _sample_sequence_with_tolerance(
                    n_hh, n_occ, prob, allowed_x, tol=0.95, rng=rng
                )
            except ValueError:
                # No continuation within the tolerance: give every household
                # the allowed size closest to the building mean.
                return [_closest_allowed(n_occ / n_hh, allowed_x)] * n_hh

def _assign_household_occupancy(df_buildings, base_seed):
    df_buildings["occ_list"] = pd.NA
    if len(df_buildings) == 0:
        return df_buildings
    df_prob = config.HH_SIZE_DISTRIBUTION
    prob = dict(zip(df_prob["size"], df_prob["probability"]))
    residential = df_buildings["component_category"].eq("Residential")
    df_buildings.loc[residential, "occ_list"] = df_buildings.loc[residential].apply(
        lambda row: _get_occupancy_distribution(
            prob,
            row["households"],
            row["occupants"],
            rng=random.Random(stable_seed(
                base_seed,
                physical_building_id(row),
                row["component_category"],
                "electricity",
                "occupancy",
            )),
        ),
        axis=1,
    )
    return df_buildings
    

##############################################################
################ Sampling building demands ###################
##############################################################
def _get_total_demands(cdf, occ_list, rng=None):
    if len(occ_list)==0:
        return []
    else:
        demand_list = []

        u = (rng or np.random.default_rng()).random(len(occ_list))
        columns = cdf.columns.get_level_values(0).unique()

        for i, n_occ in enumerate(occ_list):
            if n_occ > 3: n_occ = 4
            n_occ = int(n_occ)
            demand_list.append(np.interp(u[i], cdf[columns[n_occ-1], "Y"], cdf[columns[n_occ-1], "X"]))
        return demand_list

def _assign_total_elec_demands(df_buildings, base_seed):
    df_buildings["demand_tot_list"] = pd.NA
    if len(df_buildings) == 0:
        return df_buildings
    df_cdfs = config.ELEC_BY_HHSIZE_CDFS_NOHEAT
    residential = df_buildings["component_category"].eq("Residential")
    df_buildings.loc[residential, "demand_tot_list"] = df_buildings.loc[residential].apply(
        lambda row: _get_total_demands(
            df_cdfs,
            row["occ_list"],
            rng=np.random.default_rng(stable_seed(
                base_seed,
                physical_building_id(row),
                row["component_category"],
                "electricity",
                "annual_demand",
            )),
        ),
        axis=1,
    )
    return df_buildings


##############################################################
################## Sampling building use #####################
##############################################################
def _get_use_type(dist, source_building_type, component_category, rng=None):
    if component_category == "Residential":
        return source_building_type
    if component_category == "Public":
        return (rng or np.random.default_rng()).choice(dist["type"], p=dist["public_prob"])
    if component_category == "Commercial":
        return (rng or np.random.default_rng()).choice(dist["type"], p=dist["commercial_prob"])
    raise ValueError(f"Unsupported electricity component category: {component_category!r}")

def _assign_use_type(df_buildings, base_seed):
    df_type_dist = config.TYPE_GHD_DISTRIBUTION
    df_buildings["profile_type"] = df_buildings.apply(
        lambda row: _get_use_type(
            df_type_dist,
            row["source_building_type"],
            row["component_category"],
            rng=np.random.default_rng(stable_seed(
                base_seed,
                physical_building_id(row),
                row["component_category"],
                "electricity",
                "profile_type",
            )),
        ),
        axis=1,
    )
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

def _get_single_building_elec_timeseries_ghd(building_type, floor_area, df_normalized_lps_ghd):
    return df_normalized_lps_ghd[building_type] * floor_area


@functools.cache
def residential_load_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the normalized household load profiles and their annual sums.

    The 113 MB table is read once per process; callers must not modify it.
    """
    return (
        pd.read_hdf(config.ELEC_LPS_PATH, key="df_normalized_scaled"),
        pd.read_hdf(config.ELEC_LPS_PATH, key="df_sums"),
    )


@functools.cache
def commercial_load_profiles() -> pd.DataFrame:
    """Return the per-m2 commercial/public (GHD) load profiles, read once per process."""
    return pd.read_csv(config.ELEC_GHD_PATH, skiprows=1, header=[0])


##############################################################
############## Generation, Publicly Callable #################
##############################################################
def sample_statistics(df_components, base_seed=0):
    """Sample statistics on the explicit Residential/Commercial/Public rows."""
    if "component_category" not in df_components.columns:
        raise ValueError("Electricity sampling requires the building component manifest.")
    df_components = df_components.copy()
    df_components = _assign_household_occupancy(df_components, base_seed)
    df_components = _assign_total_elec_demands(df_components, base_seed)
    df_components = _assign_use_type(df_components, base_seed)
    return df_components

def get_elec_demand(df_components, base_seed=0, return_component_profiles=False):
    """Generate component profiles and aggregate them once at their shared bus."""
    required = {"component_id", "component_category", "effective_floor_area_m2", "bus", "included_in_lv"}
    missing = sorted(required.difference(df_components.columns))
    if missing:
        raise ValueError(f"Electricity demand requires component columns: {missing}")
    df_normalized_lps_res, lps_res_total_demand = residential_load_profiles()
    df_normalized_lps_ghd = commercial_load_profiles()

    result = df_components.copy()
    profile_by_component = {}
    profile_methods = {}
    for _, row in result.iterrows():
        if not bool(row["included_in_lv"]):
            continue
        bus = row["bus"]
        category = str(row["component_category"])
        if category == "Residential":
            profile = _get_single_building_elec_timeseries_res(
                row["demand_tot_list"],
                df_normalized_lps_res,
                lps_res_total_demand,
            )
            method = "residential_lps_by_household"
        else:
            profile = _get_single_building_elec_timeseries_ghd(
                row["profile_type"],
                float(row["effective_floor_area_m2"]),
                df_normalized_lps_ghd,
            )
            method = "ghd_by_effective_component_area"
        profile = profile.reset_index(drop=True)
        profile_by_component[str(row["component_id"])] = profile
        profile_methods[str(row["component_id"])] = method

    if not profile_by_component:
        raise ValueError("No included LV component remains for electricity allocation.")
    component_profiles = pd.DataFrame(profile_by_component)
    component_profiles.columns = pd.MultiIndex.from_tuples(
        [(component_id, "electricity") for component_id in component_profiles.columns]
    )
    result["annual_electricity_kwh"] = result["component_id"].map(
        {component_id: float(profile.sum()) for component_id, profile in profile_by_component.items()}
    ).fillna(0.0)
    result["profile_method"] = result["component_id"].map(profile_methods)
    result["profile_hash"] = result["component_id"].map(
        {
            component_id: frame_fingerprint(pd.DataFrame({"electricity": profile}))
            for component_id, profile in profile_by_component.items()
        }
    )
    first_rows = result.drop_duplicates("component_id")
    seed_by_component = {
        component_id: stable_seed(base_seed, object_id, category, "electricity", "profile")
        for component_id, object_id, category in zip(
            first_rows["component_id"], first_rows["objectid"], first_rows["component_category"]
        )
    }
    result["stable_seed"] = result["component_id"].map(seed_by_component)

    physical_profiles = {}
    physical_buses = {}
    component_annual = 0.0
    for _, row in result[result["annual_electricity_kwh"].gt(0)].iterrows():
        component_id = str(row["component_id"])
        profile = component_profiles[(component_id, "electricity")]
        object_id = str(row["objectid"])
        bus = int(row["bus"])
        previous_bus = physical_buses.setdefault(object_id, bus)
        if previous_bus != bus:
            raise AssertionError(
                f"Physical building {object_id} maps components to multiple buses."
            )
        physical_profiles[object_id] = (
            profile
            if object_id not in physical_profiles
            else physical_profiles[object_id].add(profile, fill_value=0.0)
        )
        component_annual += float(profile.sum())
    physical_annual = float(sum(profile.sum() for profile in physical_profiles.values()))
    if not np.isclose(component_annual, physical_annual, rtol=1e-9, atol=1e-6):
        raise AssertionError("Component-to-physical-building electricity aggregation does not conserve annual energy.")
    demand_by_bus = {}
    for object_id, profile in physical_profiles.items():
        bus = physical_buses[object_id]
        demand_by_bus[bus] = demand_by_bus.get(bus, 0.0) + profile
    df_elec_demand = pd.DataFrame(demand_by_bus).reset_index(drop=True)
    df_elec_demand.columns = pd.MultiIndex.from_product([df_elec_demand.columns, ["electricity"]])
    if not np.isclose(component_annual, float(df_elec_demand.to_numpy().sum()), rtol=1e-9, atol=1e-6):
        raise AssertionError("Component-to-bus electricity aggregation does not conserve annual energy.")
    if return_component_profiles:
        return result, df_elec_demand, component_profiles
    return result, df_elec_demand

def profile_components(df_components, base_seed=0):
    """Sample and profile the electricity components of one grid.

    Returns:
        ``(components, bus_demand, component_profiles)`` as from
        :func:`get_elec_demand` after :func:`sample_statistics`.
    """
    sampled = sample_statistics(df_components, base_seed)
    return get_elec_demand(sampled, base_seed=base_seed, return_component_profiles=True)


def aggregate_components_to_buildings(physical, components, *, residential_area=False):
    """Attach component results to one-row-per-building data.

    Adds ``occ_list`` (households of the building's Residential component, at
    most one per building, else ``[]``) and ``annual_electricity_kwh`` (sum over
    all its components, else 0). With ``residential_area`` it also adds
    ``residential_effective_floor_area_m2``.

    Args:
        physical: One row per physical building with ``objectid``.
        components: Profiled components from :func:`profile_components`.
        residential_area: Also add the residential effective floor area.

    Returns:
        A copy of ``physical`` with the added columns.
    """
    result = physical.copy()
    building_ids = result["objectid"].astype(str)
    component_ids = components["objectid"].astype(str)
    residential = components["component_category"].eq("Residential")
    occupancy = dict(zip(component_ids[residential], components.loc[residential, "occ_list"]))
    annual = components.groupby(component_ids)["annual_electricity_kwh"].sum()
    result["occ_list"] = building_ids.map(occupancy).apply(
        lambda value: value if isinstance(value, (list, tuple, np.ndarray)) else []
    )
    result["annual_electricity_kwh"] = building_ids.map(annual).fillna(0.0)
    if residential_area:
        area = (
            components.loc[residential]
            .groupby(component_ids[residential])["effective_floor_area_m2"]
            .sum()
        )
        result["residential_effective_floor_area_m2"] = building_ids.map(area).fillna(0.0)
    return result

def demand_component_audit(components, profiled, component_profiles):
    """Return the per-component electricity evidence of one allocation run.

    Args:
        components: Every component of the grid (``raw_data/building_components``).
        profiled: The profiled components of the demand scope.
        component_profiles: Hourly profiles, columns ``(component_id, "electricity")``.

    Returns:
        One row per component, written to ``raw_data/demand_component_audit``
        and ``surrogrid.demand_component_audit``. ``suppression_reason`` is
        ``outside_lv_scope`` (MV-direct), ``outside_demand_scope`` (e.g. a
        non-residential component in a residential-only run) or None.
    """
    indexed = profiled.set_index("component_id")
    profile_max = component_profiles.max(axis=0)
    profile_max.index = [str(column[0]) for column in component_profiles.columns]
    audit = components.copy().rename(columns={"component_category": "category"})
    audit["scenario_unit_id"] = audit["objectid"].astype(str)
    audit["commodity"] = "electricity"
    audit["annual_energy_kwh"] = audit["component_id"].map(
        indexed["annual_electricity_kwh"]
    ).fillna(0.0)
    audit["max_profile_value"] = audit["component_id"].map(profile_max).fillna(0.0)
    audit["profile_hash"] = audit["component_id"].map(indexed["profile_hash"])
    audit["profile_method"] = audit["component_id"].map(
        indexed["profile_method"]
    ).fillna("not_allocated")
    audit["stable_seed"] = audit["component_id"].map(indexed["stable_seed"])
    selected_ids = set(profiled["component_id"].astype(str))
    audit["suppression_reason"] = np.select(
        [
            ~audit["included_in_lv"].map(bool),
            audit["component_id"].astype(str).isin(selected_ids),
        ],
        ["outside_lv_scope", None],
        default="outside_demand_scope",
    )
    audit["source_asset_count"] = pd.NA
    audit["matched_swf_asset_count"] = pd.NA
    audit["mv_direct"] = audit["mv_direct"].astype(bool)
    return audit[
        [
            "component_id", "objectid", "scenario_unit_id", "bus", "category",
            "commodity", "annual_energy_kwh", "max_profile_value", "profile_hash",
            "profile_method", "stable_seed", "source_asset_count",
            "matched_swf_asset_count", "included_in_lv", "suppression_reason",
            "pylovo_version_id", "mix_score", "mix_rule", "mix_confidence", "mv_direct",
        ]
    ]

def create_pro_elec(consumer_bus_list, parameters):
    """Return the grid ``import`` and ``feed_in`` process rows of every consumer bus.

    Args:
        consumer_bus_list: Buses with base electricity demand.
        parameters: ``technologies.processes["grid_connection"]`` of the scenario.
    """
    df_pro_base = pd.DataFrame(consumer_bus_list, columns=['Site'])
    df_pro_base[["Process","inst-cap","cap-up","inv-cost-fix","inv-cost","fix-cost","var-cost","wacc","depreciation","pf-min"]] = (
        "import", parameters["installed_capacity_kw"], parameters["capacity_upper_kw"],
        parameters["fixed_investment_cost_eur"], parameters["investment_cost_eur_per_kw"],
        parameters["fixed_cost_eur_per_hour"], parameters["variable_cost_eur_per_kwh"],
        parameters["wacc"], parameters["depreciation_years"], parameters["minimum_power_factor"])

    df_pro_feed = df_pro_base.copy()
    df_pro_feed["Process"] = "feed_in"

    df_pro = pd.concat([df_pro_base, df_pro_feed], axis=0)
    return df_pro.reset_index(drop=True)

def create_com_elec(consumer_bus_list):
    df_com_base = pd.DataFrame(consumer_bus_list, columns=['Site'])
    df_com_base[["Commodity","Type","price"]] = ("electricity", "Demand", np.nan)

    df_com_imp = df_com_base.copy()
    df_com_imp[["Commodity","Type","price"]] = ("electricity_import", "Buy", 1)

    df_com_feed = df_com_base.copy()
    df_com_feed[["Commodity","Type","price"]] = ("electricity_feed_in", "Sell", 1)

    df_com = pd.concat([df_com_base, df_com_imp, df_com_feed], axis=0)
    return df_com.reset_index(drop=True)

def create_pro_com_elec():
    df_pro_com = pd.DataFrame({
        'Process':   ["import",             "import",       "feed_in",     "feed_in"],
        'Commodity': ["electricity_import", "electricity",  "electricity", "electricity_feed_in"],
        'Direction': ["In",                 "Out",          "In",          "Out"],
        'ratio':     [1,                    1,              1,              1]
    })
    return df_pro_com.reset_index(drop=True)

def create_sto_elec(consumer_bus_list, parameters):
    """Return one generic ``battery_private`` storage row per bus.

    Args:
        consumer_bus_list: Buses that receive the storage.
        parameters: ``technologies.storages["stationary_battery"]`` of the scenario.
    """
    df_sto = pd.DataFrame(consumer_bus_list, columns=['Site'])
    df_sto[["Storage","Commodity","inst-cap-c","cap-up-c","inst-cap-p","cap-up-p","eff-in","eff-out","discharge","ep-ratio",
            "inv-cost-p","inv-cost-c","fix-cost-p","fix-cost-c","var-cost-p","wacc","depreciation"]] = (
            "battery_private", "electricity", parameters["installed_energy_kwh"],
            parameters["capacity_upper_kwh"], parameters["installed_power_kw"], parameters["power_upper_kw"],
            parameters["charge_efficiency"], parameters["discharge_efficiency"],
            parameters["self_discharge_per_timestep"], parameters["energy_to_power_hours"],
            parameters["investment_cost_eur_per_kw"], parameters["investment_cost_eur_per_kwh"],
            parameters["fixed_investment_cost_power_eur"], parameters["fixed_investment_cost_energy_eur"],
            parameters["variable_cost_eur_per_kwh"], parameters["wacc"], parameters["depreciation_years"])
    return df_sto.reset_index(drop=True)