"""Model features used by an input and the building partition for parallel solves."""

import pandas as pd


def identify_mode(data):
    """Identify the urbs features that the input data require.

    Args:
        data: urbs input dict (see ``input.read_input_h5``).

    Returns:
        dict of bools: ``sto`` (storage), ``bsp`` (buy/sell prices), ``tve``
        (time-variable efficiency), ``tdy`` (type-period weights), ``tsam``
        (time series aggregation requested in ``global_prop``) and ``evs``
        (dedicated EV charging sessions).
    """
    mode = {
        'sto': False,                   # storage
        'bsp': False,                   # buy sell price
        'tve': False,                   # time variable efficiency
        'tdy': False,                   # type periods
        'tsam': False,                  # time series aggregation method
        'evs': False,                   # dedicated EV charging sessions
        }

    if not data['storage'].empty:
        mode['sto'] = True
    if not data['buy_sell_price'].empty:
        mode['bsp'] = True
    if not data['eff_factor'].empty:
        mode['tve'] = True
    if any(data['type_period']['weight_typeperiod'] > 0):
        mode['tdy'] = True
    if data['global_prop'].loc[pd.IndexSlice[:,'tsam'],'value'].iloc[0]:
        mode['tsam'] = True
    # Dedicated EV sessions replace the legacy mobility deadline-demand and
    # mobility-storage path. The *presence* of the session table declares the
    # methodology; its row count only decides whether any constraint is
    # instantiated. A building population with no electric vehicles still uses
    # the dedicated contract.
    mode['thermal'] = 'building_thermal_parameters' in data and not data['building_thermal_parameters'].empty
    if mode['thermal'] and (mode['tsam'] or mode['tdy']):
        raise ValueError('Internal thermal inertia requires chronological inputs; TSAM is unsupported.')
    if 'ev_sessions' in data:
        mode['evs'] = True
    return mode


def get_parallel_building_clusters(data, n_cpu):
    """Split the buildings into ``n_cpu`` contiguous, equally sized clusters.

    The first ``len(buildings) % n_cpu`` clusters receive one extra building;
    empty clusters (``n_cpu`` > number of buildings) are dropped. The partition
    changes the optimization result (each cluster is one model), so keep it
    stable.
    """
    buildings = list(data["site"].index.get_level_values("Name").unique())
    div, mod = divmod(len(buildings), n_cpu)                # div = number of buildings per cluster, mod = remainder of buildings
    clusters = [buildings[i * div + min(i, mod):(i + 1) * div + min(i + 1, mod)] for i in range(n_cpu)]  # slice buildings list s.t. first mod cluster receive one extra building to calculate
    clusters = [lst for lst in clusters if len(lst) > 0]    # remove empty elements which can happen if n_cpu > n_buildings
    print(f"Created {min(n_cpu, len(buildings))} parallel optimization model(s) with up to {len(clusters[0])} nodes per model!")
    return clusters
