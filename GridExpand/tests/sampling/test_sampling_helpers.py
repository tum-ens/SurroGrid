"""Pure helpers of Step 1 (no database)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandapower as pp

from gridexpand.sampling import grid_topol
from gridexpand.sampling.db_read import consumer_bus_frame, impute_missing_occupants


def test_consumer_bus_frame_maps_vertices_to_buses():
    df_bus = pd.DataFrame(
        {"name": ["Transformer", "Consumer Nodebus 17", "Connection Nodebus 3", "Consumer Nodebus 5"]},
        index=[0, 4, 7, 9],
    )
    frame = consumer_bus_frame(df_bus)
    assert frame.to_dict("records") == [{"vertice_id": 17, "bus": 4}, {"vertice_id": 5, "bus": 9}]


def test_missing_occupants_are_imputed_like_db_mode():
    df = pd.DataFrame(
        {
            "residential_floor_area": [120.0, 0.0, 80.0, 90.0],
            "households": [2, 1, 0, 1],
            "occupants": [None, None, None, 3.0],
        }
    )
    out = impute_missing_occupants(df.copy(), mean_household_size=2.0)
    assert out["occupants_imputed"].tolist() == [True, False, False, False]
    assert out["occupants"].tolist()[0] == 4.0
    assert np.isnan(out["occupants"].tolist()[1]) and out["occupants"].tolist()[3] == 3.0


def _net_with_loads():
    net = pp.create_empty_network()
    buses = [pp.create_bus(net, vn_kv=0.4, name=f"Consumer Nodebus {i}") for i in range(3)]
    pp.create_line_from_parameters(net, buses[0], buses[1], length_km=0.0, r_ohm_per_km=0.2, x_ohm_per_km=0.08,
                                   c_nf_per_km=0.0, max_i_ka=0.27)
    pp.create_load(net, buses[1], p_mw=0.01, name="b")
    pp.create_load(net, buses[1], p_mw=0.02, name="a")
    pp.create_load(net, buses[2], p_mw=0.03, name="c")
    return net


def test_zero_line_length_and_one_zeroed_load_per_bus():
    net = grid_topol.assign_min_linelen(_net_with_loads())
    assert net.line["length_km"].tolist() == [0.000001]
    net = grid_topol.normalize_scenario_loads(net)
    assert net.load["bus"].tolist() == [1, 2]
    assert net.load["name"].tolist() == ["a", "c"]
    assert (net.load["p_mw"] == 0.0).all() and (net.load["q_mvar"] == 0.0).all()
    assert grid_topol.get_consumers(net)["bus"].tolist() == [0, 1, 2]
