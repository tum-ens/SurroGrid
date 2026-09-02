"""Utilities for cleaning and inspecting pandapower LV grid topologies.

Used by the sampling notebooks to make pylovo-exported grids more robust for
subsequent simulations (e.g., avoid 0-length lines, remove duplicate loads,
identify consumer buses). Scenario-load normalization retains one zeroed row
per consumer bus; component categories remain in the component manifest.
"""

import networkx as nx
import matplotlib.pyplot as plt


def draw_grid(df_lines):
    """ Creates an image of the network graph
        Args: net.line dataframe from pandapower network
    """
    # Create an undirected graph
    G = nx.Graph()

    # Add edges to the graph
    for _, row in df_lines.iterrows():
        G.add_edge(row["from_bus"], row["to_bus"])

    pos = nx.spring_layout(G, k=0.05, iterations=120)
    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=100, font_size=12, font_weight='bold', edge_color='gray')
    plt.title("Network Graph of Nodes")
    plt.show()


def assign_min_linelen(net):
    """ Removes bug where some line lengths are 0 which crashes powerflow """
    df_lines = net.line
    df_lines.loc[df_lines["length_km"] == 0.0, "length_km"] = 0.000001
    
    net.line = df_lines
    return net

def normalize_scenario_loads(net):
    """Keep one zeroed topology load row for every demand bus.

    PyLoVo may persist separate category rows for a mixed building on one
    consumer bus.  Step 4 writes dynamic scenario demand once per bus, so the
    exported topology must not retain duplicate static rows.
    """
    df_load = net.load.copy()
    if "bus" not in df_load.columns:
        raise ValueError("Scenario-load normalization requires a bus column.")
    if df_load.empty:
        net.load = df_load
        return net

    sort_columns = ["bus"] + (["name"] if "name" in df_load.columns else [])
    df_load = df_load.sort_values(sort_columns, kind="stable")
    df_load = df_load.drop_duplicates(subset="bus", keep="first").reset_index(drop=True)
    for column in ("p_mw", "q_mvar"):
        if column in df_load.columns:
            df_load[column] = 0.0
    if "scaling" in df_load.columns:
        df_load["scaling"] = 1.0
    if "in_service" in df_load.columns:
        df_load["in_service"] = True
    if "max_p_mw" in df_load.columns:
        df_load["max_p_mw"] = 1000.0
    if "min_p_mw" in df_load.columns:
        df_load["min_p_mw"] = -1000.0
    if df_load["bus"].duplicated().any():
        raise ValueError("Scenario-load normalization left duplicate demand buses.")
    net.load = df_load
    return net

def get_consumers(net):
    df_bus = net.bus["name"].reset_index().rename(columns={"index":"bus"})
    # df_bus["consumer"] = df_bus["name"].str.startswith("Consumer Nodebus")
    df_bus = df_bus[df_bus["name"].str.startswith("Consumer Nodebus")]
    return df_bus
