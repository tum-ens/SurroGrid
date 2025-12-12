"""Small utilities for inspecting and cleaning the pandapower grid topology.

These helpers are not required for the standard run, but can be useful when
debugging problematic networks or power-flow convergence issues.

Functions:

- `draw_grid(net)`: visualize the line graph (NetworkX) for quick inspection.
- `assign_min_linelen(net)`: replace zero-length lines to avoid pandapower
    crashes.
- `remove_duplicate_loads(net)`: drop duplicate loads connected to the same
    bus.
- `get_consumers(net)`: list consumer buses by name pattern.
- `get_household_strands(net)`: return indices of first section of household
    cable strands (based on transformer LV bus).
"""

import networkx as nx
import matplotlib.pyplot as plt


def draw_grid(grid):
    """ Creates an image of the network graph
        Args: net.line dataframe from pandapower network
    """
    df_lines = grid.line
    # Create an undirected graph
    G = nx.Graph()

    # Add edges to the graph
    for _, row in df_lines.iterrows():
        G.add_edge(row["from_bus"], row["to_bus"])

    pos = nx.spring_layout(G, k=0.05, iterations=120)
    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=25, font_size=6, font_weight='bold', edge_color='gray')
    plt.title("Network Graph of Nodes")
    plt.show()


def assign_min_linelen(net):
    """ Removes bug where some line lengths are 0 which crashes powerflow """
    df_lines = net.line
    df_lines.loc[df_lines["length_km"] == 0.0, "length_km"] = 0.000001
    
    net.line = df_lines
    return net

def remove_duplicate_loads(net):
    # Remove duplicate households for same building
    df_load = net.load
    df_load = df_load.drop_duplicates(subset="bus", keep="first")
    df_load['name'] = df_load['name'].str.extract(r'Load (\d+)')[0].astype(int)
    
    net.load = df_load
    return net

def get_consumers(net):
    df_bus = net.bus["name"].reset_index().rename(columns={"index":"bus"})
    # df_bus["consumer"] = df_bus["name"].str.startswith("Consumer Nodebus")
    df_bus = df_bus[df_bus["name"].str.startswith("Consumer Nodebus")]
    return df_bus

def get_household_strands(grid):
    trafo_buses = grid.trafo[["hv_bus", "lv_bus"]].values[0]
    ext_grid_bus = int(grid.ext_grid.loc[0, "bus"]) # bus which is the external import bus
    lv_bus = [bus for bus in trafo_buses if bus!=ext_grid_bus][0]

    # Retrieve first section of household cable strands
    hh_strands = grid.line[grid.line["from_bus"]==lv_bus].index.values     # retrieve all household strands

    return hh_strands