"""Utilities for cleaning and inspecting pandapower LV grid topologies.

Used by the sampling notebooks to make pylovo-exported grids more robust for
subsequent simulations (e.g., avoid 0-length lines, remove duplicate loads,
identify consumer buses).
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
