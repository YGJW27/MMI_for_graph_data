import networkx as nx
import numpy as np


def nodes_selection_ADNI():
    # AD vs. CN
    # connected_nodes = np.array([31, 35, 36, 38, 39, 40, 53, 61, 62, 65, 70, 82, 84, 86, 87, 89]) # 75

    # MCI vs. CN
    # connected_nodes = np.array([4, 11, 15, 19, 20, 27, 32, 37, 44, 47, 48, 67, 69, 71, 73, 75]) # 96

    # AD vs. MCI
    connected_nodes = np.array([0,  9, 10, 16, 47, 54, 55, 62, 64, 70, 72, 82, 84, 86, 87, 88]) # 75
    return connected_nodes
