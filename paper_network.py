import networkx as nx
import numpy as np


def nodes_selection_ADNI():
    # AD vs. CN
    # connected_nodes = np.array([9, 18, 37, 38, 39, 40, 43, 45, 50, 53, 61, 65, 70, 72, 73, 74, 82, 87]) # 80

    # MCI vs. CN
    # connected_nodes = np.array([3, 5, 11, 15, 20, 27, 30, 32, 41, 42, 43, 47, 48, 49, 59, 66, 73]) # 90

    # AD vs. MCI
    connected_nodes = np.array([0,  9, 10, 16, 47, 54, 55, 62, 64, 70, 72, 82, 84, 86, 87, 88]) # 75
    return connected_nodes
