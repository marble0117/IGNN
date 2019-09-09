import torch

def eliminate_edges(edge_index, labels):
    source, target = edge_index
    same_lbl = labels[source] == labels[target]
    edge_index = edge_index.T
    edge_index = edge_index[same_lbl]
    edge_index = edge_index.T
    return edge_index