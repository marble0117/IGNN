import torch


def eliminate_interclass_edges(edge_index, labels):
    source, target = edge_index
    same_lbl = labels[source] == labels[target]
    edge_index = edge_index.T
    new_edge_index = edge_index[same_lbl]
    new_edge_index = new_edge_index.T
    return new_edge_index