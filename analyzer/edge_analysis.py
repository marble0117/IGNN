import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops

from edge_centrality import load_centrality
from node_similarity import load_similarity


def divide_edges_by_class(data):
    source, target = data.edge_index
    labels = data.y
    same_labels = labels[source] == labels[target]
    diff_labels = labels[source] != labels[target]
    edge_index = data.edge_index.T
    inner_edges = edge_index[same_labels]
    inter_edges = edge_index[diff_labels]
    return inner_edges.T, inter_edges.T


def make_histogram(edges_dict, same_labels, diff_labels):
    for name, cent in edges_dict.items():
        same_cent = cent[same_labels].numpy()
        diff_cent = cent[diff_labels].numpy()
        plt.hist(same_cent, bins=50, label='same')
        plt.hist(diff_cent, bins=50, label='diff')
        plt.title(name)
        plt.show()


def make_scatter(edges_dict, same_labels, diff_labels):
    name_list = list(edges_dict.keys())
    cent_list = list(edges_dict.values())
    ntype = len(name_list)
    for i in range(ntype-1):
        name1 = name_list[i]
        cent1 = cent_list[i]
        cent1 = cent1 / torch.max(cent1)
        cent1_same = cent1[same_labels].numpy()
        cent1_diff = cent1[diff_labels].numpy()
        for j in range(i+1, ntype):
            name2 = name_list[j]
            cent2 = cent_list[j]
            cent2 = cent2 / torch.max(cent2)
            cent2_same = cent2[same_labels].numpy()
            cent2_diff = cent2[diff_labels].numpy()
            plt.scatter(cent1_same, cent2_same)
            plt.scatter(cent1_diff, cent2_diff)
            plt.xlabel(name1)
            plt.ylabel(name2)
            plt.title(name1 + ", " + name2)
            plt.show()


def analyze_edge_statistics():
    net_name = 'Citeseer'
    dataset = Planetoid(root='/tmp/' + net_name, name=net_name)
    data = dataset[0]
    edge_index = add_self_loops(data.edge_index)[0]
    data.edge_index = edge_index
    edges_centrality = load_centrality(data, name=dataset.name)
    edges_similarity = load_similarity(data, name=dataset.name)
    source, target = edge_index
    labels = data.y
    same_labels = labels[source] == labels[target]
    diff_labels = labels[source] != labels[target]


    # make_histogram(edges_centrality, same_labels, diff_labels)
    # make_histogram(edges_similarity, same_labels, diff_labels)
    make_scatter({**edges_centrality, **edges_similarity}, same_labels, diff_labels)
    make_scatter(edges_similarity, same_labels, diff_labels)

def main():
    analyze_edge_statistics()


main()
