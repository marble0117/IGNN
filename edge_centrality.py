import networkx as nx
from networkx.algorithms.centrality import edge_betweenness_centrality\
                                          ,edge_load_centrality\
                                          ,degree_centrality\
                                          ,eigenvector_centrality\
                                          ,closeness_centrality
import os
import pickle
import torch


def calc_edge_based_centrality(edge_index, centrality='betweenness'):
    adj_list = edge_index.numpy().T
    G = nx.Graph()
    G.add_edges_from(adj_list)
    if centrality == 'betweenness':
        edges_centrality = edge_betweenness_centrality(G)
    elif centrality == 'load':
        edges_centrality = edge_load_centrality(G)
    else:
        print(centrality, "is not defined")
        exit(1)
    return edges_centrality


def calc_node_based_centrality(edge_index, centrality='degree'):
    adj_list = edge_index.numpy().T
    G = nx.Graph()
    G.add_edges_from(adj_list)
    if centrality == 'degree':
        nodes_centrality = degree_centrality(G)
    elif centrality == 'eigenvector':
        nodes_centrality = eigenvector_centrality(G)
    elif centrality == "closeness":
        nodes_centrality = closeness_centrality(G)
    else:
        print(centrality, "is not defined")
        exit(1)

    edges_centrality = dict()
    for u, v in adj_list:
        edges_centrality[(u, v)] = nodes_centrality[u] * nodes_centrality[v]
    return edges_centrality


def init_centrality(data, file_path):
    """
    :param dataset:
    :param file_path:
    :return:
    """
    print("no pickle file")
    edge_index = data.edge_index
    centrality_dict = dict()

    # Edge based centrality
    for c in ["betweenness", "load"]:
        E = calc_edge_based_centrality(edge_index, centrality=c)
        centrality_dict[c] = E
        print(c, 'done')

    # Node based centrality
    for c in ["degree", "eigenvector", "closeness"]:
        E = calc_node_based_centrality(edge_index, centrality=c)
        centrality_dict[c] = E
        print(c, 'done')

    with open(file_path, 'wb') as f:
        pickle.dump(centrality_dict, f)


def convert_dict_to_tensor(cent_dict, edge_index):
    E = []
    adj_list = edge_index.numpy().T
    for u, v in adj_list:
        if (u, v) in cent_dict:
            E.append(cent_dict[(u, v)])
        else:
            E.append(cent_dict[(v, u)])
    return torch.tensor(E).view(-1, 1)


def load_centrality(data, name):
    """

    :param dataset:
    :return:
    """
    file_dir = os.path.dirname(os.path.abspath(__file__)) + '/baselines/'
    data_name = name
    file_path = file_dir + data_name + '.pkl'

    if not os.path.exists(file_path):
        os.makedirs(file_dir, exist_ok=True)
        init_centrality(data, file_path)

    centrality_dict = dict()
    with open(file_path, 'rb') as f:
        cents_dict = pickle.load(f)
        edge_index = data.edge_index
        for cent_name, cent_dict in cents_dict.items():
            centrality_dict[cent_name] = convert_dict_to_tensor(cent_dict, edge_index)
        print("finish loading the pkl file")

    return centrality_dict
