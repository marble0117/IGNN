import networkx as nx
from networkx.algorithms.centrality import edge_betweenness_centrality\
                                          ,edge_load_centrality\
                                          ,degree_centrality\
                                          ,eigenvector_centrality\
                                          ,closeness_centrality
import torch


def calc_edge_based_centrality(edge_index, centrality='betweenness'):
    adj_list = edge_index.numpy().T
    G = nx.Graph()
    G.add_edges_from(adj_list)
    if centrality == 'betweenness':
        edges_centrality = edge_betweenness_centrality(G)
    elif centrality == 'load':
        edges_centrality = edge_load_centrality(G)
    E = []
    for u, v in adj_list:
        if (u, v) in edges_centrality:
            E.append(edges_centrality[(u, v)])
        else:
            E.append(edges_centrality[(v, u)])
    return torch.tensor(E).view(-1, 1)



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
        print("not defined:", centrality)
        exit(1)
    E = []
    for u, v in adj_list:
        c = nodes_centrality[u] * nodes_centrality[v]
        E.append(c)
    return torch.tensor(E).view(-1, 1)
