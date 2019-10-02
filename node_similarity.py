import networkx as nx
import os
import pickle
import torch
import torch.nn.functional as F


def calc_feature_similarity(edge_index, x, sim):
    source = x[edge_index[0]]
    target = x[edge_index[1]]
    if sim == 'cosine':
        similarity = F.cosine_similarity(source, target)
    elif sim == 'norm':
        similarity = torch.norm(source - target, 2, dim=1)
    else:
        print(sim, "is not defined")
        exit(1)

    return similarity


def calc_node_similarity(edge_index, sim):
    adj_list = edge_index.numpy().T
    G = nx.Graph()
    G.add_edges_from(adj_list)
    E = []
    if sim == 'neighbors':
        for u, v in adj_list:
            E.append(len([nx.common_neighbors(G, u, v)]))
    elif sim == 'jaccard':
        for u, v in adj_list:
            jac = len([nx.common_neighbors(G, u, v)])
            jac = jac / len(set(G.neighbors(u)) | set(G.neighbors(v)))
            E.append(jac)
    else:
        print(sim, "is not defined")
        exit(1)
    return torch.tensor(E).view(-1, 1)


def init_similarity(data, file_path):
    """
    :param data:
    :param file_path:
    :return:
    """
    print("no pickle file")
    edge_index = data.edge_index
    x = data.x
    similarity_dict = dict()

    # Node similarity
    for s in ["neighbors", "jaccard"]:
        E = calc_node_similarity(edge_index, sim=s)
        similarity_dict[s] = E
        print(s, "done")

    # Feature similarity
    for s in ["cosine", "norm"]:
        E = calc_feature_similarity(edge_index, x, sim=s)
        similarity_dict[s] = E.view(-1, 1)
        print(s, "done")

    with open(file_path, 'wb') as f:
        pickle.dump(similarity_dict, f)


def convert_dict_to_tensor(cent_dict, edge_index):
    E = []
    adj_list = edge_index.numpy().T
    for u, v in adj_list:
        if (u, v) in cent_dict:
            E.append(cent_dict[(u, v)])
        else:
            E.append(cent_dict[(v, u)])
    return torch.tensor(E).view(-1, 1)


def load_similarity(data, name):
    """

    :param dataset:
    :return:
    """
    file_dir = os.path.dirname(os.path.abspath(__file__)) + '/similarity/'
    data_name = name
    file_path = file_dir + data_name + '.pkl'

    if not os.path.exists(file_path):
        os.makedirs(file_dir, exist_ok=True)
        init_similarity(data, file_path)

    with open(file_path, 'rb') as f:
        similarity_dict = pickle.load(f)
        print("finish loading the pkl file")
    return similarity_dict
