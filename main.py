import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops

from allsumSVC import *
from allsumSLP import *
from edge_centrality import *
from gcelw import train_gcelw
import exp_baselines as ex
from exp_vis import test_on_gcn
from functions import *
from learnProp import *
from improvedGCN import *
from models import *
from utils import *


def draw_nx(graph, E, labels):
    graph = add_self_loops(graph)[0]
    edges = list(graph)
    weight = list(E.detach().numpy())
    edgelist = []
    weakedge = []
    strongedge = []
    for i, w in enumerate(weight):
        edgelist.append((int(edges[0][i]), int(edges[1][i]), float(w)))
        if w > 0.5:
            strongedge.append((int(edges[0][i]), int(edges[1][i])))
        else:
            weakedge.append((int(edges[0][i]), int(edges[1][i])))
    G = nx.Graph()
    G.add_weighted_edges_from(edgelist)
    y = list(labels)
    color = []
    for v in list(G):
        color.append(y[v])

    train_node = torch.where(train_mask == 1)[0].tolist()
    pos = nx.spring_layout(G, k=0.04, weight=None)
    # nodes
    # set the size of training nodes to 90
    node_size = [10] * G.number_of_nodes()
    for v in list(G):
        if v in train_node:
            node_size[v] = 90
    nx.draw_networkx_nodes(G, pos, node_color=color, node_size=node_size)
    # edge importance < 0.5
    nx.draw_networkx_edges(G, pos, edgelist=weakedge, edge_color='red', width=2.0)
    # edge importance > 0.5
    nx.draw_networkx_edges(G, pos, edgelist=strongedge, edge_color='black', width=0.3, alpha=0.1)
    plt.show()

    # historgram
    plt.hist(E.detach().numpy(), bins=20, range=(0, 1.0))
    plt.show()


if __name__ == "__main__":
    net_name = 'Cora'
    dataset = Planetoid(root='/tmp/' + net_name, name=net_name)
    data = dataset[0]
    features = data.x
    labels = data.y
    edge_index = add_self_loops(data.edge_index)[0]
    data.edge_index = edge_index
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # train_mask, val_mask, test_mask = divide_dataset(dataset, 80, 500, 1000)

    # E = calc_edge_based_centrality(edge_index, centrality='load')
    # E = calc_node_based_centrality(edge_index, centrality='closeness')
    # svc_experiment(graph, features, labels, train_mask, test_mask, 3)
    # neural_experiment(graph, features, labels, train_mask, test_mask, 3)
    E = learnProp_experiment(data, dataset.name, 'cent')
    # E = improvedGCN(data, dataset.name, 'cent')
    # new_edge_index = eliminate_interclass_edges(edge_index, labels)
    # data.edge_index = new_edge_index

    # runGAT(data)
    # train_gcelw(data)
    # exit()
    # improvedGCN(edge_index, features, labels, train_mask, val_mask, test_mask, sim='cat')

    # print(acc_list)
    # acc_test = test_on_gcn(data, edge_index, E, train_mask, val_mask, test_mask, True)
    # print(acc_test)
    # acc_test = test_on_gcn(data, edge_index, E, train_mask, val_mask, test_mask, False)
    # print(acc_test)

    # eliminate edges on each percentage
    acc_list_imp_avg = []
    acc_list_imp_std = []
    acc_list_un_avg = []
    acc_list_un_std = []
    for ratio in [i * 0.05 for i in range(1, 21)]:
        acc_avg, acc_std = ex.test_on_gcn(data, edge_index, E, train_mask, val_mask, test_mask, True, 10, ratio=ratio)
        print("avg:", acc_avg, "std:", acc_std)
        acc_list_imp_avg.append(acc_avg)
        acc_list_imp_std.append(acc_std)
    for ratio in [i * 0.05 for i in range(1, 21)]:
        acc_avg, acc_std = ex.test_on_gcn(data, edge_index, E, train_mask, val_mask, test_mask, False, 10, ratio=ratio)
        print("avg:", acc_avg, "std:", acc_std)
        acc_list_un_avg.append(acc_avg)
        acc_list_un_std.append(acc_std)

    print(acc_list_imp_avg)
    print(acc_list_imp_std)
    print(acc_list_un_avg)
    print(acc_list_un_std)
    # draw_nx(graph, E, labels)
    plt.hist(E.detach().numpy(), bins=50, range=(0, 1.0))
    plt.show()
