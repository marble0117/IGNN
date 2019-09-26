import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops

from allsumSVC import *
from allsumSLP import *
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
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # dataset = Planetoid(root='/tmp/Pubmed', name="Pubmed")
    # dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    data = dataset[0]
    features = data.x
    labels = data.y
    edge_index = add_self_loops(data.edge_index)[0]
    data.edge_index = edge_index
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    lam1 = 0


    # train_mask, val_mask, test_mask = divide_dataset(dataset, 80, 500, 1000)

    # svc_experiment(graph, features, labels, train_mask, test_mask, 3)
    # neural_experiment(graph, features, labels, train_mask, test_mask, 3)
    # E = learnProp_experiment("conv", edge_index, features, labels, train_mask, val_mask, test_mask, lam1, sim='cat')
    improvedGCN(edge_index, features, labels, train_mask, val_mask, test_mask, sim='cat')
    # acc_list = []
    # for train_size in [20, 40, 60, 80, 100]:
    #     train_mask, val_mask, test_mask = divide_dataset(dataset, train_size, 500, 1000)
    #     acc_test = 0
    #     for _ in range(5):
    #         acc_test += runGCN(data, train_mask, val_mask, test_mask, verbose=False)
    #     acc_list.append(acc_test / 5)
    # print(acc_list)

    # draw_nx(graph, E, labels)
    # plt.hist(E.detach().numpy(), bins=20, range=(0, 1.0))
    # plt.show()
