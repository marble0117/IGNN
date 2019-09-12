import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops

from allsumSVC import *
from allsumSLP import *
from learnProp import *
from improvedGCN import *
from noprop import *
from utils import *
from models import *

def draw_nx(graph, Elist, labels, train_mask):
    pos = None
    for E in Elist:
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
        if pos == None:
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
    # dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # dataset = Planetoid(root='/tmp/Pubmed', name="Pubmed")
    # dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    datasets = [Planetoid(root='/tmp/Cora', name='Cora'),
                Planetoid(root='/tmp/Citeseer', name='Citeseer'),
                Planetoid(root='/tmp/Pubmed', name="Pubmed")]
    acc_lists = []
    for dataset in datasets:
        data = dataset[0]
        features = data.x
        labels = data.y
        edge_index = add_self_loops(data.edge_index)[0]
        data.edge_index = edge_index
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        lam1 = 0

        E_sum = learnProp_experiment(edge_index, features, labels, train_mask, val_mask, test_mask, lam1, sim='sum')
        E_mul = learnProp_experiment(edge_index, features, labels, train_mask, val_mask, test_mask, lam1, sim='mul')
        E_cat = learnProp_experiment(edge_index, features, labels, train_mask, val_mask, test_mask, lam1, sim='cat')
        E_l1  = learnProp_experiment(edge_index, features, labels, train_mask, val_mask, test_mask, lam1, sim='l1')

        Elist = [E_sum, E_mul, E_cat, E_l1]

        acc_list = []
        for i, E in enumerate(Elist):
            print("Eliminate important edges")
            new_edge_index = eliminate_edges(edge_index, E, ratio=0.2, important=True)
            data.edge_index = new_edge_index
            acc_test = 0
            for _ in range(5):
                acc_test += runGCN(data, verbose=False)
            acc_list.append(acc_test / 5)

            print("Eliminate not important edges")
            new_edge_index = eliminate_edges(edge_index, E, ratio=0.2, important=False)
            data.edge_index = new_edge_index
            acc_test = 0
            for _ in range(5):
                acc_test += runGCN(data, verbose=False)
            acc_list.append(acc_test / 5)
        acc_lists.append(acc_list)

    names = ["Cora", "Citeseer", "pubmed"]
    for i, acc_list in enumerate(acc_lists):
        print(names[i])
        print(acc_list)