from dgl.data import citation_graph as citegrh
from torch_geometric.datasets import Planetoid
import numpy as np

from allsumSVC import *
from allsumSLP import *
from learnProp import *

if __name__ == "__main__":
    # data = citegrh.load_cora()
    # data = citegrh.load_citeseer()
    # dataset = Planetoid(root='/tmp/Cora', name='Cora')
    dataset = Planetoid(root='/tmp/Pubmed', name="Pubmed")
    # dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    data = dataset[0]
    features = data.x
    labels = data.y
    graph = data.edge_index
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # svc_experiment(graph, features, labels, train_mask, test_mask, 3)
    # neural_experiment(graph, features, labels, train_mask, test_mask, 3)
    learnProp_experiment(graph, features, labels, train_mask, val_mask, test_mask)
