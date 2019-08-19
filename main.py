from dgl.data import citation_graph as citegrh

from allsumSVC import *
from allsumSLP import *

if __name__ == "__main__":
    # data = citegrh.load_cora()
    data = citegrh.load_citeseer()
    features = data.features
    labels = data.labels
    graph = data.graph
    train_mask = np.array(data.train_mask, dtype=int)
    val_mask = np.array(data.val_mask, dtype=int)
    test_mask = np.array(data.test_mask, dtype=int)

    svc_experiment(graph, features, labels, train_mask, test_mask, 3)
    neural_experiment(graph, features, labels, train_mask, test_mask, 3)