from torch_geometric.datasets import Planetoid

from edge_centrality import load_centrality
from node_similarity import load_similarity
from models import *
from utils import *


def test_on_gcn(data, edge_index, E, train_mask, val_mask, test_mask, important, ratio=0.2):
    new_edge_index = eliminate_edges(edge_index, E, ratio=ratio, important=important)
    data.edge_index = new_edge_index
    acc_test = 0
    print("Eliminate important edges:", important)
    print("Ratio:", ratio)
    for _ in range(5):
        print(data.edge_index.size())
        acc_test += runGCN(data, train_mask, val_mask, test_mask, verbose=False)
    return acc_test / 5


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

    # edges_centrality = load_centrality(data, name=dataset.name)
    edges_centrality = load_similarity(data, name=dataset.name)

    all_acc_list_imp = []
    all_acc_list_un = []
    for name, E in edges_centrality.items():
        print(name)
        acc_list_imp = []
        acc_list_un = []
        for ratio in [i * 0.05 for i in range(1, 21)]:
            acc_test = test_on_gcn(data, edge_index, E, train_mask, val_mask, test_mask, important=True, ratio=ratio)
            acc_list_imp.append(acc_test)
        all_acc_list_imp.append(acc_list_imp)
        for ratio in [i * 0.05 for i in range(1, 21)]:
            acc_test = test_on_gcn(data, edge_index, E, train_mask, val_mask, test_mask, important=False, ratio=ratio)
            acc_list_un.append(acc_test)
        all_acc_list_un.append(acc_list_un)

    print('Eliminate important edges')
    for acc_list in all_acc_list_imp:
        print(acc_list)

    print('Eliminate unimpoartant edges')
    for acc_list in all_acc_list_un:
        print(acc_list)
