from torch_geometric.datasets import Planetoid

from edge_centrality import *
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

    E_bet = calc_edge_based_centrality(edge_index, centrality='betweenness')
    E_load = calc_edge_based_centrality(edge_index, centrality='load')
    E_deg = calc_node_based_centrality(edge_index, centrality='degree')
    E_eig = calc_node_based_centrality(edge_index, centrality='eigenvector')
    E_cls = calc_node_based_centrality(edge_index, centrality='closeness')

    Elist = [E_bet, E_load, E_deg, E_eig, E_cls]
    # Elist = [E_deg, E_eig, E_cls]

    all_acc_list_imp = []
    all_acc_list_un = []
    for E in Elist:
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
