from torch_geometric.datasets import Planetoid

from edge_centrality import load_centrality
from node_similarity import load_similarity
from models import *
from utils import *


def test_on_gcn(data, edge_index, E, train_mask, val_mask, test_mask, important, niter, ratio):
    new_edge_index = eliminate_edges(edge_index, E, ratio=ratio, important=important)
    data.edge_index = new_edge_index
    acc_list = []
    print("Eliminate important edges:", important)
    print("Ratio:", ratio)
    print("Size:", data.edge_index.size())
    for _ in range(niter):
        acc_score = runGCN(data, train_mask, val_mask, test_mask, verbose=False)
        acc_list.append(acc_score)
    return np.mean(acc_list), np.std(acc_list)


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
    print("Dataset:", dataset.name)
    for name, E in edges_centrality.items():
        print(name)
        acc_list_imp_avg = []
        acc_list_imp_std = []
        acc_list_un_avg = []
        acc_list_un_std = []
        for ratio in [i * 0.05 for i in range(1, 21)]:
            acc_avg, acc_std = test_on_gcn(data, edge_index, E, train_mask, val_mask, test_mask, True, 10, ratio=ratio)
            print("avg:", acc_avg, "std:", acc_std)
            acc_list_imp_avg.append(acc_avg)
            acc_list_imp_std.append(acc_std)
        all_acc_list_imp.append(acc_list_imp_avg)
        all_acc_list_imp.append(acc_list_imp_std)
        for ratio in [i * 0.05 for i in range(1, 21)]:
            acc_avg, acc_std = test_on_gcn(data, edge_index, E, train_mask, val_mask, test_mask, False, 10, ratio=ratio)
            print("avg:", acc_avg, "std:", acc_std)
            acc_list_un_avg.append(acc_avg)
            acc_list_un_std.append(acc_std)
        all_acc_list_un.append(acc_list_un_avg)
        all_acc_list_un.append(acc_list_un_std)

    print("similarities:")
    print(edges_centrality.keys())
    print('Eliminate important edges')
    for acc_list in all_acc_list_imp:
        print(acc_list)

    print('Eliminate unimpoartant edges')
    for acc_list in all_acc_list_un:
        print(acc_list)
