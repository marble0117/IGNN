import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F


def check_feature_similarity(data, output):
    train_node_feature = data.x[data.train_mask == 1]
    train_node_class = data.y[data.train_mask == 1]
    _, indices = torch.max(output, 1)
    result = (data.y == indices)
    wronglist = list((result == False).nonzero().numpy().T[0])
    correctlist = list((result == True).nonzero().numpy().T[0])

    nclass = int(torch.max(data.y)) + 1

    correct_avg_list = [[] for _ in range(nclass)]
    for node in correctlist:
        c = data.y[node]
        f = data.x[node]
        train_feature_same_class = train_node_feature[train_node_class == c]
        sim = F.cosine_similarity(train_feature_same_class, f.view(1, -1))
        correct_avg_list[c].append(float(torch.mean(sim)))

    wrong_avg_list = [[] for _ in range(nclass)]
    for node in wronglist:
        c = data.y[node]
        f = data.x[node]
        train_feature_same_class = train_node_feature[train_node_class == c]
        sim = F.cosine_similarity(train_feature_same_class, f.view(1, -1))
        wrong_avg_list[c].append(float(torch.mean(sim)))
    print("wrong nodes")
    for c in range(nclass):
        print("class", c)
        correct_array = np.array(correct_avg_list[c])
        wrong_array = np.array(wrong_avg_list[c])
        print("correct mean", np.mean(correct_array))
        print("wrong   mean", np.mean(wrong_array))
        print("correct sim=0", np.count_nonzero(correct_array == 0) / correct_array.shape[0])
        print("wrong   sim=0", np.count_nonzero(wrong_array == 0) / wrong_array.shape[0])
        print("correct max", np.max(correct_array), "correct min", np.min(correct_array))
        print("wrong max", np.max(wrong_array), "wrong min", np.min(wrong_array))


def check_neighbor_class(data, output):
    adj_list = data.edge_index.numpy().T
    G = nx.Graph()
    G.add_edges_from(adj_list)

    _, indices = torch.max(output, 1)
    result = (data.y == indices)
    wronglist = list((result == False).nonzero().numpy().T[0])

    result = list(result.numpy())
    wrong_same = []
    wrong_neigh = []
    for node in wronglist:
        neigh = list(G.neighbors(node))
        nneigh = len(neigh)
        wrongcount = 0
        samecount = 0
        for v in neigh:
            if result[v]:
                wrongcount += 1
            if data.y[node] == data.y[v]:
                samecount += 1
        wrong_neigh.append(wrongcount / nneigh)
        wrong_same.append(samecount / nneigh)
    plt.scatter(wrong_same, wrong_neigh)
    plt.xlabel('percentage of the same class neighbors')
    plt.ylabel('percentage of neighbor nodes classified mistakenly')
    plt.show()


