import matplotlib.pyplot as plt
import networkx as nx
import torch


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
    print(wrong_neigh)
    print(wrong_same)
    plt.scatter(wrong_same, wrong_neigh)
    plt.xlabel('percentage of the same class neighbors')
    plt.ylabel('percentage of neighbor nodes classified mistakenly')
    plt.show()


