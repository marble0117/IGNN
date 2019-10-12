import matplotlib.pyplot as plt
import networkx as nx
import torch


def draw_classification_result(data, output):
    adj_list = data.edge_index.numpy().T
    G = nx.Graph()
    G.add_edges_from(adj_list)

    labels = data.y
    _, indices = torch.max(output, 1)
    result = (labels == indices)
    wronglist = list((result == False).nonzero().numpy().T[0])

    trainlist = list(data.train_mask.nonzero().numpy().T[0])
    ordered_labels = labels[list(G)]
    color = list(ordered_labels.numpy())

    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, node_color=color, with_labels=False, node_size=40, width=0.5, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=wronglist, node_color='k', node_size=80)
    nx.draw_networkx_nodes(G, pos, nodelist=trainlist, node_color='pink', node_shape='s', node_size=80)
    plt.show()


def draw_nx(graph, E, labels, train_mask):
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

