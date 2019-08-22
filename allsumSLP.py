import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input, output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input, output)

    def forward(self, x):
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def accuracy(pred, labels):
    _, indices = torch.max(pred, 1)
    correct = (indices == labels).sum().item()
    return correct / labels.size()[0]


def neural_experiment(graph, features, labels, train_mask, test_mask, num_aggregate, neigs_list=None):
    graph = np.array(graph).T
    G = nx.Graph()
    G.add_nodes_from(list(range(np.max(graph))))
    G.add_edges_from(graph)
    if neigs_list == None:
        neigs_list = []
        for i in list(G):
            neigs_list.append(np.array(list(G.neighbors(i)) + [i]))

    agg_features = features
    trainY = labels[train_mask == 1]
    testY  = labels[test_mask == 1]
    for exp_num in range(1, num_aggregate + 1):
        new_agg = []
        for i in list(G):
            neigs = neigs_list[i]
            neigs_f = torch.mean(agg_features[neigs], dim=0)
            neigs_f = neigs_f.view(-1, neigs_f.shape[0])
            new_agg.append(neigs_f)
        agg_features = torch.cat(new_agg)
        trainX = agg_features[train_mask == 1]
        testX = agg_features[test_mask == 1]
        net = Net(features.shape[1], int(max(labels))+1)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
        for i in range(500):
            optimizer.zero_grad()
            output = net(trainX)
            loss = F.nll_loss(output, trainY)
            loss.backward()
            optimizer.step()
        output = net(trainX)
        acc_train = accuracy(output, trainY)
        print("train accuracy :", acc_train)
        output = net(testX)
        acc_test = accuracy(output, testY)
        print("test  accuracy :", acc_test)
