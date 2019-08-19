import numpy as np
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
    if neigs_list == None:
        neigs_list = []
        for i in list(graph):
            neigs_list.append(np.array(list(graph.neighbors(i))))

    agg_features = features
    for exp_num in range(1, num_aggregate + 1):
        new_agg = []
        for i in list(graph):
            neigs = neigs_list[i]
            neigs_f = np.average(agg_features[neigs], axis=0)
            new_agg.append(neigs_f)
        agg_features = np.array(new_agg)
        train_set, train_label = agg_features[train_mask == 1], labels[train_mask == 1]
        test_set, test_label = agg_features[test_mask == 1], labels[test_mask == 1]

        trainX, trainY = torch.from_numpy(train_set).type('torch.FloatTensor'), torch.from_numpy(train_label)
        testX, testY = torch.from_numpy(test_set).type('torch.FloatTensor'), torch.from_numpy(test_label)
        net = Net(features.shape[1], max(labels)+1)
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
