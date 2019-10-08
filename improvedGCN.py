import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.handlers import EarlyStopping
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from edge_processing import *
from utils import accuracy
from edge_processing import *


class Net(MessagePassing):
    def __init__(self, data, name, e_type, nhid, nclass, sim='cat'):
        super(Net, self).__init__()
        if e_type == 'sim':
            self.edge_func = EdgeSimNet(data.edge_index, data.x, sim)
        elif e_type == 'nn':
            self.edge_func = EdgeCatNet(data.edge_index, data.x)
        elif e_type == 'conv':
            self.edge_func = EdgeConvNet(data.edge_index, data.x, n_filt=2, d_out=4)
        elif e_type == 'cent':
            cent_list = ["betweenness", "eigenvector", "cosine", "degree"]
            self.edge_func = EdgeCentralityNet(data, name, cent_list)
        else:
            print("Invalid edge importance calculator:", e_type)
            exit(1)
        self.edge_index = data.edge_index
        self.fc1 = nn.Linear(data.x.size(1), nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(p=0.25)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # make a new (sparse) adjacency list
        E = self.edge_func()

        # convolution
        x = self.relu(self.fc1(x))
        x = self.propagate(self.edge_index, size=(x.size(0), x.size(0)), x=x, E=E, aggr='mean')
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.propagate(self.edge_index, size=(x.size(0), x.size(0)), x=x, E=E, aggr='mean')
        print(np.histogram(E.detach().numpy()))

        # prediction
        return F.log_softmax(x, dim=1), E

    def message(self, x_j, E):
        return x_j * E


def accuracy(pred, labels):
    _, indices = torch.max(pred, 1)
    correct = (indices == labels).sum().item()
    return correct / labels.size()[0]


def improvedGCN(data, name, e_type):
    features = data.x
    labels = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    trainY = labels[train_mask == 1]
    valY = labels[val_mask == 1]
    testY = labels[test_mask == 1]

    net = Net(data, name, e_type, 8, int(max(labels)) + 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    net.train()
    for i in range(200):
        optimizer.zero_grad()
        output, _ = net(features)
        train_loss = F.nll_loss(output[train_mask == 1], trainY)
        val_loss = F.nll_loss(output[val_mask == 1], valY)
        val_acc = accuracy(output[val_mask == 1], valY)
        print("epoch:", i + 1, "training loss:", train_loss.item(), "val loss:", val_loss.item(), "val acc :", val_acc)
        loss = train_loss
        loss.backward()
        optimizer.step()
    net.eval()
    output, E = net(features)
    acc_train = accuracy(output[train_mask == 1], trainY)
    print("train accuracy :", acc_train)
    acc_test = accuracy(output[test_mask == 1], testY)
    print("test  accuracy :", acc_test)
    return E
