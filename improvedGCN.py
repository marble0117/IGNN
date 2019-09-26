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
    def __init__(self, e_type, edge_index, features, nhid, nclass, sim):
        super(Net, self).__init__()
        if e_type == 'sim':
            self.edge_func = EdgeSimNet(edge_index, features, sim)
        elif e_type == 'nn':
            self.edge_func = EdgeCatNet(edge_index, features)
        elif e_type == 'conv':
            self.edge_func = EdgeConvNet(edge_index, features, n_filt=2, d_out=4)
        else:
            print("Invalid edge importance calculator:", e_type)
            exit(1)
        self.edge_index = edge_index
        self.fc1 = nn.Linear(features.size(1), nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(p=0.5)
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

        # prediction
        return F.log_softmax(x, dim=1)

    def message(self, x_j, E):
        return x_j * E


def accuracy(pred, labels):
    _, indices = torch.max(pred, 1)
    correct = (indices == labels).sum().item()
    return correct / labels.size()[0]


def improvedGCN(edge_index, features, labels, train_mask, val_mask, test_mask, sim):

    trainY = labels[train_mask == 1]
    valY = labels[val_mask == 1]
    testY = labels[test_mask == 1]

    e_type = "conv"
    net = Net(e_type, edge_index, features, 8, int(max(labels)) + 1, sim)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    net.train()
    for i in range(100):
        optimizer.zero_grad()
        output = net(features)
        train_loss = F.nll_loss(output[train_mask == 1], trainY)
        val_loss = F.nll_loss(output[val_mask == 1], valY)
        val_acc = accuracy(output[val_mask == 1], valY)
        print("epoch:", i + 1, "training loss:", train_loss.item(), "val loss:", val_loss.item(), "val acc :", val_acc)
        loss = train_loss
        loss.backward()
        optimizer.step()
    net.eval()
    output = net(features)
    acc_train = accuracy(output[train_mask == 1], trainY)
    print("train accuracy :", acc_train)
    output = net(features)
    acc_test = accuracy(output[test_mask == 1], testY)
    print("test  accuracy :", acc_test)
