import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.handlers import EarlyStopping
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from learnProp import similarity
from utils import accuracy
from edge_processing import *

class Net(MessagePassing):
    def __init__(self, edge_index, nefeat, nvfeat, nhid, nclass):
        super(Net, self).__init__()
        self.edge_index = edge_index
        self.edge_func = EdgeSimNet(nefeat)
        self.fc1 = nn.Linear(nvfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_features):
        # make a new (sparse) adjacency list
        E = self.edge_func(edge_features)

        # convolution
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.propagate(self.edge_index, size=(x.size(0), x.size(0)), x=x, E=E, aggr='mean')
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
    edge_features = similarity(edge_index, features, sim)

    trainY = labels[train_mask == 1]
    valY = labels[val_mask == 1]
    testY = labels[test_mask == 1]

    net = Net(edge_index, edge_features.size(1), features.size(1), 16, int(max(labels)) + 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    net.train()
    for i in range(100):
        optimizer.zero_grad()
        output = net(features, edge_features)
        train_loss = F.nll_loss(output[train_mask == 1], trainY)
        val_loss = F.nll_loss(output[val_mask == 1], valY)
        val_acc = accuracy(output[val_mask == 1], valY)
        print("epoch:", i + 1, "training loss:", train_loss.item(), "val loss:", val_loss.item(), "val acc :", val_acc)
        loss = train_loss
        loss.backward()
        optimizer.step()
    net.eval()
    output = net(features, edge_features)
    acc_train = accuracy(output[train_mask == 1], trainY)
    print("train accuracy :", acc_train)
    output = net(features, edge_features)
    acc_test = accuracy(output[test_mask == 1], testY)
    print("test  accuracy :", acc_test)
