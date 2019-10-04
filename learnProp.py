import torch.nn as nn
import torch.nn.functional as F
from ignite.handlers import EarlyStopping
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from edge_processing import *
from utils import accuracy


class Net(MessagePassing):
    def __init__(self, data, name, e_type, sim):
        super(Net, self).__init__()
        if e_type == 'sim':
            self.edge_func = EdgeSimNet(data.edge_index, data.x, sim)
        elif e_type == 'nn':
            self.edge_func = EdgeCatNet(data.edge_index, data.x)
        elif e_type == 'conv':
            self.edge_func = EdgeConvNet(data.edge_index, data.x, n_filt=2, d_out=4)
        elif e_type == 'cent':
            self.edge_func = EdgeCentralityNet(data, name)
        else:
            print("Invalid edge importance calclator:", e_type)
            exit(1)
        self.edge_index = data.edge_index
        self.fc1 = nn.Linear(data.x.size(1), int(data.y.max()) + 1)

    def forward(self, x):
        # make a new (sparse) adjacency list
        E = self.edge_func()

        # convolution
        x = self.propagate(self.edge_index, size=(x.size(0), x.size(0)), x=x, E=E, aggr='mean')
        x = self.propagate(self.edge_index, size=(x.size(0), x.size(0)), x=x, E=E, aggr='mean')

        # prediction
        x = self.fc1(x)
        return F.log_softmax(x, dim=1), E

    def message(self, x_j, E):
        return x_j * E


# def learnProp_experiment(e_type, edge_index, features, labels, train_mask, val_mask, test_mask, sim='cat'):
def learnProp_experiment(data, name, e_type, sim='cat'):
    # add self-loops and make edge features
    features = data.x
    labels = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    trainY = labels[train_mask == 1]
    valY = labels[val_mask == 1]
    testY = labels[test_mask == 1]

    net = Net(data, name, e_type, sim)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    net.train()
    for i in range(30):
        optimizer.zero_grad()
        output, _ = net(features)
        train_loss = F.nll_loss(output[train_mask == 1], trainY)
        val_loss = F.nll_loss(output[val_mask == 1], valY)
        val_acc = accuracy(output[val_mask == 1], valY)
        print("epoch:", i + 1, "training loss:", train_loss.item(), "val loss:", val_loss.item(), "val acc :", val_acc)
        loss = train_loss
        loss.backward()
        optimizer.step()
    output, E = net(features)
    acc_train = accuracy(output[train_mask == 1], trainY)
    print("train accuracy :", acc_train)
    acc_test = accuracy(output[test_mask == 1], testY)
    print("test  accuracy :", acc_test)
    return E
