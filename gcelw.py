import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from utils import accuracy


class GCELW(MessagePassing):
    def __init__(self, data, nhid):
        super(GCELW, self).__init__()
        self.edge_index = data.edge_index
        self.fc1 = nn.Linear(data.x.size(1), nhid)
        self.fc2 = nn.Linear(nhid, int(data.y.max())+1)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.E = torch.ones((data.edge_index.size(1), 1))

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.propagate(self.edge_index, size=(x.size(0), x.size(0)), x=x, E=self.E, aggr='mean')
        x = self.relu(self.fc2(x))
        x = self.propagate(self.edge_index, size=(x.size(0), x.size(0)), x=x, E=self.E, aggr='mean')
        return F.log_softmax(x, dim=1)

    def message(self, x_j, E):
        return x_j * E


def train_gcelw(data):
    features = data.x
    labels = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    trainY = labels[train_mask == 1]
    valY = labels[val_mask == 1]
    testY = labels[test_mask == 1]

    net = GCELW(data, nhid=16)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    net.train()

    for i in range(200):
        optimizer.zero_grad()
        output = net(features)
        loss = F.nll_loss(output[train_mask == 1], trainY)
        val_loss = F.nll_loss(output[val_mask == 1], valY)
        val_acc = accuracy(output[val_mask == 1], valY)
        print(loss, val_loss, val_acc)
        loss.backward()
        optimizer.step()

    net.eval()
    output = net(features)
    acc_train = accuracy(output[train_mask == 1], trainY)
    print("train accuracy :", acc_train)
    acc_test = accuracy(output[test_mask == 1], testY)
    print("test  accuracy :", acc_test)
