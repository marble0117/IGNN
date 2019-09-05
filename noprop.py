import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.handlers import EarlyStopping
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class Net(MessagePassing):
    def __init__(self, edge_index, nvfeat, nclass):
        super(Net, self).__init__()
        self.edge_index = edge_index
        self.fc1 = nn.Linear(nvfeat, nclass)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # convolution
        x = self.propagate(self.edge_index, size=(x.size(0), x.size(0)), x=x, aggr='mean')
        x = self.propagate(self.edge_index, size=(x.size(0), x.size(0)), x=x, aggr='mean')

        # prediction
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def accuracy(pred, labels):
    _, indices = torch.max(pred, 1)
    correct = (indices == labels).sum().item()
    return correct / labels.size()[0]

def noProp_experiment(edge_index, features, labels, train_mask, val_mask, test_mask, lam1):
    # add self-loops and make edge features
    edge_index = add_self_loops(edge_index)[0]

    trainY = labels[train_mask == 1]
    valY = labels[val_mask == 1]
    testY = labels[test_mask == 1]

    net = Net(edge_index,  features.size(1), int(max(labels)) + 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    net.train()
    for i in range(40):
        optimizer.zero_grad()
        output = net(features)
        train_loss = F.nll_loss(output[train_mask == 1], trainY)
        # traval = torch.cat((output[train_mask == 1], output[val_mask == 1]), dim=0)
        # travalY = torch.cat((trainY, valY))
        # train_loss = F.nll_loss(traval, travalY)
        val_loss = F.nll_loss(output[val_mask == 1], valY)
        val_acc = accuracy(output[val_mask == 1], valY)
        print("epoch:", i + 1, "training loss:", train_loss.item(), "val loss:", val_loss.item(), "val acc :", val_acc)
        l1reg = torch.norm(net.fc1.weight, 1)
        loss = train_loss + lam1 * l1reg
        loss.backward()
        optimizer.step()
    net.eval()
    output = net(features)
    acc_train = accuracy(output[train_mask == 1], trainY)
    print("train accuracy :", acc_train)
    acc_test = accuracy(output[test_mask == 1], testY)
    print("test  accuracy :", acc_test)
