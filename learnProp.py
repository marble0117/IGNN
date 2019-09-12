import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.handlers import EarlyStopping
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from utils import accuracy


class EdgeNet1(nn.Module):
    def __init__(self, nefeat):
        super(EdgeNet1, self).__init__()
        self.edge1 = nn.Linear(nefeat, 8)
        self.edge2 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, edge_features):
        E = self.edge1(edge_features)
        E = self.relu(E)
        E = self.dropout(E)
        E = self.edge2(E)
        E = self.sigmoid(E)
        return E


class EdgeNet2(nn.Module):
    def __init__(self, nfeat):
        super(EdgeNet2, self).__init__()
        self.fc1 = nn.Linear(nfeat, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        source, target = edge_index
        feat1 = x[source]
        feat1 = self.relu(self.fc1(feat1))
        feat1 = self.dropout(feat1)
        feat1 = self.relu(self.fc2(feat1))
        feat1 = self.dropout(feat1)
        feat2 = x[target]
        feat2 = self.relu(self.fc1(feat2))
        feat2 = self.dropout(feat2)
        feat2 = self.relu(self.fc2(feat2))
        feat2 = self.dropout(feat2)
        feat = torch.cat((feat1, feat2), dim=1)
        feat = self.fc3(feat)
        feat = self.sigmoid(feat)
        return feat


class Net(MessagePassing):
    def __init__(self, edge_index, nefeat, nvfeat, nclass):
        super(Net, self).__init__()
        self.edge_index = edge_index
        # self.edge_func = EdgeNet1(nefeat)
        self.edge_func = EdgeNet2(nvfeat)
        self.fc1 = nn.Linear(nvfeat, nclass)
        self.th = nn.Threshold(0.5, 0)
        self.ones = torch.ones(edge_index.size()[1], 1)
        self.zeros = torch.zeros(edge_index.size()[1], 1)

    def forward(self, x, edge_features):
        # make a new (sparse) adjacency list
        # E = self.edge_func(edge_features)
        E = self.edge_func(x, self.edge_index)

        # convolution
        # E = torch.where(E > 0.5, self.ones, self.zeros)
        # E = self.th(E)
        x = self.propagate(self.edge_index, size=(x.size(0), x.size(0)), x=x, E=E, aggr='mean')
        x = self.propagate(self.edge_index, size=(x.size(0), x.size(0)), x=x, E=E, aggr='mean')

        # prediction
        x = self.fc1(x)
        return F.log_softmax(x, dim=1), E

    def message(self, x_j, E):
        return x_j * E


def similarity(edge_index, features, sim='sum'):
    if sim == 'sum':
        edge_features = features[edge_index[0]] + features[edge_index[1]]
    elif sim == 'mul':
        edge_features = features[edge_index[0]] * features[edge_index[1]]
    elif sim == 'cat':
        edge_features = torch.cat((features[edge_index[0]], features[edge_index[1]]), dim=1)
    elif sim == 'l1':
        edge_features = torch.abs(features[edge_index[0]] - features[edge_index[1]])
    elif sim == 'l2':
        edge_features = (features[edge_index[0]] - features[edge_index[1]]) ** 2
    else:
        print('invalid sim:', sim)
        exit(-1)
    return edge_features


def learnProp_experiment(edge_index, features, labels, train_mask, val_mask, test_mask, lam1, sim):
    # add self-loops and make edge features
    edge_features = similarity(edge_index, features, sim)

    trainY = labels[train_mask == 1]
    valY = labels[val_mask == 1]
    testY = labels[test_mask == 1]

    net = Net(edge_index, edge_features.size(1), features.size(1), int(max(labels)) + 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    net.train()
    for i in range(30):
        optimizer.zero_grad()
        output, _ = net(features, edge_features)
        train_loss = F.nll_loss(output[train_mask == 1], trainY)
        val_loss = F.nll_loss(output[val_mask == 1], valY)
        val_acc = accuracy(output[val_mask == 1], valY)
        print("epoch:", i + 1, "training loss:", train_loss.item(), "val loss:", val_loss.item(), "val acc :", val_acc)
        l1reg = torch.norm(net.fc1.weight, 1)
        loss = train_loss + lam1 * l1reg
        loss.backward()
        optimizer.step()
    net.eval()
    output, E = net(features, edge_features)
    acc_train = accuracy(output[train_mask == 1], trainY)
    print("train accuracy :", acc_train)
    acc_test = accuracy(output[test_mask == 1], testY)
    print("test  accuracy :", acc_test)
    return E
