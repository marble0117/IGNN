import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.handlers import EarlyStopping
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class Net(MessagePassing):
    def __init__(self, edge_index, nefeat, nvfeat, nclass):
        super(Net, self).__init__()
        self.edge_index = edge_index
        self.edge1 = nn.Linear(nefeat, 8)
        self.edge2 = nn.Linear(8, 1)
        self.fc1 = nn.Linear(nvfeat, nclass)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.th = nn.Threshold(0.5, 0)
        self.ones = torch.ones(edge_index.size()[1], 1)
        self.zeros = torch.zeros(edge_index.size()[1], 1)

    def forward(self, x, edge_features):
        # make a new (sparse) adjacency list
        E = self.edge1(edge_features)
        E = self.relu(E)
        E = self.dropout(E)
        E = self.edge2(E)
        E = self.sigmoid(E)

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


def accuracy(pred, labels):
    _, indices = torch.max(pred, 1)
    correct = (indices == labels).sum().item()
    return correct / labels.size()[0]

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
    edge_index = add_self_loops(edge_index)[0]
    edge_features = similarity(edge_index, features, sim)
    print(edge_features.size())

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
    output, E = net(features, edge_features)
    acc_train = accuracy(output[train_mask == 1], trainY)
    print("train accuracy :", acc_train)
    acc_test = accuracy(output[test_mask == 1], testY)
    print("test  accuracy :", acc_test)
    return E
