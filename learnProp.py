import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class Net(MessagePassing):
# class Net(nn.Module):
    def __init__(self, edge_index, nnode, nfeat, nclass):
        super(Net, self).__init__()
        self.edge_index = edge_index
        self.nnode = nnode
        self.edge1 = nn.Linear(nfeat, 16)
        self.edge2 = nn.Linear(16, 1)
        self.fc1 = nn.Linear(nfeat, nclass)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_features):
        # make a new (sparse) adjacency list
        E = self.edge1(edge_features)
        E = self.relu(E)
        E = self.edge2(E)
        E = self.sigmoid(E)
        source = self.edge_index[0][(E >= 0.5)[:, 0]]
        target = self.edge_index[1][(E >= 0.5)[:, 0]]
        new_edges = torch.cat((source.view(-1, source.size(0)), target.view(-1, source.size(0))), dim=0)
        # convolution
        x = self.propagate(new_edges, size=(x.size(0), x.size(0)), x=x)
        x = self.propagate(new_edges, size=(x.size(0), x.size(0)), x=x)
        # prediction
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

    def conv(self, source, target, x, nconv):
        node_index = torch.arange(self.nnode)
        source[source==node_index]
        for _ in range(nconv):
            pass

def accuracy(pred, labels):
    _, indices = torch.max(pred, 1)
    correct = (indices == labels).sum().item()
    return correct / labels.size()[0]

def learnProp_experiment(edge_index, features, labels, train_mask, test_mask):
    # add self-loops and make edge features
    nnode = int(torch.max(edge_index))
    edge_index = add_self_loops(edge_index)[0]
    edge_features = features[edge_index[0]] + features[edge_index[1]]

    trainY = labels[train_mask == 1]
    testY = labels[test_mask == 1]

    net = Net(edge_index, nnode, features.shape[1], int(max(labels)) + 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    net.train()
    for i in range(100):
        optimizer.zero_grad()
        output = net(features, edge_features)
        loss = F.nll_loss(output[train_mask == 1], trainY)
        print("epoch:", i+1, "loss:", loss.item())
        loss.backward()
        optimizer.step()
    net.eval()
    output = net(features, edge_features)
    acc_train = accuracy(output[train_mask == 1], trainY)
    print("train accuracy :", acc_train)
    output = net(features, edge_features)
    acc_test = accuracy(output[test_mask == 1], testY)
    print("test  accuracy :", acc_test)