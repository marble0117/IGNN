import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, nnode, nfeat, nclass):
        super(Net, self).__init__()
        self.P_mat = torch.randn(nnode, nnode)
        self.fc1 = nn.Linear(nfeat, nclass)

    def forward(self, x):
        x = torch.t(x)
        x = x.mm(self.P_mat).mm(self.P_mat)
        x = torch.t(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

def accuracy(pred, labels):
    _, indices = torch.max(pred, 1)
    correct = (indices == labels).sum().item()
    return correct / labels.size()[0]

def learnProp_experiment(graph, features, labels, train_mask, test_mask):
    train_set, train_label = features[train_mask == 1], labels[train_mask == 1]
    test_set, test_label = features[test_mask == 1], labels[test_mask == 1]

    allX = torch.from_numpy(features).type('torch.FloatTensor')
    trainX, trainY = torch.from_numpy(train_set).type('torch.FloatTensor'), torch.from_numpy(train_label)
    testX, testY = torch.from_numpy(test_set).type('torch.FloatTensor'), torch.from_numpy(test_label)
    net = Net(graph.number_of_nodes() , features.shape[1], max(labels) + 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    for i in range(100):
        optimizer.zero_grad()
        output = net(allX)
        loss = F.nll_loss(output[train_mask == 1], trainY)
        loss.backward()
        optimizer.step()
    output = net(allX)
    acc_train = accuracy(output[train_mask == 1], trainY)
    print("train accuracy :", acc_train)
    output = net(allX)
    acc_test = accuracy(output[test_mask == 1], testY)
    print("test  accuracy :", acc_test)