import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops

from utils import accuracy
from functions import *


def get_masks(data, train_mask, val_mask, test_mask):
    if train_mask is None:
        train_mask = data.train_mask
    if val_mask is None:
        val_mask = data.val_mask
    if test_mask is None:
        test_mask = data.test_mask
    return train_mask, val_mask, test_mask


def get_labels(data, train_mask, val_mask, test_mask):
    trainY = data.y[train_mask == 1]
    valY = data.y[val_mask == 1]
    testY = data.y[test_mask == 1]
    return trainY, valY, testY


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, heads):
        super(GAT, self).__init__()
        self.gc1 = GATConv(nfeat, nhid, heads=heads, dropout=0.6)
        self.gc2 = GATConv(nhid*heads, nclass, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gc1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)


def run_gat(data, train_mask=None, val_mask=None, test_mask=None,
            early_stopping=True, patience=100, verbose=True):
    edge_index = data.edge_index
    features = data.x
    train_mask, val_mask, test_mask = get_masks(data, train_mask, val_mask, test_mask)
    trainY, valY, testY = get_labels(data, train_mask, val_mask, test_mask)

    model = GAT(features.size()[1], 8, int(torch.max(data.y)) + 1, heads=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    model.train()
    count = 0
    min_val_loss = 10000000
    max_val_acc = 0
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(features, edge_index)
        train_loss = F.nll_loss(output[train_mask == 1], trainY)
        val_loss = F.nll_loss(output[val_mask == 1], valY)
        val_acc = accuracy(output[val_mask == 1], valY)
        if early_stopping:
            if min_val_loss >= val_loss:
                count = 0
                min_val_loss = val_loss
            else:
                count += 1
            if count == patience:
                break
        if verbose:
            print("epoch:", epoch + 1, "training loss:", train_loss.item(), "val loss:", val_loss.item(), "val acc :", val_acc)
        loss = train_loss
        loss.backward()
        optimizer.step()
    model.eval()
    output = model(features, edge_index)
    acc_train = accuracy(output[train_mask == 1], trainY)
    print("train accuracy :", acc_train)
    output = model(features, edge_index)
    acc_test = accuracy(output[test_mask == 1], testY)
    print("test  accuracy :", acc_test)
    return acc_test, output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN3layer(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN3layer, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.gc3 = GCNConv(nhid, nclass)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gc2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gc3(x, edge_index)
        return F.log_softmax(x, dim=1)


def run_gcn(data, train_mask=None, val_mask=None, test_mask=None,
            early_stopping=True, patience=10, verbose=True, edge_weight=None):

    model = GCN(data.x.size()[1], 16, int(torch.max(data.y)) + 1)
    # model = GCN3layer(features.size()[1], 16, int(torch.max(labels)) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    val_loss_history = []
    best_val_loss = float('inf')
    for epoch in range(200):
        train(model, optimizer, data)
        eval_info = evaluate(model, data)
        if eval_info['val_loss'] < best_val_loss:
            best_val_loss = eval_info['val_loss']
            test_acc = eval_info['test_acc']

        val_loss_history.append(eval_info['val_loss'])
        if early_stopping:
            tmp = torch.tensor(val_loss_history[-(patience + 1):-1])
            if eval_info['val_loss'] > tmp.mean().item():
                break
        if verbose:
            print("epoch:", epoch + 1, "training loss:", eval_info['train_loss'], "val loss:", eval_info['val_loss'], "val acc :", eval_info['val_acc'])
    eval_info = evaluate(model, data)
    print("train accuracy :", eval_info['train_acc'])
    print("test  accuracy :", eval_info['test_acc'])
    return eval_info


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output[data.train_mask == 1], data.y[data.train_mask == 1])
    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    outputs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask == 1], data.y[mask == 1]).item()
        pred = logits[mask == 1].max(1)[1]
        acc = pred.eq(data.y[mask == 1]).sum().item() / mask.sum().item()

        outputs['{}_loss'.format(key)] = loss
        outputs['{}_acc'.format(key)] = acc

    return outputs
