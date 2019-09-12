import torch
import torch.nn as nn


class EdgeSimNet(nn.Module):
    def __init__(self, nefeat):
        super(EdgeSimNet, self).__init__()
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


class EdgeCatNet(nn.Module):
    def __init__(self, nfeat):
        super(EdgeCatNet, self).__init__()
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


class EdgeConvNet(nn.Module):
    def __init__(self):
        super(EdgeConvNet, self).__init__()
        self.conv1 = nn.Conv2d()


    def forward(self):
        pass
