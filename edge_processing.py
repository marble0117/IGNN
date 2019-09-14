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
    def __init__(self, features, edge_index):
        super(EdgeConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, (2, 1))
        self.fc1 = nn.Linear(features.size(1), 16)
        self.fc2 = nn.Linear(16, 1)
        self.fc3 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.feat_ts = convert_feature_to_tensor(features, edge_index)

    def forward(self, x):
        feat = self.relu(self.conv1(self.feat_ts))
        feat = self.relu(self.fc1(feat))
        feat = self.relu(self.fc2(feat))
        feat = torch.reshape(feat, (-1, 8))
        feat = self.sigmoid(self.fc3(feat))
        return feat


def convert_feature_to_tensor(features, edge_index):
    source, target = edge_index
    f1, f2 = features[source], features[target]
    f1 = torch.reshape(f1, (-1, 1, 1, f1.size(1)))
    f2 = torch.reshape(f2, (-1, 1, 1, f2.size(1)))
    feat_ts = torch.cat((f1, f2), dim=2)
    return feat_ts