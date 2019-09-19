import torch
import torch.nn as nn


class EdgeSimNet(nn.Module):
    def __init__(self, edge_index, features, sim='cat'):
        super(EdgeSimNet, self).__init__()
        self.edge_features = self.get_similarity(edge_index, features, sim)
        self.edge1 = nn.Linear(self.edge_features.size(1), 8)
        self.edge2 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        print("This program is using EdgeSimNet with", sim, "as the similarity")

    def forward(self):
        E = self.edge1(self.edge_features)
        E = self.relu(E)
        E = self.dropout(E)
        E = self.edge2(E)
        E = self.sigmoid(E)
        return E

    def get_similarity(self, edge_index, features, sim='cat'):
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


class EdgeCatNet(nn.Module):
    def __init__(self, edge_index, features):
        super(EdgeCatNet, self).__init__()
        self.edge_index = edge_index
        self.features = features
        self.fc1 = nn.Linear(features.size(1), 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        print("This program is using NN to calculate the edge importance")

    def forward(self):
        source, target = self.edge_index
        feat1 = self.features[source]
        feat1 = self.relu(self.fc1(feat1))
        feat1 = self.dropout(feat1)
        feat1 = self.relu(self.fc2(feat1))
        feat1 = self.dropout(feat1)
        feat2 = self.features[target]
        feat2 = self.relu(self.fc1(feat2))
        feat2 = self.dropout(feat2)
        feat2 = self.relu(self.fc2(feat2))
        feat2 = self.dropout(feat2)
        feat = torch.cat((feat1, feat2), dim=1)
        feat = self.fc3(feat)
        feat = self.sigmoid(feat)
        return feat


class EdgeConvNet(nn.Module):
    def __init__(self, edge_index, features, n_filt=2, d_out=4):
        super(EdgeConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, n_filt, (2, 1))
        self.fc1 = nn.Linear(features.size(1), 16)
        self.fc2 = nn.Linear(16, d_out)
        self.fc3 = nn.Linear(n_filt * d_out, 1)
        self.reshape = lambda feat: torch.reshape(feat, (-1, n_filt * d_out))
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.feat_ts = convert_feature_to_tensor(features, edge_index)
        print("This program is using Conv layer with n_filt =", n_filt,
              "d_out =", d_out,"to calculate the edge importance")

    def forward(self):
        feat = self.relu(self.conv1(self.feat_ts))
        feat = self.relu(self.fc1(feat))
        feat = self.relu(self.fc2(feat))
        feat = self.reshape(feat)
        feat = self.sigmoid(self.fc3(feat))
        return feat


def convert_feature_to_tensor(features, edge_index):
    source, target = edge_index
    f1, f2 = features[source], features[target]
    f1 = torch.reshape(f1, (-1, 1, 1, f1.size(1)))
    f2 = torch.reshape(f2, (-1, 1, 1, f2.size(1)))
    feat_ts = torch.cat((f1, f2), dim=2)
    return feat_ts
