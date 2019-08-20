import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


def svc_experiment(graph, features, labels, train_mask, test_mask, num_aggregate, neigs_list=None):
    if neigs_list == None:
        neigs_list = []
        for i in list(graph):
            neigs_list.append(np.array(list(graph.neighbors(i))))
    agg_features = features
    for exp_num in range(1, num_aggregate + 1):
        new_agg = []
        for i in list(graph):
            neigs = neigs_list[i]
            neigs_f = np.average(agg_features[neigs], axis=0)
            new_agg.append(neigs_f)
        agg_features = np.array(new_agg)
        train_set, train_label = agg_features[train_mask == 1], labels[train_mask == 1]
        test_set, test_label = agg_features[test_mask == 1], labels[test_mask == 1]
        clf = LinearSVC(loss='hinge', C=100, max_iter=10000)
        clf.fit(train_set, train_label)
        pre = clf.predict(test_set)
        f1 = f1_score(test_label, pre, average="micro")
        print('f1_micro', f1)
