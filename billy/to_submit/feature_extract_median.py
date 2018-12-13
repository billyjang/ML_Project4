import numpy as np
import dill as pickle
import sklearn
from numpy import genfromtxt
from sklearn.svm import SVR
import time
from sklearn.preprocessing import MinMaxScaler
from operator import add

def testindexing():
    graph_raw_data = open("graph.pkl", "rb")
    graph_data = pickle.load(graph_raw_data)
    print(graph_data)
    
def training_med():
    #need to update revised in gce
    raw_data = open("revised_total_data.pkl", "rb")
    X_total, X_tr, y_tr, X_te, X_te_total = pickle.load(raw_data)
    test = np.array([0])
    print(X_total.shape)
    #remove null island
    graph_raw_data = open("graph.pkl", "rb")
    graph_data = pickle.load(graph_raw_data)
    first_ids = graph_data[:, 0]
    connected_ids = graph_data[:, 1]
    training_ids = X_total[:, 0]
    added_features = []
    count = 0
    length_graph = first_ids.shape[0]
    length_training = training_ids.shape[0]
    for row in X_total:
        id_first = row[0]
        index = np.searchsorted(first_ids, id_first)
        #to_add = [0, 0, 0, 0, 0, 0]
        to_add = []
        while True:
            if index >= length_graph:
                break
            if first_ids[index] == id_first:
                id_second = connected_ids[index]
                index_second = np.searchsorted(training_ids, id_second)
                if index_second >= length_training:
                    break
                if training_ids[index_second] == id_second:
                    #to_add = to_add + X_total[index_second, [1:7]]
                    #to_add = list (map(add, to_add, X_total[index_second, 1:7].tolist()))
                    to_add.append(X_total[index_second, 1:7].tolist())
                else:
                    break
            else:
                break
            index = index + 1
        medians = np.zeros(6)
        if to_add != []:
            np_to_add = np.array(to_add)
            medians = np.median(np_to_add, axis=0)
        added_features.append(medians.tolist())
    total_added_features = np.array(added_features)
    return total_added_features
                

    #decide what to do with friends if friends are all null?
def training_med_test():
    raw_data = open("revised_total_data.pkl", "rb")
    X_total, X_tr, y_tr, X_te, X_te_total = pickle.load(raw_data)

    #remove null island

    graph_raw_data = open("graph.pkl", "rb")
    graph_data = pickle.load(graph_raw_data)

    first_ids = graph_data[:, 0]
    connected_ids = graph_data[:, 1]
    training_ids = X_total[:, 0]
    test_ids = X_te_total[:, 0]

    added_features = []
    count = 0
    length_graph = first_ids.shape[0]
    length_training = training_ids.shape[0]
    for row in X_te_total:
        id_first = row[0]
        index = np.searchsorted(first_ids, id_first)
        #to_add = [0, 0, 0, 0, 0, 0]
        to_add = []
        while True:
            if index >= length_graph:
                break
            if first_ids[index] == id_first:
                id_second = connected_ids[index]
                index_second = np.searchsorted(training_ids, id_second)
                if index_second >= length_training:
                    break
                if training_ids[index_second] == id_second:
                    #to_add = to_add + X_total[index_second, [1:7]]
                    #to_add = list (map(add, to_add, X_total[index_second, 1:7].tolist()))
                    to_add.append(X_total[index_second, 1:7].tolist())
                else:
                    break
            else:
                break
            index = index + 1
        medians = np.zeros(6)
        if to_add != []:
            np_to_add = np.array(to_add)
            medians = np.median(np_to_add, axis=0)
        added_features.append(medians.tolist())
    total_added_features = np.array(added_features)
    return total_added_features

if __name__ == "__main__":
    X_tr_added_features = training_med()
    X_te_added_features = training_med_test()
    full_set = (X_tr_added_features, X_te_added_features)
    with open('added_feat_median.pkl','wb') as f:
        pickle.dump(full_set, f)
