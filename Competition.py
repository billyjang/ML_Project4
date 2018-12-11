import numpy as np
import dill as pickle
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from math import sqrt
import datetime

from sklearn.model_selection import GridSearchCV
# https://stackoverflow.com/questions/3518778/how-do-i-read-csv-data-into-a-record-array-in-numpy
from numpy import genfromtxt
from bisect import bisect_left


def binary_search(a, x, lo=0, hi=None):  # can't use a to specify default for hi
    hi = hi if hi is not None else len(a)  # hi defaults to len(a)
    pos = bisect_left(a, x, lo, hi)  # find insertion position
    return pos if pos != hi and a[pos] == x else -1  # don't walk off the end


def generate_average_location(train_data, test_data, graph, flat_graph):
    to_return = []
    edge_pointer = 0
    flat_train = train_data[:, 0].flatten()
    # print("flat_train: ", flat_train)
    # isequal = train_data == test_data
    for datum in test_data:
        # print("Currently on data id: ", datum[0])
        sum_latitude = 0
        sum_longitude = 0
        n = 0
        start_index = np.searchsorted(flat_graph, datum[0], side='left')
        end_index = np.searchsorted(flat_graph, datum[0], side='right')

        for index in range(start_index, end_index):
            if graph[index][0] == datum[0]:
                train_index = np.searchsorted(flat_train, graph[index][1], side='left')
                """
                print("-------------------------------")
                print("Test Data index: ", datum[0])
                print("Graph index 0: ", graph[index][0])
                print("Train index: ", train_index)
                print("Graph index 1: ", graph[index][1])
                print("Train Data Index: ", train_data[train_index][0])
                print("Friend Latitude: ", train_data[train_index][4])
                print("Friend Longitude: ", train_data[train_index][5])
                print("-------------------------------")
                # """
                if train_index < len(train_data) and train_data[train_index][0] == graph[index][1]:
                    # print(" adding new data")
                    sum_latitude += train_data[train_index][4]
                    sum_longitude += train_data[train_index][5]
                    n += 1
        avg_latitude = 0
        avg_longitude = 0
        if n != 0:
            # print("N is not 0")
            avg_latitude = sum_latitude / n
            avg_longitude = sum_longitude / n
        to_return.append(np.append(datum, [avg_latitude, avg_longitude]))
    # print("To Return: ")
    # print(to_return)
    return np.array(to_return)
    pass


def remove_nulls(data):
    delete_array = []
    to_return = data
    for index in range(0, len(data)):
        if data[index][4] == 0 and data[index][5] == 0:
            delete_array.append(index)
    for delete_index in reversed(delete_array):
        to_return = np.delete(to_return, delete_index, 0)
    return to_return
    pass


def RMSE(true_values, predictions=None, lat_preds=None, lon_preds=None):
    print("In Error Part")
    sum = 0
    n = len(true_values)
    # print("True Values: ", true_values)
    # print("Predictions: ", predictions)
    for index in range(0, n):
        sum += ((true_values[index][0] - predictions[index][0]) ** 2
                + (true_values[index][1] - predictions[index][1]) ** 2)
    return sqrt(sum / (2 * n))


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("Start time: ", start_time)
    test_data = genfromtxt('data_kaggle\posts_test.txt', delimiter=',', skip_header=1)
    train_data = genfromtxt('data_kaggle\posts_train.txt', delimiter=',', skip_header=1)
    graph_data = genfromtxt('data_kaggle\graph.txt', skip_header=1)
    # print("Test Data: ", test_data)
    # print("Train Data: ", train_data)
    # print("Graph Data: ", graph_data)
    train_data = remove_nulls(train_data)

    zero_test_flag = False
    zero_eliminated_test_array = np.copy(train_data)
    print("Zero Test: ")
    for test_element in zero_eliminated_test_array:
        if test_element[4] == 0 and test_element[5] == 0:
            print("Warning, null island detected")
            zero_test_flag = True
    if not zero_test_flag:
        print("All clear, null island removed!")
    # """
    y_tr = train_data[25000:, [4, 5]]
    y_lat_tr = train_data[25000:, 4]
    y_lon_tr = train_data[25000:, 5]
    # print("Training Y: ", y_tr.shape)
    X_tr = train_data[25000:, :]
    # X_lon_tr = train_data[25000:, [1, 2, 3, 4]]
    # X_lat_tr = train_data[25000:, [1, 2, 3, 5]]
    # print("Training X: ", X_tr.shape)

    X_te = train_data[:25000, :]
    # X_lon_te = train_data[:25000, [1, 2, 3, 4]]
    # X_lat_te = train_data[:25000, [1, 2, 3, 5]]

    flat_graph = graph_data[:, 0].flatten()
    X_tr = generate_average_location(X_tr, X_tr, graph_data, flat_graph)
    X_te = generate_average_location(X_tr, X_te, graph_data, flat_graph)
    X_tr = X_tr[:, [1, 2, 3, 7, 8]]
    print("New Training Data: ")
    print(X_tr)
    X_te = X_te[:, [1, 2, 3, 7, 8]]
    print("New Testing Data: ")
    print(X_te)

    #####################################
    # Split into friends and no friends #
    #####################################
    X_tr_friends = []
    X_tr_nofriends = []
    X_te_friends = []
    X_te_nofriends = []

    y_tr_lat_friends = []
    y_tr_lat_nofriends = []
    y_tr_lon_friends = []
    y_tr_lon_nofriends = []

    y_te_lat_friends = []
    y_te_lat_nofriends = []
    y_te_lon_friends = []
    y_te_lon_nofriends = []

    y_te = train_data[:25000, [4, 5]]

    for index in range(0, len(X_tr)):
        row = X_tr[index]
        y_row = y_tr[index]
        if row[3] == 0 and row[4] == 0:
            X_tr_nofriends.append(row)
            y_tr_lat_nofriends.append(y_row[0])
            y_tr_lon_nofriends.append(y_row[1])
        else:
            X_tr_friends.append(row)
            y_tr_lat_friends.append(y_row[0])
            y_tr_lon_friends.append(y_row[1])
    for index in range(0, len(X_te)):
        row = X_te[index]
        y_row = y_te[index]
        # print("X_te row: ", row)
        if row[3] == 0 and row[4] == 0:
            X_te_nofriends.append(row)
            y_te_lat_nofriends.append(y_row[0])
            y_te_lon_nofriends.append(y_row[1])
        else:
            # print("We have friends!")
            X_te_friends.append(row)
            y_te_lat_friends.append(y_row[0])
            y_te_lon_friends.append(y_row[1])

    X_tr_friends = np.array(X_tr_friends)
    X_tr_nofriends = np.array(X_tr_nofriends)
    X_te_friends = np.array(X_te_friends)
    X_te_nofriends = np.array(X_te_nofriends)

    y_tr_lat_friends = np.array(y_tr_lat_friends)
    y_tr_lat_nofriends = np.array(y_tr_lat_nofriends)
    y_tr_lon_friends = np.array(y_tr_lon_friends)
    y_tr_lon_nofriends = np.array(y_tr_lon_nofriends)

    y_te_lat_friends = np.array(y_te_lat_friends)
    y_te_lat_nofriends = np.array(y_te_lat_nofriends)
    y_te_lon_friends = np.array(y_te_lon_friends)
    y_te_lon_nofriends = np.array(y_te_lon_nofriends)
    print("There are " + str(len(X_tr_friends)) + " training points with friends")
    print("There are " + str(len(X_tr_nofriends)) + " training points with no friends")
    print("There are " + str(len(X_te_friends)) + " testing points with friends")
    print("There are " + str(len(X_te_nofriends)) + " testing points with no friends")

    """
    scaler = MinMaxScaler()
    scaler.fit(X_tr)
    scaled_X_tr = scaler.transform(X_tr)
    scaler.fit(X_te)
    scaled_X_te = scaler.transform(X_te)
    """
    scaler = MinMaxScaler()
    scaler.fit(X_tr_friends)
    scaled_X_tr_friends = scaler.transform(X_tr_friends)
    scaler.fit(X_tr_nofriends)
    scaled_X_tr_nofriends = scaler.transform(X_tr_nofriends)

    scaler.fit(X_te_friends)
    scaled_X_te_friends = scaler.transform(X_te_friends)
    scaler.fit(X_tr_nofriends)
    scaled_X_te_nofriends = scaler.transform(X_te_nofriends)

    # y_te = train_data[:25000, [4, 5]]
    # y_lat_te = train_data[:25000, 4]
    # y_lon_te = train_data[:25000, 5]
    # """

    """
    y_tr = train_data[100:200, [4, 5]]
    y_lat_tr = train_data[100:200, 4]
    y_lon_tr = train_data[100:200, 5]
    print("Training Y: ", y_tr.shape)
    X_tr = train_data[100:200, 1:4]
    print("Training X: ", X_tr.shape)

    X_te = train_data[:10, 1:4]
    y_te = train_data[:10, [4, 5]]
    y_lat_te = train_data[:10, 4]
    y_lon_te = train_data[:10, 5]
    """

    params = [
        # {'C': [1, 10, 100], 'gamma': [1, 0.1], 'kernel': ['rbf', 'poly', 'sigmoid']}
        {'C': [1000], 'kernel':['rbf'], 'gamma': [1]}
    ]

    clf = GridSearchCV(SVR(), param_grid=params, cv=5, n_jobs=-1)
    # clf = SVR(gamma='auto', kernel='rbf')

    ###################
    # Testing by Hand #
    ###################
    print("Starting regression at time: ", datetime.datetime.now())
    """
    svr_kernel = 'rbf'
    svr_gamma = 'scale'
    clf = SVR(gamma=svr_gamma, kernel=svr_kernel, C=10)
    print("SVR with " + svr_kernel + " kernel and " + svr_gamma + " gamma")
    """
    """
    # clf.fit(X_tr, y_tr)
    clf.fit(scaled_X_tr, y_lat_tr)
    lat_preds = clf.predict(scaled_X_te)
    print("Latitude best regressor: ")
    print(clf.cv_results_)

    clf.fit(scaled_X_tr, y_lon_tr)
    lon_preds = clf.predict(scaled_X_te)
    print("Longitude best regressor: ")
    print(clf.cv_results_)
    """

    clf.fit(scaled_X_tr_friends, y_tr_lat_friends)
    lat_preds = clf.predict(scaled_X_te_friends)
    print("Latitude best regressor: ")
    print(clf.cv_results_)

    clf.fit(scaled_X_tr_friends, y_tr_lon_friends)
    lon_preds = clf.predict(scaled_X_te_friends)
    print("Longitude best regressor: ")
    print(clf.cv_results_)
    # """
    predictions = np.column_stack((lat_preds, lon_preds))
    y_te_friends = np.column_stack((y_te_lat_friends, y_te_lon_friends))
    error = RMSE(y_te_friends, predictions)

    ##############
    # Gridsearch #
    ##############

    print("RMSE Error: ", error)

    mseError = sqrt(mean_squared_error(y_te_friends, predictions))
    print("Scikit Learn Error: ", mseError)
    # """
    end_time = datetime.datetime.now()
    print("End time: ", end_time)
    print("Total Time Difference: ", end_time - start_time)

    pass
