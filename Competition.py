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
from sklearn.svm import SVC
from sklearn.svm import SVR
from math import sqrt
import datetime

from sklearn.model_selection import GridSearchCV
# https://stackoverflow.com/questions/3518778/how-do-i-read-csv-data-into-a-record-array-in-numpy
from numpy import genfromtxt


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
    print("Start time: ", datetime.datetime.now())
    test_data = genfromtxt('data_kaggle\posts_test.txt', delimiter=',', skip_header=1)
    train_data = genfromtxt('data_kaggle\posts_train.txt', delimiter=',', skip_header=1)
    graph_data = genfromtxt('data_kaggle\graph.txt', skip_header=1)
    # print("Test Data: ", test_data)
    # print("Train Data: ", train_data)
    # print("Graph Data: ", graph_data)

    # """
    y_tr = train_data[25000:, [4, 5]]
    y_lat_tr = train_data[25000:, 4]
    y_lon_tr = train_data[25000:, 5]
    # print("Training Y: ", y_tr.shape)
    X_tr = train_data[25000:, 1:4]
    X_lon_tr = train_data[25000:, [1, 2, 3, 4]]
    X_lat_tr = train_data[25000:, [1, 2, 3, 5]]
    # print("Training X: ", X_tr.shape)

    X_te = train_data[:25000, 1:4]
    X_lon_te = train_data[:25000, [1, 2, 3, 4]]
    X_lat_te = train_data[:25000, [1, 2, 3, 5]]

    y_te = train_data[:25000, [4, 5]]
    y_lat_te = train_data[:25000, 4]
    y_lon_te = train_data[:25000, 5]
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
        {'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']},
        {'C': [1, 10, 100, 1000], 'coef0': [0.1, 1, 10, 100], 'kernel': ['sigmoid']}
    ]

    clf = GridSearchCV(SVR(), param_grid=params, cv=5)
    # clf = SVR(gamma='auto', kernel='rbf')
    # clf.fit(X_tr, y_tr)
    clf.fit(X_lat_tr, y_lat_tr)
    lat_preds = clf.predict(X_lat_te)

    clf.fit(X_lon_tr, y_lon_tr)
    lon_preds = clf.predict(X_lon_te)

    predictions = np.column_stack((lat_preds, lon_preds))
    error = RMSE(y_te, predictions)

    print("RMSE Error: ", error)

    mseError = sqrt(mean_squared_error(y_te, predictions))
    print("Scikit Lean Error: ", mseError)

    print("End time: ", datetime.datetime.now())
    pass
