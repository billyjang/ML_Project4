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

from sklearn.model_selection import GridSearchCV
# https://stackoverflow.com/questions/3518778/how-do-i-read-csv-data-into-a-record-array-in-numpy
from numpy import genfromtxt

def RMSE(true_values, predictions=None, lat_preds=None, lon_preds=None):
    sum = 0
    n = len(true_values)
    for index in range(0, n):
        sum += (true_values[0] - predictions[0]) ** 2 + (true_values[1] - predictions[1]) ** 2)
    return sum / (2 * n)

if __name__ == "__main__":
    test_data = genfromtxt('data_kaggle\posts_test.txt', delimiter=',', skip_header=1)
    train_data = genfromtxt('data_kaggle\posts_train.txt', delimiter=',', skip_header=1)
    graph_data = genfromtxt('data_kaggle\graph.txt', skip_header=1)
    # print("Test Data: ", test_data)
    # print("Train Data: ", train_data)
    # print("Graph Data: ", graph_data)

    y_tr = train_data[10000:, [4, 5]]
    y_lat_tr = train_data[10000:, 4]
    print("Training Y: ", y_tr.shape)
    X_tr = train_data[10000:, 1:4]
    print("Training X: ", X_tr.shape)

    X_te = train_data[:10000, 1:4]
    y_te = train_data[:10000, [4, 5]]
    y_lat_te = train_data[:10000, 4]

    params = [
        {'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']},
        {'C': [1, 10, 100, 1000], 'coef0': [0.1, 1, 10, 100], 'kernel': ['sigmoid']}
    ]

    # clf = GridSearchCV(SVR(gamma="auto"), param_grid=params, cv=5, scoring='f1')
    clf = SVR(gamma='auto', kernel='rbf')
    # clf.fit(X_tr, y_tr)
    clf.fit(X_tr, y_lat_tr)
    predictions = clf.predict(X_te)
    mse = mean_squared_error(y_lat_te, predictions)
    print("Mean Squared Error: ", mse)
    pass
