import numpy as np
import dill as pickle
import sklearn
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
import time
import sklearn.model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
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

    total_data = open("total_data.pkl", "rb")
    X_tr, y_tr, X_te = pickle.load(total_data)

    added_raw_features = open("added_feat_median.pkl", "rb")
    added_tr, added_te = pickle.load(added_raw_features)
    
    X_tr = np.concatenate((X_tr, added_tr), axis = 1)
    X_te = np.concatenate((X_te, added_te), axis = 1)

    lis = []
    for i in range(y_tr.shape[0]):
        row = y_tr[i]
        if row[0] == 0.0 and row[1] == 0.0:
            lis.append(i)
    np_lis = np.array(lis)
    X_tr = np.delete(X_tr, np_lis, axis=0)
    y_tr = np.delete(y_tr, np_lis, axis=0)
    
    scaler = sklearn.preprocessing.MinMaxScaler()
    X_tr_transformed = scaler.fit_transform(X_tr, y_tr)
    X_te_transformed = scaler.fit_transform(X_te)
    
    #params for svr gridsearch
    '''
    params = [
        {'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01 ], 'kernel': ['rbf']},
        {'C': [1, 10, 100, 1000], 'degree': [2, 3, 4, 5], 'kernel': ['poly']}
    ]
    '''
    
    #params for mlp 
    
    params = [
            {'hidden_layer_sizes': [(100,), (10, 10), (10, 10, 10)], 'activation': ['logistic', 'relu'], 'solver': ['sgd', 'adam']}
            ]
    

    #clf = GridSearchCV(SVR(), param_grid=params, cv=5, n_jobs = 23, verbose = 7, scoring = 'neg_mean_squared_error')
    clf = GridSearchCV(MLPRegressor(max_iter = 400), params, scoring='neg_mean_squared_error', cv=5, n_jobs=23, verbose=7)

    star = time.time()
    print("grid search starts", star)

    clf.fit(X_tr_transformed, y_tr[:,0])
    lat_preds = clf.predict(X_te_transformed)
    med = time.time()
    print("grid search fits and predicts lat", med)
    print(med - star)
    print("best estimator: ")
    print(clf.best_estimator_.get_params())



    clf.fit(X_tr_transformed, y_tr[:,1])
    lon_preds = clf.predict(X_te_transformed)
    end = time.time()
    print("best esimator: ")
    print(clf.best_estimator_.get_params())

    print("grid search fits and predicts long", end)
    print(end - med)
    np.savetxt("lat_predictions_v04.csv", lat_preds, delimiter=",", fmt='%5.3f')
    np.savetxt("lon_predictions_v04.csv", lon_preds, delimiter=",", fmt='%5.3f')
    predictions = np.column_stack((X_te[:, 0], lat_preds, lon_preds))
    np.savetxt("ltot_predictions_v04.csv", predictions, delimiter=",", fmt='%5.3f')

    error = RMSE(np.column_stack((lat_preds, lon_preds)), predictions)
    fini = time.time()
    print("calc error", fini)
    print(fini - end)
    print("RMSE Error: ", error)

    #mseError = sqrt(mean_squared_error(y_te, predictions))
    #print("Scikit Lean Error: ", mseError)
    
    print("End time: ", datetime.datetime.now())
    pass
