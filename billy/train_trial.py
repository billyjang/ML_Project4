import numpy as np
import dill as pickle
import sklearn
from numpy import genfromtxt
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

def openFile(fileName):
    raw_data = open(fileName, "rb")
    X_tr, y_tr, X_te = pickle.load(raw_data)
    print(X_tr)

def third_regressor(file_name):
    
def grid_search_ml(file_name):
    raw_data = open(file_name, "rb")
    X_tr, y_tr, X_te = pickle.load(raw_data)
    lis = []
    for i in range(y_tr.shape[0]):
        row = y_tr[i]
        #print(row)
        if row[0] == 0.0 and row[1] == 0.0:
            lis.append(i)
    np_lis = np.array(lis)
    print(X_tr.shape)
    X_tr = np.delete(X_tr, np_lis, axis=0)
    y_tr = np.delete(y_tr, np_lis, axis=0)
    print(X_tr.shape)
    scaler = MinMaxScaler()
    X_tr_transformed = scaler.fit_transform(X_tr, y_tr)
    X_te_transformed = scaler.fit_transform(X_te)

    params = [
        {'hidden_layer_sizes': [(100,100)], 'activation': ['logistic'], 'solver': ['sgd']},
        
    ]
    grid_searcher = GridSearchCV(MLPRegressor(), params, scoring = 'neg_mean_squared_error', cv = 5, verbose = 7)

    #grid_searcher.fit(X_tr_transformed, y_tr)
def first_regressor(file_name):
    raw_data = open(file_name, "rb")
    X_tr, y_tr, X_te = pickle.load(raw_data)
    #np.savetxt("foo.csv", X_te, delimiter=",", fmt='%.3e')

    scaler = MinMaxScaler()
    X_tr_transformed = scaler.fit_transform(X_tr, y_tr)
    X_te_transformed = scaler.fit_transform(X_te)
    
    start_fit = time.time()
    #svm_regressor = SVR(kernel = 'rbf', gamma = 1, C=1000)
    #svm_regressor.fit(X_tr_transformed, y_tr[:,0])
    #mlp_regressor = 
    end_fit = time.time()
    print("fit: ", end_fit - start_fit)

    lat_preds = svm_regressor.predict(X_te_transformed)

    svm_regressor.fit(X_tr_transformed, y_tr[:, 1])
    long_preds = svm_regressor.predict(X_te_transformed)
    #total = np.column_stack((X_tr[:, 0], lat_preds, long_preds))
    np.savetxt("test_long_preds.csv", long_preds, delimiter=",", fmt='%.3e')
    np.savetxt("test_lat_preds.csv", lat_preds, delimiter=",", fmt='%.3e')
    np.savetxt("test_total_preds.csv", total, delimiter=",", fmt='%.3e')
    print(lat_preds.shape)
    print(long_preds.shape)
    print(X_tr[:, 0].shape)

if __name__ == "__main__":
    grid_search_ml("total_data.pkl")


