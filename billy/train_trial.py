import numpy as np
import dill as pickle
import sklearn
from numpy import genfromtxt
from sklearn.svm import SVR
import time
from sklearn.preprocessing import MinMaxScaler

def openFile(fileName):
    raw_data = open(fileName, "rb")
    X_tr, y_tr, X_te = pickle.load(raw_data)
    print(X_tr)

def first_regressor(file_name):
    raw_data = open(file_name, "rb")
    X_tr, y_tr, X_te = pickle.load(raw_data)
    #np.savetxt("foo.csv", X_te, delimiter=",", fmt='%.3e')
    
    scaler = MinMaxScaler()
    X_tr_transformed = scaler.fit_transform(X_tr, y_tr)
    X_te_transformed = scaler.fit_transform(X_te)
    
    start_fit = time.time()
    svm_regressor = SVR(kernel = 'rbf', gamma = 1, C=1000)
    svm_regressor.fit(X_tr_transformed, y_tr[:,0])
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
    first_regressor("total_data.pkl")


