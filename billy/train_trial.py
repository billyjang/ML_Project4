import numpy as np
import dill as pickle
import sklearn
from numpy import genfromtxt
from sklearn.svm import SVR
import time

def openFile(fileName):
    raw_data = open(fileName, "rb")
    X_tr, y_tr, X_te = pickle.load(raw_data)
    print(X_tr)

def first_regressor(file_name):
    raw_data = open(file_name, "rb")
    X_tr, y_tr, X_te = pickle.load(raw_data)
    #np.savetxt("foo.csv", X_te, delimiter=",", fmt='%.3e')
    
    start_fit = time.time()
    svm_regressor = SVR(kernel = 'rbf')
    svm_regressor.fit(X_tr, y_tr[:,0])
    end_fit = time.time()
    print("fit: ", end_fit - start_fit)
    y_preds = svm_regressor.predict(X_te)
    np.savetxt("lat_predictions_v01.csv", y_preds, delimiter=",", fmt='%.3e')
    

if __name__ == "__main__":
    first_regressor("total_data.pkl")


