import numpy as np
import dill as pickle
import sklearn
from numpy import genfromtxt

def openFile(fileName):
    raw_data = open(fileName, "rb")
    X_tr, y_tr, X_te = pickle.load(raw_data)
    print(X_te)

if __name__ == "__main__":
    #training_data = genfromtxt('/Users/william/Documents/2018-2019/Machine_Learning/Project4/data_kaggle/posts_train.txt', delimiter=',', skip_header=1)
    #test_data = genfromtxt('/Users/william/Documents/2018-2019/Machine_Learning/Project4/data_kaggle/posts_test.txt', delimiter=',', skip_header=1)
    #X_tr = training_data[:, [1,2,3,6]]
    #y_tr = training_data[:, [4,5]]
    #X_te = test_data[:, 1:5]
    #full_data = (X_tr, y_tr, X_te)
    graph_data = genfromtxt('/Users/william/Documents/2018-2019/Machine_Learning/Project4/data_kaggle/graph.txt', skip_header=1)
    full_mat = np.array(graph_data)
    with open('graph.pkl','wb') as f:
        pickle.dump(full_mat, f)