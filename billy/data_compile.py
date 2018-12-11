import numpy as np
import dill as pickle
from numpy import genfromtxt

if __name__ == "__main__":
    lat = genfromtxt('/Users/william/Documents/2018-2019/Machine_Learning/Project4/billy/test_lat_preds.csv')
    long = genfromtxt('/Users/william/Documents/2018-2019/Machine_Learning/Project4/billy/test_long_preds.csv')
    test = genfromtxt('/Users/william/Documents/2018-2019/Machine_Learning/Project4/data_kaggle/posts_test.txt', delimiter=",", skip_header=1)
    ids = test[:,0]
    #print(lat)
    #print(long)
    #print(ids.shape)
    full = np.column_stack((ids, lat, long))
    np.savetxt("first_full_predictions.csv", full, '%5.3f')

    total_set = genfromtxt('/Users/william/Documents/2018-2019/Machine_Learning/Project4/billy/first_full_predictions.csv')
    for row in total_set:
        first = row[0]
        