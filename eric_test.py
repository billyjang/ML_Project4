import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    test_data = genfromtxt('Test.txt', delimiter=',', skip_header=1)
    arr = np.array(test_data)
    newarr = arr[arr[:, 0].argsort()]
    np.savetxt("Test3.txt", newarr, delimiter=",", header="Id,Lat,Lon", fmt=['%i', '%1.3f', '%1.3f'])

    """
    train_data = genfromtxt('data_kaggle/posts_train.txt', delimiter=',', skip_header=1)

    X_tr = train_data[:, 1:4]


    X = train_data[:, 4]
    Y = train_data[:, 5]
    plt.scatter(X, Y)
    plt.show()
