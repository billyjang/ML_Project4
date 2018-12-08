import numpy as np
import dill as pickle
import sklearn
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/3518778/how-do-i-read-csv-data-into-a-record-array-in-numpy
from numpy import genfromtxt


if __name__ == "__main__":
    test_data = genfromtxt('data_kaggle\posts_test.txt', delimiter=',')

    pass
