import numpy as np
from numpy import genfromtxt


if __name__ == "__main__":
    train_data = genfromtxt('data_kaggle\posts_train.txt', delimiter=',', skip_header=1)
    graph = genfromtxt('data_kaggle\graph.txt', skip_header=1)
    train_data = np.array(train_data)
    graph = np.array(graph)
    print(graph)
    flat_graph = graph[:, 0].flatten()
    flat_train = train_data[:, 0].flatten()

    to_return = []
    for datum in train_data:
        sum_latitude = 0
        sum_longitude = 0
        n = 0
        start_index = np.searchsorted(flat_graph, datum[0], side='left')
        end_index = np.searchsorted(flat_graph, datum[0], side='right')

        for index in range(start_index, end_index):
            if graph[index][0] == datum[0]:
                train_index = np.searchsorted(flat_train, graph[index][1], side='left')
                """
                print("-------------------------------")
                print("Test Data index: ", datum[0])
                print("Graph index 0: ", graph[index][0])
                print("Train index: ", train_index)
                print("Graph index 1: ", graph[index][1])
                print("Train Data Index: ", train_data[train_index][0])
                print("Friend Latitude: ", train_data[train_index][4])
                print("Friend Longitude: ", train_data[train_index][5])
                print("-------------------------------")
                # """
                if train_index < len(train_data) and train_data[train_index][0] == graph[index][1]:
                    # print(" adding new data")
                    sum_latitude += train_data[train_index][4]
                    sum_longitude += train_data[train_index][5]
                    n += 1
        avg_latitude = 0
        avg_longitude = 0
        if n != 0:
            # print("N is not 0")
            avg_latitude = sum_latitude / n
            avg_longitude = sum_longitude / n
        to_return.append(np.append(datum, [avg_latitude, avg_longitude]))
    to_return = np.array(to_return)
    np.savetxt("train_data_with_friends.txt", to_return, delimiter=",", header="Id,Lat,Lon",
               fmt=['%i', '%i', '%i', '%i', '%1.3f', '%1.3f', '%i', '%1.3f', '%1.3f'])
    pass
