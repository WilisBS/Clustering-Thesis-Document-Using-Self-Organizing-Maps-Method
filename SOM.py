import numpy as np
import pandas as pd
import Topology

"""
variable note:
x -> data (array of features)
X -> array of data
w -> neuron (array of weight)
W -> array of neuron
D -> array of distance
winner -> index of winner neuron
epoch -> loop number
Et -> maximum loops
E0 -> epoch ordering phase
n_clusters = number of clusters
topology = architecture of SOM
r = neighbour's radius
"""

# method to count distance
def get_Distance(x, W):
    return [np.sqrt(sum((x - w) ** 2)) for w in W]          # euclidean distance


# method to get the index of winner neuron
def get_Winner(D):
    return np.argmin(D)                                     # index of minimum distance


# method to update winner's weight (*and neighbour's)
def update_weight(W, winner, a, x, r, n_clusters, topology):
    W[winner] += a * (x - W[winner])                        # update winner's weight

    if r > 0:
        r1, r2 = Topology.get_neighbours(n_clusters, topology)            # SOM topology
        if r == 1:
            for neighbour in r1[winner]:                            # winner's neighbour in r = 1
                W[neighbour] += a * (x - W[neighbour])                      # update neighbour's weight
        elif r == 2:
            for neighbour in r2[winner]:                            # winner's neighbour in r = 2
                W[neighbour] += a * (x - W[neighbour])                      # update neighbour's weight

    return W                                                # updated weight


# method training of SOM Algorithm
def training(X, a, b, Et, E0, n_clusters, topology, r):
    W = np.around(np.random.uniform(size=(n_clusters, len(X[0]))), decimals=3)     # random W
    epoch = 0  # initialize epoch

    while epoch < Et:                                                       # looping untill max_epoch
        for x in X:                                                         # looping to every x
            D = get_Distance(x, W)                                          # get distance between x and every w
            winner = get_Winner(D)                                          # get index of winner neuron
            W = update_weight(W, winner, a, x, r, n_clusters, topology)     # update the winner's weight (*and neighbour's according to value of r)

        a *= b                                                              # update a
        if(Et % E0 == 0):                                                   # jika E0 kelipatan Et, maka:
            r -= 1                                                          # decrement r
        epoch += 1                                                          # increment epoch

    return W                                                                # final weight


# method testing of SOM Algorithm
def testing(W, X):
    clusters = []                                   # array of clusters
    for n in range(len(W)):                         # looping till the length of weight
        n = []                                      # create empty cluster
        clusters.append(n)                          # append cluster to array

    i = 0
    for x in X:                                     # looping for every x
        D = get_Distance(x, W)                      # count distance
        winner = get_Winner(D)                      # get Winner's index
        clusters[winner].append(i)                  # append the data into winner cluster
        i += 1

    filled_clusters = []                             # filter empty cluster
    for cluster in clusters:
        if (len(cluster) > 0):
            filled_clusters.append(cluster)

    cluster = []                                    # get cluster number for every data
    for i in range(len(filled_clusters)):
        for x in filled_clusters[i]:
            cluster.append(i + 1)

    index = [j for sub in filled_clusters for j in sub]    # index clustered data

    cluster_df = pd.DataFrame(list(zip(index, cluster)), columns=["index", "cluster"])   # cluster dataframe
    cluster_df = cluster_df.sort_values(by='index')                                               # sort by index
    final_clusters = cluster_df['cluster'].tolist()                                      # get sorted cluster

    return final_clusters                                 # final clusters


# ==============================================================================================


if __name__ == '__main__' :

    # read data : data iris -> 150 data
    data_iris = np.genfromtxt('iris.csv', delimiter=',', dtype=str)
    data_iris = data_iris[:,:-1]
    data_iris = data_iris.astype(np.float)

    # SOM
    neurons = training(X=data_iris, a=0.5, b=0.7, Et=100, E0=25, n_clusters=3, topology="hexagonal", r=2)
    clusters = testing(W=neurons, X=data_iris)

    # print
    # j = 1
    for i in range(len(clusters)):
        print("cluster",i+1,"(",len(clusters[i]),"data ):\n",clusters[i],"\n")

    print()

    cluster = []
    for i in range(len(clusters)):
        for x in clusters[i]:
            cluster.append(i + 1)
    print(cluster)

    print(
        "\n",clusters
    )

    clusters = [j for sub in clusters for j in sub]
    print(clusters)

    # import pandas as pd

    a = pd.DataFrame(list(zip(clusters,cluster)), columns=["index","cluster"])
    a = a.sort_values(by='index')
    cluster = a['cluster'].tolist()
    print(cluster)
    # print(a.iloc[50:105, :])
