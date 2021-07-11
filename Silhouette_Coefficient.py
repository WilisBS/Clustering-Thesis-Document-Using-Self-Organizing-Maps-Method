import numpy as np
import pandas as pd

# jarak antara dua data
def get_Distance(x, y):
    return np.around(np.sum((x - y) ** 2), decimals=3)

# rata-rata jarak antar data di cluster yang sama
def get_a(x, cluster_x):
    a = 0
    if len(cluster_x) > 1:
        for i in range(len(cluster_x)):
            if  x != i:
                a += get_Distance(cluster_x[x], cluster_x[i])
        a /= (len(cluster_x)-1)
    return np.around(a, decimals=3)

# rata-rata jarak antar data di cluster berbeda
def get_b(x, indeks_cluster_x, clusters):
    b = 0; n = 0
    for j in range(len(clusters)):
        if j != indeks_cluster_x:
            n += len(clusters[j])
            b += np.sum([get_Distance(clusters[indeks_cluster_x][x], y) for y in clusters[j]])
    b /= n
    return np.around(b, decimals=3)

def get_Silhouette(cluster):
    # preparation from df
    n_clusters = len(cluster.cluster.value_counts())
    all_cluster = []
    for i in range(n_clusters):
        df_i = cluster.loc[cluster.cluster == i + 1]
        df_i = df_i.drop(['judul skripsi', 'cluster'], axis=1)
        arr_i = df_i.to_numpy()
        all_cluster.append(arr_i)

    # get silhouette
    silhouette_arr = []
    for i in range(n_clusters):
        silhouette_i = []
        for x in range(len(all_cluster[i])):
            a = get_a(x, all_cluster[i])
            b = get_b(x, i, all_cluster)
            s = (b - a) / max(a, b)
            silhouette_i.append(s)
        silhouette_arr.append(silhouette_i)

    # perhitungan rata-rata silhouette(local dan global)
    global_S = 0
    global_n = 0
    local_S = []
    for i in range(len(silhouette_arr)):
        local_s = sum(j for j in silhouette_arr[i])
        global_S += local_s
        global_n += len(silhouette_arr[i])
        local_s /= len(silhouette_arr[i])
        local_S.append(local_s)
    global_S /= global_n

    # local silhouette to df
    local_S_df = pd.DataFrame(list(zip([i + 1 for i in range(n_clusters)], local_S)), columns=['cluster', 'silhouette'])

    return global_S, local_S_df


if __name__ == '__main__':

    cluster = pd.read_csv('cluster.csv', index_col=0)
    global_s, local_s_df = get_Silhouette(cluster)
    print("local Silhouette:\n", local_s_df)
    print("\nglobal Silhouette:", global_s, "~", round(global_s, 1))

    # # preparation from df
    # n_clusters = len(cluster.cluster.value_counts())
    # all_cluster = []
    # for i in range(n_clusters):
    #     df_i = cluster.loc[cluster.cluster == i+1]
    #     df_i = df_i.drop(['judul skripsi','cluster'], axis=1)
    #     arr_i = df_i.to_numpy()
    #     all_cluster.append(arr_i)
    #
    # # get silhouette
    # silhouette_arr = []
    # for i in range(n_clusters):
    #     print("cluster",i+1)
    #     silhouette_i = []
    #     for x in range(len(all_cluster[i])):
    #         a = get_a(x, all_cluster[i])
    #         b = get_b(x, i, all_cluster)
    #         s = (b - a) / max(a,b)
    #         silhouette_i.append(s)
    #     silhouette_arr.append(silhouette_i)
    #
    # # perhitungan rata-rata silhouette(local dan global)
    # global_S = 0
    # global_n = 0
    # local_S = []
    # for i in range(len(silhouette_arr)):
    #     local_s = sum(j for j in silhouette_arr[i])
    #     global_S += local_s
    #     global_n += len(silhouette_arr[i])
    #     local_s /= len(silhouette_arr[i])
    #     local_S.append(local_s)
    # global_S /= global_n
    #
    # # local silhouette to df
    # local_S_df = pd.DataFrame(list(zip([i+1 for i in range(n_clusters)], local_S)), columns=['cluster', 'silhouette'])
    # # print()
    #
    # print("local Silhouette:\n",local_S_df)
    # print("global Silhouette:", global_S, "setara dengan:", round(global_S,1))


