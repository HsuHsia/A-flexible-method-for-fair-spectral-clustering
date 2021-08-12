import scipy as sp
import numpy as np
from sklearn.cluster import KMeans


from get_data import get_data
from util.spectral_clustering import get_adjacency_matrix, get_laplacian_matrix, unnormalized_spectral_clustering
import util.measurement as meas
from time import time

import judge


class Res:
    def __init__(self, ed, sh, balance, cost):
        self.ed = ed
        self.sh = sh
        self.balance = balance
        self.cost = cost


def pr(obj):
    for i in obj.__dict__.items():
        print(list(i))


def ufsc(dataset, num_clusters, dataset_config_file, max_points, eta):
    data, constraint_matrix, attributes, df_or, representation, fairness_variable, f = get_data(dataset, dataset_config_file, max_points)
    adjacency_matrix = get_adjacency_matrix(data)
    laplacian_matrix = get_laplacian_matrix(adjacency_matrix)

    # # case 1
    Z = sp.linalg.null_space(constraint_matrix['Gender'])

    # 令L0= (Z^T) * L * Z
    L0 = np.matmul(np.transpose(Z), np.matmul(laplacian_matrix, Z))

    # 得到L0的特征值和特征向量（特征向量为Y）
    eigvals, eigvecs = np.linalg.eigh(L0)
    indices = np.argsort(eigvals)[:num_clusters]

    # 取出前k小的特征值对应的特征向量，并进行正则化
    k_smallest_eigenvectors_y = eigvecs[:, indices]

    # H = ZY
    H = np.matmul(Z, k_smallest_eigenvectors_y)

    # # 利用KMeans进行聚类
    f1_labels = KMeans(n_clusters=num_clusters).fit_predict(H)

    # 第二种情况
    r_eigvals, r_eigvecs = np.linalg.eigh(laplacian_matrix)
    # 构建约束条件上界beta
    beta = {}
    for variable in fairness_variable:
        alpha = np.ones(len(constraint_matrix[variable]))
        beta[variable] = eta * alpha
    candidate_set = []
    candidate_values = []
    for i in range(len(r_eigvecs)):
        r = np.matmul(constraint_matrix[fairness_variable[0]], r_eigvecs[i])
        if np.all(r < np.array(beta[fairness_variable[0]])) and np.all(r > -np.array(beta[fairness_variable[0]])):
            candidate_set.append(list(r_eigvecs[:, i]))
            candidate_values.append(r_eigvals[i])

    candidate_set = np.transpose(candidate_set)
    # print(candidate_set.shape)
    r_indices = np.argsort(candidate_values)[:num_clusters]
    k_smallest_eigenvectors = np.array(candidate_set)[:, r_indices]
    f2_labels = KMeans(n_clusters=num_clusters).fit_predict(k_smallest_eigenvectors)

    uf_labels = unnormalized_spectral_clustering(adjacency_matrix, num_clusters)

    f2_ed = meas.euclidean_D(f2_labels, df_or, attributes, representation, fairness_variable, num_clusters)
    f2_sh = meas.sc_score(data, f2_labels)
    f1_ed = meas.euclidean_D(f1_labels, df_or, attributes, representation, fairness_variable, num_clusters)
    f1_sh = meas.sc_score(data, f1_labels)

    uf_ed = meas.euclidean_D(uf_labels, df_or, attributes, representation, fairness_variable, num_clusters)
    uf_sh = meas.sc_score(data, uf_labels)
    print("ed")
    print(f1_ed)
    print(f2_ed)
    print(uf_ed)

    res1 = Res(f1_ed, f1_sh, 1, 1)
    res2 = Res(f2_ed, f2_sh, 1, 1)

    sc = Res(uf_ed, uf_sh, 1, 1)

    res = judge.judge(res1, res2, sc)

    return sc, res, res1

