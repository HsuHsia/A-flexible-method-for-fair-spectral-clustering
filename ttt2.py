import scipy as sp
import numpy as np
from sklearn.cluster import KMeans
import math

from get_data import get_data
from util.spectral_clustering import get_adjacency_matrix, get_laplacian_matrix, unnormalized_spectral_clustering
import util.measurement as meas
import balance


def norm(v):

    return math.sqrt(sum(e ** 2 for e in v))


def vol(d):
    res = 0
    for i in range(len(d)):
        res += d[i][i]
    return res


def normalize(matrix, d):
    m = []
    for i in range(len(matrix[1])):
        if norm(matrix[:, i]):
            m.append(matrix[:, i] * vol(d)/ norm(matrix[:, i]))
        else:
            print("err")
    return np.transpose(m)


def ufsc(dataset, num_clusters, dataset_config_file, max_points, eta):
    data, constraint_matrix, attributes, df_or, representation, fairness_variable, f = get_data(dataset, dataset_config_file, max_points)
    # adjacency_matrix = get_adjacency_matrix(data)
    adjacency_matrix = data
    laplacian_matrix = get_laplacian_matrix(adjacency_matrix)

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
        # print(np.matmul(constraint_matrix[fairness_variable[0]], eigvecs[i]))
        if np.all(np.matmul(constraint_matrix[fairness_variable[0]], r_eigvecs[i]) < np.array(beta[fairness_variable[0]])) \
                and np.all(np.matmul(constraint_matrix[fairness_variable[0]], r_eigvecs[i]) > -np.array(beta[fairness_variable[0]])):
            # and np.all(np.matmul(F[fairness_variable[1]], eigvecs[i]) < np.array(beta[fairness_variable[1]])) and \
            # np.all(np.matmul(F[fairness_variable[1]], eigvecs[i]) > -np.array(beta[fairness_variable[1]])):
            candidate_set.append(list(r_eigvecs[:, i]))
            candidate_values.append(r_eigvals[i])
#
    d = np.diag(np.sum(adjacency_matrix, axis=1))
    candidate_set = np.transpose(candidate_set)

    normalize_candidate_vec = normalize(candidate_set, d)
    print(normalize_candidate_vec.shape)

    r_indices = np.argsort(candidate_values)[:num_clusters]
    k_smallest_eigenvectors = np.array(normalize_candidate_vec)[:, r_indices]
    r_f_labels = KMeans(n_clusters=num_clusters).fit_predict(k_smallest_eigenvectors)
    f_ed = meas.euclidean_D(r_f_labels, df_or, attributes, representation, fairness_variable, num_clusters)
    # f_balance = meas.balance_of_clustering(f_labels, df_or, attributes, num_clusters, representation)

    uf_labels = unnormalized_spectral_clustering(adjacency_matrix, num_clusters)
    uf_ed = meas.euclidean_D(uf_labels, df_or, attributes, representation, fairness_variable, num_clusters)
    # uf_balance = meas.balance_of_clustering(uf_labels, df_or, attributes, num_clusters, representation)
    fb = balance.balance(f['Gender'][0].astype(int), r_f_labels)
    ufb = balance.balance(f['Gender'][0].astype(int), uf_labels)

    print('eta:')
    print(eta)
    print('fair')
    print(f_ed)
    print('unfair')
    print(uf_ed)
    print(fb)
    print(ufb)
    print('')
    print('------------------------------------------------------')