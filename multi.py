import scipy as sp
import numpy as np
from sklearn.cluster import KMeans
from time import time


from get_data import get_data
from util.spectral_clustering import get_adjacency_matrix, get_laplacian_matrix, unnormalized_spectral_clustering
import util.measurement as meas


class Res:
    def __init__(self, ed, sh, ch, t):
        self.ed = ed
        self.sh = sh
        self.ch = ch
        self.t = t


def pr(obj):
    for i in obj.__dict__.items():
        print(list(i))


def ufsc(dataset, num_clusters, dataset_config_file, max_points, eta):
    data, constraint_matrix, attributes, df_or, representation, fairness_variable, f = get_data(dataset, dataset_config_file, max_points)
    adjacency_matrix = get_adjacency_matrix(data)
    laplacian_matrix = get_laplacian_matrix(adjacency_matrix)

    # # case 1
    # constraint_matrix = np.vstack(np.array(constraint_matrix['class'], np.array(constraint_matrix['sex'])))
    constraint_matrix = np.append(np.array(constraint_matrix['Gender']), np.array(constraint_matrix['family_history_with_overweight']), axis=0)
    t1 = time()
    Z = sp.linalg.null_space(constraint_matrix)

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
    t2 = time()

    uf_labels, uf_cost = unnormalized_spectral_clustering(adjacency_matrix, num_clusters)
    t3 = time()
    uf_ed = meas.euclidean_D(uf_labels, df_or, attributes, representation, fairness_variable, num_clusters)
    uf_sh, uf_ch = meas.score(data, uf_labels)

    f1_ed = meas.euclidean_D(f1_labels, df_or, attributes, representation, fairness_variable, num_clusters)
    f1_sh, f1_ch = meas.score(data, f1_labels)

    res1 = Res(f1_ed, f1_sh, f1_ch, t2-t1)

    sc = Res(uf_ed, uf_sh, uf_ch, t3-t2)

    return sc, res1

