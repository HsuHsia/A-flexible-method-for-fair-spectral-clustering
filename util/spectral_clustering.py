import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from numpy import linalg as LA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import scipy as sp
import networkx as nx


# 利用高斯核函数构造邻接矩阵
def get_adjacency_matrix(df):
    res = rbf_kernel(df)
    for i in range(len(res)):
        res[i, i] = 0
    return res


# 构造拉普拉斯矩阵
def get_laplacian_matrix(a):
    # 计算度矩阵d
    d = np.diag(np.sum(a, axis=1))
    L = d - a
    return L


def normalized_spectral_clustering(df, a, k):
    D = np.diag(np.power(np.sum(a, axis=1), -0.5))
    L = np.eye(len(df)) - np.dot(np.dot(D, a), D)
    eigvals, eigvecs = LA.eig(L)

    # # 前k小的特征值对应的索引，argsort函数
    indices = np.argsort(eigvals)[:k]
    # # 取出前k小的特征值对应的特征向量
    k_smallest_eigenvectors = eigvecs[:, indices]
    # # 利用KMeans进行聚类
    return KMeans(n_clusters=k).fit_predict(k_smallest_eigenvectors)


def unnormalized_spectral_clustering(a, k):
    L = get_laplacian_matrix(a)
    eigvals, eigvecs = LA.eigh(L)
    # # 前k小的特征值对应的索引，argsort函数

    indices = np.argsort(eigvals)[:k]
    # # 取出前k小的特征值对应的特征向量
    k_smallest_eigenvectors = eigvecs[:, indices]
    # # 利用KMeans进行聚类
    return KMeans(n_clusters=k).fit_predict(k_smallest_eigenvectors)


def un_spectral_clustering(G, k):

    laplacian_matrix = nx.normalized_laplacian_matrix(G)
    eigvals, eigvecs = np.linalg.eig(laplacian_matrix.toarray())
    # # 前k小的特征值对应的索引，argsort函数
    indices = np.argsort(eigvals)[:k]
    # # 取出前k小的特征值对应的特征向量
    k_smallest_eigenvectors = eigvecs[:, indices]
    # # 利用KMeans进行聚类
    return KMeans(n_clusters=k).fit_predict(k_smallest_eigenvectors)




