import networkx as nx
import numpy as np
from time import time
from util.spectral_clustering import un_spectral_clustering, un_fair_spectral_clustering
import balance


class Res:
    def __init__(self, fair_ed, fair_wd, fair_balance, fair_e):
        self.fair_ed = fair_ed
        self.fair_wd = fair_wd
        self.fair_balance = fair_balance
        self.fair_e = fair_e


# The functions to help us build a suitable ground-truth for the fair-clustering approach

# !!! REQUIRES OPTIMISATION FOR SCALABILITY

# This function merges
def merging2(array, index, h, k):
    arr = []
    arr = np.append(arr, array[index])

    while index < h * k:
        if index + k < h * k:
            arr = np.append(arr, array[index + k])
            index += k
        else:
            return arr

    return arr


def get_ground_truth(h, k, labels, group_ID):
    """
    Parameters
    ----------
    h : INTEGER
        NUMBER OF GROUPS.
    k : INTEGER
        NUMBER OF CLUSTERS.
    labels : TYPE
        UNFAIR CLUSTERING LABELS.
    group_ID : INTEGER
        GROUP TO RETURN (used for recursivity).
    Returns
    -------
    x : ARRAY
        GROUND-TRUTH SET REPRESENTING A FAIR CLUSTERING.
    """
    # my_set = get_ground_truth(5, 5, my_range, k).astype(int)
    arr = []
    count = 0
    size = len(labels)
    for y in range(k):
        for i in range(k):
            arr.append(labels[count:size // (h * k) + count])
            count += size // (h * k)
    x = merging2(arr, group_ID, h, k)

    return x


def error_sym(labels, fair_labels):

    pre_results = []
    ground_truth = []
    results = []
    # my_range = np.arange(0, len(fair_labels)) 与下面等价
    my_range = np.arange(0, len(labels))
    # print(max(labels))
    # print(max(fair_labels))
    lengths = []
    # Return the indices of elements of the same group and store them
    for h in range(max(fair_labels) + 1):
        # h 是聚类的数目
        my_set = {i for i, x in enumerate(fair_labels) if x == h}
        pre_results.append(my_set)
        lengths.append(len(my_set))

    # Generate ground-truth in the same format (indices)
    # 每个组包含 n / groups 的point
    # 返回一个定义的敏感属性标签
    for k in range(max(labels) + 1):
        # my_set = get_ground_truth(组数, 集群数, 标签, 组的序号).astype(int)
        my_set = get_ground_truth(5, 5, my_range, k).astype(int)
        ground_truth.append(set(my_set))
        lengths.append(len(my_set))
    # lengths [96, 1, 1, 1, 1, 20, 20, 20, 20, 20]
    print(ground_truth)

    # Cross-compute the symmetric difference between the 2

    # len(ground_truth) = k
    for i in range(len(ground_truth)):
        for y in range(len(pre_results)):
            # symmetric_difference()方法返回两个集合中不重复的元素集合，即会移除两个集合中都存在的元素。
            x = len(ground_truth[i].symmetric_difference(pre_results[y]))
            results.append(x)
    # print(results)
    # print(min(results))
    return (min(results) * 100) / len(labels)


def generate_sbm_h5_k5(size, a, b, c, d, i):
    arr = [size // 25] * 25

    probs = [[a, b, b, b, b, c, d, d, d, d, c, d, d, d, d, c, d, d, d, d, c, d, d, d, d],
             [b, a, b, b, b, d, c, d, d, d, d, c, d, d, d, d, c, d, d, d, d, c, d, d, d],
             [b, b, a, b, b, d, d, c, d, d, d, d, c, d, d, d, d, c, d, d, d, d, c, d, d],
             [b, b, b, a, b, d, d, d, c, d, d, d, d, c, d, d, d, d, c, d, d, d, d, c, d],
             [b, b, b, b, a, d, d, d, d, c, d, d, d, d, c, d, d, d, d, c, d, d, d, d, c],

             [c, d, d, d, d, a, b, b, b, b, c, d, d, d, d, c, d, d, d, d, c, d, d, d, d],
             [d, c, d, d, d, b, a, b, b, b, d, c, d, d, d, d, c, d, d, d, d, c, d, d, d],
             [d, d, c, d, d, b, b, a, b, b, d, d, c, d, d, d, d, c, d, d, d, d, c, d, d],
             [d, d, d, c, d, b, b, b, a, b, d, d, d, c, d, d, d, d, c, d, d, d, d, c, d],
             [d, d, d, d, c, b, b, b, b, a, d, d, d, d, c, d, d, d, d, c, d, d, d, d, c],

             [c, d, d, d, d, c, d, d, d, d, a, b, b, b, b, c, d, d, d, d, c, d, d, d, d],
             [d, c, d, d, d, d, c, d, d, d, b, a, b, b, b, d, c, d, d, d, d, c, d, d, d],
             [d, d, c, d, d, d, d, c, d, d, b, b, a, b, b, d, d, c, d, d, d, d, c, d, d],
             [d, d, d, c, d, d, d, d, c, d, b, b, b, a, b, d, d, d, c, d, d, d, d, c, d],
             [d, d, d, d, c, d, d, d, d, c, b, b, b, b, a, d, d, d, d, c, d, d, d, d, c],

             [c, d, d, d, d, c, d, d, d, d, c, d, d, d, d, a, b, b, b, b, c, d, d, d, d],
             [d, c, d, d, d, d, c, d, d, d, d, c, d, d, d, b, a, b, b, b, d, c, d, d, d],
             [d, d, c, d, d, d, d, c, d, d, d, d, c, d, d, b, b, a, b, b, d, d, c, d, d],
             [d, d, d, c, d, d, d, d, c, d, d, d, d, c, d, b, b, b, a, b, d, d, d, c, d],
             [d, d, d, d, c, d, d, d, d, c, d, d, d, d, c, b, b, b, b, a, d, d, d, d, c],

             [c, d, d, d, d, c, d, d, d, d, c, d, d, d, d, c, d, d, d, d, a, b, b, b, b],
             [d, c, d, d, d, d, c, d, d, d, d, c, d, d, d, d, c, d, d, d, b, a, b, b, b],
             [d, d, c, d, d, d, d, c, d, d, d, d, c, d, d, d, d, c, d, d, b, b, a, b, b],
             [d, d, d, c, d, d, d, d, c, d, d, d, d, c, d, d, d, d, c, d, b, b, b, a, b],
             [d, d, d, d, c, d, d, d, d, c, d, d, d, d, c, d, d, d, d, c, b, b, b, b, a]]

    graph = nx.stochastic_block_model(arr, probs, seed=1)

    # Run Algorithms
    t0 = time()  # Start

    labels = un_spectral_clustering(graph, 5)
    # print(labels)
    t1 = time()  # Run Time Algo 1
    fair_labels_1 = un_fair_spectral_clustering(graph, 5, labels)
    # fair_balance_1 = balance.balance(f['sex'][0].astype(int), fair_labels_1)

    t2 = time()  # Run Time Algo 2

    # 返回两位小数
    y = round(error_sym(labels, fair_labels_1), 4)
    return y, (t1-t0), (t2-t1)


def get_sbm_avg_result_h5_k5(runtimes, a, b, c, d):

    error_scores = np.array([])
    usc_scores = np.array([])
    fair_usc_scores = np.array([])

    for i in range(runtimes):
        for size in range(100, 200, 100):
            error_s, usc_time, fair_usc_time = generate_sbm_h5_k5(size, a, b, c, d, 1)
            error_scores = np.append(error_scores, [error_s])
            usc_scores = np.append(usc_scores, [usc_time])
            fair_usc_scores = np.append(fair_usc_scores, [fair_usc_time])
    return error_scores, usc_scores, fair_usc_scores


a = 0.2
b = 0.15
c = 0.1
d = 0.05
runTimes = 1
error, usc_t, fair_usc_t = get_sbm_avg_result_h5_k5(runTimes, a, b, c, d)

print(error)
print(usc_t)
print(fair_usc_t)
