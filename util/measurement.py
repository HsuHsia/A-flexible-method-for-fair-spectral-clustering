from collections import defaultdict
from functools import partial
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn import metrics as sm


def sc_score(df, label):
    score = sm.silhouette_score(df, label, metric='euclidean')
    return score


def cal_sizes(label, num_clusters):
    sizes = [0 for _ in range(num_clusters)]
    for p in label:
        sizes[p] += 1
    return sizes


def cal_ratios(attributes, df, label, num_clusters, sizes):
    fairness = {}
    # For each point in the dataset, assign it to the cluster and color it belongs too
    # 统计每个簇中各个组的数量
    for attr, colors in attributes.items():
        fairness[attr] = defaultdict(partial(defaultdict, int))
        for i, row in enumerate(df.iterrows()):
            cluster = label[i]
            for color in colors:
                if i in colors[color]:
                    fairness[attr][cluster][color] += 1
                    continue

    ratios = {}
    for attr, colors in attributes.items():
        attr_ratio = {}
        for cluster in range(num_clusters):
            if sizes[cluster] != 0:
                attr_ratio[cluster] = [fairness[attr][cluster][color] / sizes[cluster] for color in sorted(colors.keys())]
            else:
                attr_ratio[cluster] = [0 for color in sorted(colors.keys())]
        ratios[attr] = attr_ratio
    return ratios


def euclidean_D(label, df, attributes, representation, fairness_variable, num_clusters):
    sizes = cal_sizes(label, num_clusters)
    alg_ratios = cal_ratios(attributes, df, label, num_clusters, sizes)
    ed = 0
    for attr in fairness_variable:
        for cluster in range(num_clusters):
            for k in representation[attr].keys():
                dist = abs(np.array(representation[attr][k]) - np.array(alg_ratios[attr][cluster][k]))
                ed = ed + dist
    return ed / num_clusters


def balance(label, df, attributes, representation, fairness_variable, num_clusters):
    sizes = cal_sizes(label, num_clusters)
    alg_ratios = cal_ratios(attributes, df, label, num_clusters, sizes)
    b = {}
    b_sum = {}
    for attr in fairness_variable:
        c_balance = defaultdict(list)
        sum_balance = 0
        for cluster in range(num_clusters):
            curr_balance = []
            for k in representation[attr].keys():
                if alg_ratios[attr][cluster][k] != 0 and representation[attr][k] != 0:
                    c1 = representation[attr][k] / alg_ratios[attr][cluster][k]
                    c2 = alg_ratios[attr][cluster][k] / representation[attr][k]
                    curr_balance.append(min(c1, c2))
                else:
                    curr_balance.append(0)
            c_balance[cluster] = min(curr_balance)
            sum_balance += c_balance[cluster]
        b[attr] = c_balance
        b_sum[attr] = sum_balance
    return b, sum_balance


def ratio_cut(labels, num_clusters, adjacency_matrix):
    sizes = cal_sizes(labels, num_clusters)
    labels_index = []
    arr = np.array(labels)
    n = len(arr)
    cost = 0
    for k in range(num_clusters):
        x = np.array(np.where(arr == k))[0]
        labels_index.append(x)
        a = adjacency_matrix.copy()

        for i in range(n):
            if i in x:
                for j in range(n):
                    if j in x:
                        a[i, j] = 0
        cost += a.sum() / sizes[k]
    return cost
