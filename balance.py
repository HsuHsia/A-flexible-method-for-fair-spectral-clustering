# Balance
import numpy as np


# returns the % of each community in partioning
def prop(ground_truth):
    my_dic = {}
    for k in range(max(ground_truth) + 1):  # O(h)
        my_set = {i for i, x in enumerate(ground_truth) if x == k}  # O(n)
        my_calc = (len(my_set) * 100) / len(ground_truth)
        my_dic[k] = int(my_calc)

    return my_dic


def balance(ground_truth, labels):
    # Constants
    result = 0
    indices = []
    arr = np.array(labels)
    # print(arr)
    # 返回原始分布的百分比 representation
    init_distrib = prop(ground_truth)
    # Get list of indices
    for k in range(max(arr + 1)):
        # k = cluster
        x = np.where(arr == k)
        # x 是labels类别的索引 x = (array([  0,   1,   2,   3,   4,   5,   6,   7]), array([8,   9,  10,  11,  12,
        #         13,  15,  16,  17,  18,  19,  20,  21]), array([22,  23,  24,  25,  26,
        #         27,  28,  29,  30,  31,  32,  35,  36,  39,  40,  43,  44,  45])
        distribution = []
        for i in range(len(x[0])):
            distribution.append(ground_truth[x[0][i]])

        cluster_distrib = prop(distribution)
        # cluster_distrib 是每个簇中敏感属性的比例 ratios
        score = 0

        for key in cluster_distrib.keys():

            if cluster_distrib[key] != 0:
                score += abs(init_distrib[key] - cluster_distrib[key])
            #     print(init_distrib[key])
            #     print(cluster_distrib[key])
            # print('score')
            # print(score)

        # The greater the score the less balanced it is, therefore we subtract to the result to make it
        # more understandable on graphs
        result -= score

    return result / max(arr + 1)