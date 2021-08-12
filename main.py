'''
using python 3.8
# -*- coding: utf-8 -*-
Author HsuHsia
'''


import configparser
from util.config_util import read_list
# from unnormalized_fair_spectral_clustering_drug import ufsc
# from obesity import ufsc
from multi import ufsc


def main():

    config_file = "config/experiment_config.ini"
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)
    #
    config_str = "obesity"
    dataset = config[config_str].get("dataset")
    print("Clustering dataset: {}".format(dataset))
    num_clusters = config[config_str].getint("num_clusters")
    max_points = config[config_str].getint("max_points")
    dataset_config_file = config[config_str].get("config_file")

    # eta = 0.04
    # ufsc(dataset, num_clusters, dataset_config_file, max_points, eta)

    # etas = np.arange(0.1, 1.01, 0.01)
    # for eta in etas:
    #     ufsc(dataset, num_clusters, dataset_config_file, max_points, eta)
    run_num = 10
    eta = 0.04
    num_clusterss = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # num_clusterss = [20]
    # num_clusters = 5
    for num_clusters in num_clusterss:

    # max_pointss = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    # max_pointss = [10000]
    # for max_points in max_pointss:
        r0_ed = 0
        r1_ed = 0
        # r2_ed = 0
        # r0_cost = 0
        # r1_cost = 0
        # r2_cost = 0
        # r0_t = 0
        # r1_t = 0
        # r2_t = 0
        print("当前的k是")
        print(num_clusters)
        print("         ")
        print("         ")
        for i in range(run_num):
            # r0_c, r1_c, r2_c = ufsc(dataset, num_clusters, dataset_config_file, max_points, eta)
            r0_c, r1_c = ufsc(dataset, num_clusters, dataset_config_file, max_points, eta)
            i += 1
            r0_ed += r0_c.ed
            r1_ed += r1_c.ed
            # r2_ed += r2_c.ed
            # r0_cost += r0_c.cost / max_points
            # r1_cost += r1_c.cost / max_points
            # r2_cost += r2_c.cost / max_points

            # r0_t += r0_c.t
            # r1_t += r1_c.t
            # r2_t += r2_c.t
        print("ed:")
        print(r0_ed / (run_num * num_clusters))
        print(r1_ed / (run_num * num_clusters))
        # print(r2_ed / (run_num * num_clusters))
        # print("cost")
        # print(r0_cost / run_num)
        # print(r1_cost / run_num)
        # print(r2_cost / run_num)
        # print("time:")
        # print(r0_t / run_num)
        # print(r1_t / run_num)
        # print(r2_t / run_num))
        print("         ")
        print("         ")
        print("         ")
        print("         ")
        print("         ")
        print("         ")


if __name__ == "__main__":
    main()
