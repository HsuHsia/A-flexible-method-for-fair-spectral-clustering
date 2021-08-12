'''
using python 3.8
# -*- coding: utf-8 -*-
Author HsuHsia
'''


import configparser
from util.config_util import read_list
from obesity import ufsc


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

    eta = 0.04
    ufsc(dataset, num_clusters, dataset_config_file, max_points, eta)

    # etas = np.arange(0.1, 1.01, 0.01)
    # for eta in etas:
    #     ufsc(dataset, num_clusters, dataset_config_file, max_points, eta)


if __name__ == "__main__":
    main()
