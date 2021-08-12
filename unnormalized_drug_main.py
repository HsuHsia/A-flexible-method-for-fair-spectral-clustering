import configparser
from util.config_util import read_list
from unnormalized_fair_spectral_clustering_drug import ufsc


def main():

    config_file = "config/experiment_config.ini"
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)
    #
    config_str = "drug"
    dataset = config[config_str].get("dataset")
    print("Clustering dataset: {}".format(dataset))
    num_clusters = config[config_str].getint("num_clusters")
    max_points = config[config_str].getint("max_points")
    dataset_config_file = config[config_str].get("config_file")
    eta = 0.04
    # num_cluster = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    ub, b1, b2 = ufsc(dataset, num_clusters, dataset_config_file, max_points, eta)
    print("num_clusters")
    print(num_clusters)

    print("ratio_cut")
    print(ub.cost)
    print(b1.cost)
    print(b2.cost)
    print("ed")
    print(ub.ed)
    print(b1.ed)
    print(b2.ed)
    print("balance")
    print(ub.balance)
    print(b1.balance)
    print(b2.balance)
    print(" ")
    print(" ")
    print(" ")
    print(" ")


if __name__ == "__main__":
    main()
