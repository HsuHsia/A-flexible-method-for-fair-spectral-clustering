import pandas as pd
from collections import defaultdict
import configparser
from util.config_util import read_list
import numpy as np


def read_data(config, dataset):
    csv_file = config[dataset]["csv_file"]
    df = pd.read_csv(csv_file, sep=config[dataset]["separator"])

    if config["DEFAULT"].getboolean("describe"):
        print(df.describe())

    return df


def clean_data(df, config, dataset):

    # selected_columns = ['age', 'balance', 'duration']
    selected_columns = config[dataset].getlist("columns")

    #
    # (variables_of_interest) = ['marital', 'default']
    variables_of_interest = config[dataset].getlist("fairness_variable")

    # Bucketize text data
    text_columns = config[dataset].getlist("fairness_variable", [])
    for col in text_columns:
        # Cat codes is the 'category code'. Aka it creates integer buckets automatically.
        df[col] = df[col].astype('category').cat.codes

    # Remove the unnecessary columns. Save the variable of interest column, in case
    # it is not used for clustering.
    variable_columns = [df[var] for var in variables_of_interest]
    # df = df[[col for col in selected_columns]]

    # Convert to float, otherwise JSON cannot serialize int64
    for col in df:
        if col in text_columns or col not in selected_columns: continue
        df[col] = df[col].astype(float)

    if config["DEFAULT"].getboolean("describe_selected"):
        print(df.describe())

    return df, variable_columns


def subsample_data(df, N):
    return df.sample(n=N).reset_index(drop=True)


def sensitive_attr(df, fairness_variable, dataset, config_file):
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)
    # 对敏感属性建模
    # attributes 保存每个颜色类别的点的索引
    # color_flag 从点到它所属的颜色类别的映射（与“attributes”相反）
    attributes, color_flag, F = {}, {}, {}
    for variable in fairness_variable:
        bucket_conditions = config[dataset].getlist(variable + "_conditions")
        colors = defaultdict(list)
        this_color_flag = [0] * len(df)
        f = np.zeros((len(bucket_conditions), len(df)))

        # bucket_idx 表示敏感属性的取值个数
        for i, row in df.iterrows():
            for bucket_idx, bucket in enumerate(bucket_conditions):
                if eval(bucket)(row[variable]):
                    colors[bucket_idx].append(i)
                    this_color_flag[i] = bucket_idx
                    f[bucket_idx, i] = 1

        attributes[variable] = colors
        color_flag[variable] = this_color_flag
        F[variable] = f

    return attributes, color_flag, F
