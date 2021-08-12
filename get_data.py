import configparser
from util import data_processing as dp
from util.config_util import read_list
from util.constraint import constraint_matrix


def get_data(dataset, config_file, max_points):
    # 读取数据
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)
    df_or = dp.read_data(config, dataset)

    # 设置数据集大小（抽样）
    if max_points and len(df_or) > max_points:
        df_or = dp.subsample_data(df_or, max_points)

    # 清洗数据（量化敏感属性）
    df_or, _ = dp.clean_data(df_or, config, dataset)

    # 选取用作定义距离的属性
    selected_columns = config[dataset].getlist("columns")
    df = df_or[[col for col in selected_columns]].iloc[:, :].values

    # 取出敏感属性
    fairness_variable = config[dataset].getlist("fairness_variable")

    # 对敏感属性建模
    # attributes 保存每个颜色类别的点的索引
    # color_flag 从点到它所属的颜色类别的映射（与“attributes”相反）
    # f 是敏感属性的指示矩阵 即fis = 1 点i在group s中 输出的f是一个 s * n ，行向量构成的矩阵
    attributes, color_flag, f = dp.sensitive_attr(df_or, fairness_variable, dataset, config_file)

    # 计算敏感属性在整个原始数据集所占比例
    representation = {}
    for var, bucket_dict in attributes.items():
        representation[var] = {k: (len(bucket_dict[k]) / len(df_or)) for k in bucket_dict.keys()}

    # 生成约束矩阵 F（这里的F是文中的F^T）
    p = constraint_matrix(df_or, f, fairness_variable, representation)

    F = {}
    for variable in fairness_variable:
        F[variable] = p[variable] - f[variable]

    return df, F, attributes, df_or, representation, fairness_variable, f














