import numpy as np


def constraint_matrix(df_or, f, fairness_variable, representation):
    P = {}
    for variable in fairness_variable:
        p = np.zeros((len(f[variable]), len(f[variable][0])))
        for idx, value in representation[variable].items():
            for i in range(len(df_or)):
                p[idx, i] = value
        P[variable] = p
    return P