#!/goinfre/lcoiffie/miniconda3/envs/42AI-lcoiffie/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv


ROW_NAME = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']


def display_usage():
    print("USAGE : describe.py take a .csv dataset as argument, try again")
    exit(1)

###############################################################################
#                           MAIN FUNCTION                                     #
###############################################################################


def my_description(df):
    df_num = df.select_dtypes(include='number')  # selectionne numerical values
    datas = np.array(df_num)
    descript = np.zeros((len(ROW_NAME), datas.shape[1]))
    for i in range(datas.shape[1]):
        selected = datas[:, i]
        selected = selected[~np.isnan(selected)]
        if selected.shape[0] == 0:
            descript[0, i] = 0.0
            for line in range(1, 8):
                descript[line, i] = np.NAN
        else:
            descript[0, i] = count(selected)
            descript[1, i] = mean(selected)
            descript[2, i] = std(selected)
            descript[3, i], descript[7, i] = min_max(selected)
            descript[4, i], descript[6, i] = quartile(selected)
            descript[5, i] = median(selected)
    result = pd.DataFrame(descript,
                          index=ROW_NAME,
                          columns=list(df_num.columns))
    return result


###############################################################################
#                           STATISTIC TOOLS                                   #
###############################################################################

def count(x):
    """count the number of non-NAN rows"""
    return float(x.shape[0])


def mean(x):
    """Calculate the mean"""
    m = x.shape[0]
    if m == 0:
        return None
    total = 0
    for nb in x:
        total += nb
    return float(total / m)


def var(x):
    """Calculate variance"""
    meanX = mean(x)
    length = x.shape[0]
    total = 0
    for nb in x:
        total += (nb - meanX)**2
    return (float(total / (length - 1)))  # Panda style
    # return (float(total / length))  # Numpy style


def std(x):
    """Calculate standard deviation"""
    return (float(var(x)**0.5))


def min_max(x):
    """Calculate minimum and maximum value"""
    min = 1e123
    max = -1e123
    for nb in x:
        if nb > max:
            max = nb
        if nb < min:
            min = nb
    return min, max


def median(x):
    """calculate median value"""
    length = x.shape[0]
    x = sorted(x)
    if length % 2 == 1:
        return (float(x[length // 2]))
    else:
        a = x[int(length / 2) - 1]
        b = x[int(length / 2)]
        return (float((a + b) / 2))


def quartile(x):
    """Calculate first  and third quartile, linear interpolation way"""
    length = x.shape[0]
    x = sorted(x)
    first = (length - 1) * 0.25
    third = (length - 1) * 0.75
    if first == float(length // 4):
        a = x[int(first)]
        b = x[int(third)]
        return [float(a), float(b)]
    else:
        i_first = x[int(first)]
        j_first = x[int(first) + 1]
        i_third = x[int(third)]
        j_third = x[int(third) + 1]
        fract_first = first - int(first)
        fract_third = third - int(third)
        a = (i_first) + (j_first - i_first) * fract_first
        b = (i_third) + (j_third - i_third) * fract_third

        return [float(a), float(b)]


if __name__ == "__main__":
    if len(argv) != 2:
        display_usage()
    try:
        df = pd.read_csv(argv[1])
    except (FileNotFoundError):
        print("File or keys do not exist")
        display_usage()
    print("\n\n######## Results from pandas library ##############\n")
    print(df.describe())
    print("\n\n############ My results ####################\n")
    result = my_description(df)
    print(result)
    print("\nFor a better cosmetic result")
    print(result.T)
