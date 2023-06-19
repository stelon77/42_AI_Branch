from ressources.polynomial_model import add_polynomial_features
from ressources.mylinearregression import MyLinearRegression as MyLR
from ressources.data_spliter import data_spliter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

FILENAME = "../space_avocado.csv"
X_WEIGHT = "weight"
X_DIST = "prod_distance"
X_TIME = "time_delivery"
Y_KEY = "target"
# FILENAME = "../spacecraft_data.csv"
# X_WEIGHT = "Age"
# X_DIST = "Thrust_power"
# X_TIME = "Terameters"
# Y_KEY = "Sell_price"
MAX_DEGREE = 4
TRAINING_PROPORTION = 0.8
# ALPHAS = [1, 0.5, 0.1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5,
#           1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9]
ITERATIONS = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
ALPHA = 0.1  # decided after testing multiples alphas


class bcolors:
    OK = '\033[92m'  # green
    WARNING = '\033[93m'  # yellow
    FAIL = '\033[91m'  # red
    RESET = '\033[0m'  # reset color


def zscore_normalization(x):
    mu = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    X = np.divide(np.subtract(x, mu), std)
    return X, mu, std


def column_select(x_train, w, d, t):
    x1 = x_train[:, 0:w]
    x2 = x_train[:, MAX_DEGREE:(MAX_DEGREE + d)]
    x3 = x_train[:, MAX_DEGREE * 2:MAX_DEGREE * 2 + t]
    return np.concatenate((x1, x2, x3), axis=1)


if __name__ == "__main__":

    ##################################################
    # 1-read dataset from csv file and extract datas #
    ##################################################
    try:
        data = pd.read_csv(FILENAME)
        x_weight = np.array(data[X_WEIGHT]).reshape(-1, 1)
        x_dist = np.array(data[X_DIST]).reshape(-1, 1)
        x_time = np.array(data[X_TIME]).reshape(-1, 1)
        y = np.array(data[Y_KEY]).reshape(-1, 1)
    except (FileNotFoundError, KeyError):
        print(bcolors.FAIL + "File or keys do not exist" + bcolors.RESET)
        exit(1)

    ###################################################
    # 1.1 -create models file                         #
    ###################################################
    with open("models.csv", "w") as fitted:
        fitted.write("name,alpha,max_iter,J_train,J_test\n")

    ###################################################
    # 2 -create polynomial features for each of the x #
    ###################################################
    x_weight = add_polynomial_features(x_weight, MAX_DEGREE)
    x_dist = add_polynomial_features(x_dist, MAX_DEGREE)
    x_time = add_polynomial_features(x_time, MAX_DEGREE)
    x_all = np.concatenate((x_weight, x_dist, x_time), axis=1)

    ######################################################
    # 3 -normalize datas with zscore retrieve mu and std #
    ######################################################
    x_all, mu, std = zscore_normalization(x_all)

    ##################################################
    # 4 -split the set                               #
    ##################################################
    x_train, x_test, y_train, y_test = \
        data_spliter(x_all, y, TRAINING_PROPORTION)

    #####################################################
    #          boucle principale d'essais               #
    #####################################################
    for w in range(1, MAX_DEGREE + 1):  # nombre de polynomes de weight
        for d in range(1, MAX_DEGREE + 1):  # nombre de polynomes de dist
            for t in range(1, MAX_DEGREE + 1):   # nombre de polynomes de time
                theta = np.ones(((w + d + t + 1), 1))

    ##################################################
    # 5 -features selection                          #
    ##################################################
                x_to_train = column_select(x_train, w, d, t)
                x_to_test = column_select(x_test, w, d, t)

    ##################################################
    # 6 -alpha choice                                #
    ##################################################
                # chosen_alpha = 1
                # j_chosen = 1e150 # arbitraire
                # for alpha in ALPHAS:
                #     mylr = MyLR(theta, alpha, max_iter=1500) \
                #           .fit_(x_to_train, y_train)
                #     if mylr is None:
                #         continue
                #     y_hat = mylr.predict_(x_to_train)
                #     j = mylr.mse_(y_train, y_hat)
                #     if j < j_chosen:
                #         # print("alpha =", alpha)
                #         # print("j= ", j)
                #         # print("jchosen = ", j_chosen)
                #         j_chosen = j
                #         chosen_alpha = alpha
                # print(bcolors.OK + "w" + str(w) + "d"
                #       + str(d) + "t" + str(t) + bcolors.RESET )
                # print("chosen alpha = ", chosen_alpha)
                # print("j_chosen =", j_chosen)

    ##################################################
    # 6 -number of iteration choice                  #
    ##################################################
                j_train = 1e150
                nb_iter = 1
                for max_it in ITERATIONS:
                    print("MAX = ", max_it)
                    mylr = MyLR(theta, alpha=ALPHA, max_iter=max_it) \
                        .fit_(x_to_train, y_train)
                    if mylr is None:
                        print(bcolors.FAIL + "w" + str(w) + "d"
                              + str(d) + "t" + str(t))
                        print("problem with this combination" + bcolors.RESET)
                        continue
                    y_hat = mylr.predict_(x_to_train)
                    j = mylr.mse_(y_train, y_hat)
                    if ((j_train - j) / j) < 1e-3:  # le cout n'evolue plus
                        print(bcolors.OK + "< 10-3" + bcolors.RESET)
                        j_train = j
                        nb_iter = max_it
                        break
                    j_train = j
                    nb_iter = max_it
                name = "w" + str(w) + "d" + str(d) + "t" + str(t)
                print(name)
                print("nb_iter =", nb_iter)
                print("j_train = ", j_train)
                y_hat_test = mylr.predict_(x_to_test)
                j_test = mylr.mse_(y_test, y_hat_test)
                print("j_test = ", j_test)
                with open("models.csv", "a") as fitted:
                    fitted.write(name + "," + str(ALPHA) + ","
                                 + str(nb_iter) + ","
                                 + str(j_train) + ","
                                 + str(j_test) + "\n")
