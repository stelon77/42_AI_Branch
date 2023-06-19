import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_logistic_regression import MyLogisticRegression as MyLR
from utils import *


def usage(s):
    print(bcolors.FAIL + "\n" + s + bcolors.RESET)
    print(bcolors.WARNING
          + "Usage : mono_log.py needs an argument –zipcode=x (0<= x <4)\n"
          + bcolors.RESET)
    exit(1)


def check_args(arg):
    if len(arg) != 10 or \
       not arg.startswith("-zipcode="):
        usage("Argument is not correct")
    if not arg[9].isnumeric():
        usage("x has to be a number")
    n = int(arg[9])
    if n > 3 or n < 0:
        usage("x between 0 and 3")
    return float(n)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        usage("Wrong number of arguments")
    zip_code = check_args(sys.argv[1])
    # recuperer les datasets
    try:
        dataX = pd.read_csv("solar_system_census.csv")
        dataY = pd.read_csv("solar_system_census_planets.csv")
        x = np.array(dataX[['weight', 'height', 'bone_density']])
        y = np.array(dataY[['Origin']])
    except (FileNotFoundError, KeyError):
        print(bcolors.FAIL + "File or keys do not exist" + bcolors.RESET)
        exit(1)

    # split data
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.7)  # prop 0.7

    # normalize dataset x_train then x_test and the whole dataset
    x_train, mu, std = zscore_normalization(x_train)
    x_test = normalize_others(x_test, mu, std)

    # on cree l'instance de logistic regression
    thetas = np.ones((4, 1))
    mylr = MyLR(thetas, alpha=0.5)
    y_zip_train = (y_train == zip_code).astype(float)
    # entrainement du modele
    mylr = mylr.fit_(x_train, y_zip_train)

    # test du modele
    y_hat_test = mylr.predict_(x_test)
    y_hat_test = (y_hat_test >= 0.5)  # .astype(float)
    y_zip_test = (y_test == zip_code)  # .astype(float)
    true_pos = ((y_hat_test == True) & (y_zip_test == True)).astype(int)
    nb_TP = np.sum(true_pos)
    true_neg = ((y_hat_test == False) & (y_zip_test == False)).astype(int)
    nb_TN = np.sum(true_neg)
    false_pos = ((y_hat_test == True) & (y_zip_test == False)).astype(int)
    nb_FP = np.sum(false_pos)
    false_neg = ((y_hat_test == False) & (y_zip_test == True)).astype(int)
    nb_FN = np.sum(false_neg)
    m = len(y_hat_test)
    print(bcolors.OK
          + "correct predictions : {} upon {} examples for zip_code {} \
on test set"
          .format(nb_TP + nb_TN, m, int(zip_code)))
    print("accuracy on test set {:.2f}%"
          .format(np.mean((y_hat_test == y_zip_test).astype(float)) * 100)
          + bcolors.RESET)

    # plotting the model / results for the whole test
    x_norm = normalize_others(x, mu, std)
    y_hat = mylr.predict_(x_norm)
    y_hat = (y_hat >= 0.5)  # .astype(float)
    y_zip = (y == zip_code)
    print(bcolors.OK + "\nOn the whole dataset {:.2f}% of accuracy\n"
          .format(np.mean((y_hat == y_zip).astype(float)) * 100)
          + bcolors.RESET)
    tp = ((y_hat == True) & (y_zip == True)).astype(int).nonzero()[0]
    tn = ((y_hat == False) & (y_zip == False)).astype(int).nonzero()[0]
    fp = ((y_hat == True) & (y_zip == False)).astype(int).nonzero()[0]
    fn = ((y_hat == False) & (y_zip == True)).astype(int).nonzero()[0]
    # print(tp)
    plt.scatter(x[tp, 0], x[tp, 1], color="cyan",
                label="True Positive", marker=".")
    plt.scatter(x[tn, 0], x[tn, 1], color="orange",
                label="True Negative", marker=".")
    plt.scatter(x[fp, 0], x[fp, 1], color="red",
                label="False Positive", marker=".")
    plt.scatter(x[fn, 0], x[fn, 1], color="blue",
                label="False Negative", marker=".")
    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.legend()
    plt.show()

    plt.scatter(x[tp, 1], x[tp, 2], color="cyan",
                label="True Positive", marker=".")
    plt.scatter(x[tn, 1], x[tn, 2], color="orange",
                label="True Negative", marker=".")
    plt.scatter(x[fp, 1], x[fp, 2], color="red",
                label="False Positive", marker=".")
    plt.scatter(x[fn, 1], x[fn, 2], color="blue",
                label="False Negative", marker=".")
    plt.ylabel("Bone Density")
    plt.xlabel("Height")
    plt.legend()
    plt.show()

    plt.scatter(x[tp, 0], x[tp, 2], color="cyan",
                label="True Positive", marker=".")
    plt.scatter(x[tn, 0], x[tn, 2], color="orange",
                label="True Negative", marker=".")
    plt.scatter(x[fp, 0], x[fp, 2], color="red",
                label="False Positive", marker=".")
    plt.scatter(x[fn, 0], x[fn, 2], color="blue",
                label="False Negative", marker=".")
    plt.xlabel("Weight")
    plt.ylabel("Bone Density")
    plt.legend()
    plt.show()
