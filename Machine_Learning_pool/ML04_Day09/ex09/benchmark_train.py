import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_logistic_regression import MyLogisticRegression as MyLR
from utils import *

NB_LABELS = 4
POWER = 3
ALPHA = 0.5
MAX_ITER = 10000

try:
    dataX = pd.read_csv("solar_system_census.csv")
    dataY = pd.read_csv("solar_system_census_planets.csv")
    x = np.array(dataX[['height', 'weight', 'bone_density']])
    y = np.array(dataY[['Origin']])
except (FileNotFoundError, KeyError):
    print(bcolors.FAIL + "File or keys do not exist" + bcolors.RESET)
    exit(1)

# create model file
with open("models.csv", "w") as fitted:
    fitted.write("power,alpha,max_iter,lambda,F1Score\n")

# add polynomial
ext_x = add_polynomial_features(x, POWER)

m, n = ext_x.shape

# split data

x_train, x_others, y_train, y_others = data_spliter(ext_x, y, 0.6)  # prop 0.6
x_crossval, x_test, y_crossval, y_test = data_spliter(x_others, y_others, 0.5)

# normalize dataset x
x_train, mu, std = zscore_normalization(x_train)
x_crossval = normalize_others(x_crossval, mu, std)
x_test = normalize_others(x_test, mu, std)

# la boucle
for lambda_ in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    all_theta = np.zeros((NB_LABELS, n + 1))

    for K in range(NB_LABELS):
        thetas = np.ones((n + 1, 1))
        mylr = MyLR(thetas, alpha=ALPHA, lambda_=lambda_, max_iter=MAX_ITER)
        y_zip_train = (y_train == K).astype(float)

        mylr = mylr.fit_(x_train, y_zip_train)
        all_theta[K, :] = mylr.theta.reshape((1, n + 1))

    # on a entraine un modele pour un lambda
    X_train = mylr.add_intercept(x_train)
    y_all_predict_train = mylr.sigmoid_(np.dot(X_train,
                                        np.transpose(all_theta)))
    y_hat_train = (np.argmax(y_all_predict_train, axis=1)
                   .reshape(x_train.shape[0], 1))
    print(bcolors.OK
          + "\nWith lambda = {} on the train dataset {:.2f}% of accuracy"
          .format(lambda_, np.mean((y_hat_train == y_train)
                  .astype(float)) * 100)
          + bcolors.RESET)
    # calcul des resultats sur le crossval
    X_cross = mylr.add_intercept(x_crossval)
    y_all_predict_cross = mylr.sigmoid_(np.dot(X_cross,
                                        np.transpose(all_theta)))
    y_hat_cross = (np.argmax(y_all_predict_cross, axis=1)
                   .reshape(x_crossval.shape[0], 1))
    print(bcolors.OK
          + "With lambda = {} on the cross dataset {:.2f}% of accuracy"
          .format(lambda_, np.mean((y_hat_cross == y_crossval)
                  .astype(float)) * 100)
          + bcolors.RESET)
    # et sur le test set
    X_test = mylr.add_intercept(x_test)
    y_all_predict_test = mylr.sigmoid_(np.dot(X_test,
                                       np.transpose(all_theta)))
    y_hat_test = (np.argmax(y_all_predict_test, axis=1)
                  .reshape(x_test.shape[0], 1))
    print(bcolors.OK
          + "With lambda = {} on the test dataset {:.2f}% of accuracy"
          .format(lambda_, np.mean((y_hat_test == y_test).astype(float)) * 100)
          + bcolors.RESET)

    # calcul F1score sur le cross val
    f1score = 0
    for i in range(NB_LABELS):
        f1score += f1_score_(y_crossval, y_hat_cross, pos_label=i)
    f1score /= NB_LABELS
    print(bcolors.OK + "F1 score = {}".format(f1score))

    # sauvegarde dans models.csv
    with open("models.csv", "a") as fitted:
        fitted.write(str(POWER) + "," + str(ALPHA) + ","
                     + str(MAX_ITER) + "," + str(lambda_) + ","
                     + str(f1score) + "\n")
