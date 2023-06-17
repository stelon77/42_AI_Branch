#!/goinfre/lcoiffie/miniconda3/envs/42AI-lcoiffie/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv

from my_logistic_regression import MyLogisticRegression as MyLR
from utils import *

NB_HOUSES = 4
HOUSES = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
ALPHA = 0.5
MAX_ITER = 500


def display_usage():
    print(bcolors.WARNING
          + "USAGE : logreg_train.py take a dataset as argument, try again"
          + bcolors.RESET)
    exit(1)


if __name__ == '__main__':
    if len(argv) != 2:
        display_usage()
    try:
        df = pd.read_csv(argv[1])
        y = np.array(df[['Hogwarts House']])  # on a nos resultats
        df = df.select_dtypes(include='number')  # selectionne numerical values
        df = df.drop(['Index',
                      'Arithmancy',
                      'Care of Magical Creatures',
                      'Astronomy'],
                     axis=1)
        x = np.array(df)
    except (FileNotFoundError, KeyError):
        print(bcolors.FAIL + "File or keys do not exist" + bcolors.RESET)
        display_usage()

    # Variables
    m, n = x.shape
    prediction = np.zeros((m, 1))
    house_prediction = np.chararray((m, 1))
    all_theta = np.zeros((NB_HOUSES, n + 1))

    # normalize dataset and taking care of NaN
    x, mu, std = zscore_normalization(x)
    x = np.nan_to_num(x)

    for K in range(NB_HOUSES):
        thetas = np.ones((n + 1, 1))
        mylr = MyLR(thetas, alpha=ALPHA, max_iter=MAX_ITER)
        y_house = (y == HOUSES[K]).astype(float)
        mylr = mylr.fit_(x, y_house)
        all_theta[K, :] = mylr.theta.reshape((1, n + 1))

    #  prediction de chaque label
    X = mylr.add_intercept(x)
    y_all_predict = sigmoid_predict_all(np.dot(X, np.transpose(all_theta)))
    prediction = np.argmax(y_all_predict, axis=1).reshape(m, 1)
    # On transforme les predictions en strings, puis en noms
    house_prediction = prediction.astype(str)
    for i in range(NB_HOUSES):
        house_prediction[house_prediction == str(i)] = HOUSES[i]

    print(bcolors.OK + "\nOn the whole dataset {:.2f}% of accuracy\n"
          .format(np.mean((house_prediction == y).astype(float)) * 100)
          + bcolors.RESET)

######################################################################
#                     FILE WEIGHT.CSV CREATION                       #
######################################################################

    with open("weight.csv", "w") as fitted:
        fitted.write('Course,')
        for name in HOUSES:
            fitted.write(name + ',')
        fitted.write("mean,std\n")
        # ecriture premiere ligne
        fitted.write(',')
        for theta0 in all_theta[:, 0]:
            fitted.write(str(theta0) + ',')
        fitted.write(',\n')
        # ecriture autres lignes
        for index, name in enumerate(df.columns):
            fitted.write(name + ',')
            for theta in all_theta[:, index + 1]:
                fitted.write(str(theta) + ',')
            fitted.write(str(mu[index]) + ',' + str(std[index]) + '\n')
