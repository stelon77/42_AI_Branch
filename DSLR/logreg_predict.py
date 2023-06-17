#!/goinfre/lcoiffie/miniconda3/envs/42AI-lcoiffie/bin/python

import numpy as np
import pandas as pd
from sys import argv


NB_HOUSES = 4
HOUSES = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']


def add_intercept(x):
    ones = np.ones((x.shape[0], 1))
    return np.concatenate((ones, x), axis=1)


def sigmoid_predict_all(x):
    return (1 / (1 + np.exp(-x)))


def normalize_others(x, mu, std):
    X = np.divide(np.subtract(x, mu), std)
    return X


def display_usage():
    print(bcolors.WARNING
          + "USAGE : logreg_predict.py take a .csv dataset as argument\n \
and a weight.csv file created by logreg_train.py"
          + bcolors.RESET)
    exit(1)


if __name__ == '__main__':
    if len(argv) != 3:
        display_usage()
    if argv[2] != 'weight.csv':
        display_usage()

    try:
        df = pd.read_csv(argv[1])
        weight = pd.read_csv(argv[2])
        courses = list(weight['Course'].dropna())
        all_theta = weight[HOUSES]
        mudf = weight['mean'].dropna()
        mu = np.array(mudf).reshape(1, len(mudf))
        stddf = weight['std'].dropna()
        std = np.array(stddf).reshape(1, len(stddf))

    except (FileNotFoundError, KeyError):
        print(bcolors.FAIL + "File or keys do not exist" + bcolors.RESET)
        display_usage()

    df = df[courses]
    x = np.array(df)

    # variables
    m, n = x.shape
    prediction = np.zeros((m, 1))
    house_prediction = np.chararray((m, 1))

    # x normsalization and preparation
    x = normalize_others(x, mu, std)
    x = np.nan_to_num(x)
    X = add_intercept(x)
    y_predict = sigmoid_predict_all(np.dot(X, all_theta))
    # # on selectionne l'index de la prediction la plus haute
    prediction = np.argmax(y_predict, axis=1).reshape(m, 1)
    house_prediction = prediction.astype(str)
    for i in range(NB_HOUSES):
        house_prediction[house_prediction == str(i)] = HOUSES[i]

    with open("houses.csv", "w") as fitted:
        fitted.write('Index,Hogwarts House')
        for index, house in enumerate(house_prediction[:, 0]):
            fitted.write('\n' + str(index) + ',' + house)
