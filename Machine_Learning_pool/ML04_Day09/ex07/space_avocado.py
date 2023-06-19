from cProfile import label
from ressources.polynomial_model import add_polynomial_features
from ressources.ridge import MyRidge as MyR
from ressources.data_spliter import data_spliter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

FILENAME = "./space_avocado.csv"
X_WEIGHT = "weight"
X_DIST = "prod_distance"
X_TIME = "time_delivery"
Y_KEY = "target"

FILENAME2 = "./models.csv"
TRAINING_PROPORTION = 0.6
TEST_SET_SPLITTING = 0.5


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


def normalize_test_set(x, mu, std):
    X = np.divide(np.subtract(x, mu), std)
    return X


def data_distance_graph(x, y, y_hat, lambda_):
    plt.scatter(x, y, marker='o', color='green', label="Order price")
    plt.scatter(x, y_hat, marker='.', color='lime',
                label="Predicted price")
    plt.legend(loc="upper left")
    plt.xlabel(r"$x_{2}:\ production\ distance\ (in\ Mkm)$")
    plt.ylabel("y: price (in tratorian units)")
    plt.title("lambda = {}".format(lambda_))
    plt.grid()
    # plt.show()


def data_weight_graph(x, y, y_hat, lambda_):
    plt.scatter(x, y, marker='o', color='darkblue', label="Order price")
    plt.scatter(x, y_hat, marker='.', color='cornflowerblue',
                label="Predicted price")
    plt.legend(loc="lower left")
    plt.xlabel(r"$x_{1}:\ avocado\ weight\ (in\ tons)$")
    plt.ylabel("y: price (in tratorian units)")
    plt.title("lambda = {}".format(lambda_))
    plt.grid()
    # plt.show()


def data_time_graph(x, y, y_hat, lambda_):
    plt.scatter(x, y, marker='o', color='darkviolet', label="Order price")
    plt.scatter(x, y_hat, marker='.', color='violet',
                label="Predicted price")
    plt.legend(loc="upper right")
    plt.xlabel(
        (r"$x_{3}:\ delivery\ time\ (in\ days)$")
    )
    plt.ylabel("y: price (in tratorian units)")
    plt.title("lambda = {}".format(lambda_))
    plt.grid()
    # plt.show()


if __name__ == "__main__":

    # 1-recuperer les donnees de space avocado & models
    try:
        data = pd.read_csv(FILENAME)
        x_weight = np.array(data[X_WEIGHT]).reshape(-1, 1)
        x_dist = np.array(data[X_DIST]).reshape(-1, 1)
        x_time = np.array(data[X_TIME]).reshape(-1, 1)
        y = np.array(data[Y_KEY]).reshape(-1, 1)
        df = pd.read_csv(FILENAME2)
    except (FileNotFoundError, KeyError):
        print(bcolors.FAIL + "File or keys do not exist" + bcolors.RESET)
        exit(1)

    # 2-extraire lenmeilleur modele ( le plus petit J cross)
    # find_best = df.loc[df['lambda']== LAMBDA]
    best = df.loc[df['J_cross'] == df["J_cross"].min()]
    name = best.iloc[0]["name"]
    alpha = float(best.iloc[0]["alpha"])
    max_iter = int(best.iloc[0]["max_iter"])
    best_lambda = float(best.iloc[0]["lambda"])
    print(bcolors.OK + "the best model is : " + name +
          ' with lambda = ' + str(best_lambda) + bcolors.RESET)

    # 3-transformer les donnees de space avocado en fonction du nom du modele
    w = int(name[1])
    d = int(name[3])
    t = int(name[5])
    x_weight = add_polynomial_features(x_weight, w)
    x_dist = add_polynomial_features(x_dist, d)
    x_time = add_polynomial_features(x_time, t)
    x_all = np.concatenate((x_weight, x_dist, x_time), axis=1)
    theta = np.ones(((w + d + t + 1), 1))

    #  3bis-split the datas
    x_train, x_other, y_train, y_other = \
        data_spliter(x_all, y, TRAINING_PROPORTION)
    x_cross_val, x_test, y_cross_val, y_test = \
        data_spliter(x_other, y_other, TEST_SET_SPLITTING)

    #  3ter-normalize datas
    x_train, mu, std = zscore_normalization(x_train)
    x_cross_val = normalize_test_set(x_cross_val, mu, std)
    x_test = normalize_test_set(x_test, mu, std)
    norm_x_all = normalize_test_set(x_all, mu, std)

    # 4-lancer le modele - confirmation des couts tests et train
    predictions = np.zeros((norm_x_all.shape[0], 6))
    for index, l in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
        myr = (MyR(theta, alpha=alpha, max_iter=max_iter, lambda_=l)
               .fit_(x_train, y_train))
        y_hat_train = myr.predict_(x_train)
        j_train = myr.mse_(y_train, y_hat_train)
        myr.set_params_({"lambda_": 0.0})
        y_hat_cross = myr.predict_(x_cross_val)
        y_hat_test = myr.predict_(x_test)
        y_hat_all = myr.predict_(norm_x_all).reshape(norm_x_all.shape[0], 1)
        predictions[:, [index]] = y_hat_all
        j_cross = myr.mse_(y_cross_val, y_hat_cross)
        j_test = myr.mse_(y_test, y_hat_test)
        print(bcolors.OK + "lambda = " + str(l))
        print("j_train = " + str(j_train))
        print("j_cross = " + str(j_cross))
        print("j_test = " + str(j_test) + bcolors.RESET)

    # 5-extraire nom, Jtrain et J test
    costs = df[["name", "lambda", "J_train", "J_cross", ]]
    names = np.array(costs['name'])
    J_trains = np.array(costs['J_train'])
    J_crosses = np.array(costs['J_cross'])

    # totalite des mesures
    plt.scatter(names, J_trains)
    plt.scatter(names, J_crosses, marker='.')
    plt.ylabel("J train vs J cross")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

    # en mettant dans l'ordre les trains(24 meilleurs)
    sorted_train = costs.sort_values(by='J_train', ascending=False).tail(24)
    names = np.array(sorted_train['name'])
    J_crosses = np.array(sorted_train['J_cross'])
    J_trains = np.array(sorted_train['J_train'])

    plt.scatter(names, J_trains)
    plt.scatter(names, J_crosses, marker='.')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

    # 9-afficher prix reels et prix calcules
    fig = plt.figure(tight_layout=True)
    plt.subplot(3, 2, 1)
    data_weight_graph(x_weight[:, 0], y, predictions[:, 0], 0.0)
    plt.subplot(3, 2, 2)
    data_weight_graph(x_weight[:, 0], y, predictions[:, 1], 0.2)
    plt.subplot(3, 2, 3)
    data_weight_graph(x_weight[:, 0], y, predictions[:, 2], 0.4)
    plt.subplot(3, 2, 4)
    data_weight_graph(x_weight[:, 0], y, predictions[:, 3], 0.6)
    plt.subplot(3, 2, 5)
    data_weight_graph(x_weight[:, 0], y, predictions[:, 4], 0.8)
    plt.subplot(3, 2, 6)
    data_weight_graph(x_weight[:, 0], y, predictions[:, 5], 1.0)
    plt.show()

    fig = plt.figure(tight_layout=True)
    plt.subplot(3, 2, 1)
    data_distance_graph(x_dist[:, 0], y, predictions[:, 0], 0.0)
    plt.subplot(3, 2, 2)
    data_distance_graph(x_dist[:, 0], y, predictions[:, 1], 0.2)
    plt.subplot(3, 2, 3)
    data_distance_graph(x_dist[:, 0], y, predictions[:, 2], 0.4)
    plt.subplot(3, 2, 4)
    data_distance_graph(x_dist[:, 0], y, predictions[:, 3], 0.6)
    plt.subplot(3, 2, 5)
    data_distance_graph(x_dist[:, 0], y, predictions[:, 4], 0.8)
    plt.subplot(3, 2, 6)
    data_distance_graph(x_dist[:, 0], y, predictions[:, 5], 1.0)
    plt.show()

    fig = plt.figure(tight_layout=True)
    plt.subplot(3, 2, 1)
    data_time_graph(x_time[:, 0], y, predictions[:, 0], 0.0)
    plt.subplot(3, 2, 2)
    data_time_graph(x_time[:, 0], y, predictions[:, 1], 0.2)
    plt.subplot(3, 2, 3)
    data_time_graph(x_time[:, 0], y, predictions[:, 2], 0.4)
    plt.subplot(3, 2, 4)
    data_time_graph(x_time[:, 0], y, predictions[:, 3], 0.6)
    plt.subplot(3, 2, 5)
    data_time_graph(x_time[:, 0], y, predictions[:, 4], 0.8)
    plt.subplot(3, 2, 6)
    data_time_graph(x_time[:, 0], y, predictions[:, 5], 1.0)
    plt.show()
