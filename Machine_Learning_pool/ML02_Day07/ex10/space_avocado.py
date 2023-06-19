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

FILENAME2 = "./models.csv"


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


def data_distance_graph(x, y, y_hat):
    plt.scatter(x, y, marker='o', color='green', label="Order price")
    plt.scatter(x, y_hat, marker='.', color='lime',
                label="Predicted price")
    plt.legend(loc="upper left")
    plt.xlabel(r"$x_{2}:\ production\ distance\ (in\ Mkm)$")
    plt.ylabel("y: price (in tratorian units)")
    plt.grid()
    plt.show()


def data_weight_graph(x, y, y_hat):
    plt.scatter(x, y, marker='o', color='darkblue', label="Order price")
    plt.scatter(x, y_hat, marker='.', color='cornflowerblue',
                label="Predicted price")
    plt.legend(loc="lower left")
    plt.xlabel(r"$x_{1}:\ avocado\ weight\ (in\ tons)$")
    plt.ylabel("y: price (in tratorian units)")
    plt.grid()
    plt.show()


def data_time_graph(x, y, y_hat):
    plt.scatter(x, y, marker='o', color='darkviolet', label="Order price")
    plt.scatter(x, y_hat, marker='.', color='violet',
                label="Predicted price")
    plt.legend(loc="upper right")
    plt.xlabel(
        (r"$x_{3}:\ delivery\ time\ (in\ days)$")
    )
    plt.ylabel("y: price (in tratorian units)")
    plt.grid()
    plt.show()


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

    # 2-extraire lenmeilleur modele ( le plus petit J test)
    best = df.loc[df['J_test'] == df["J_test"].min()]
    name = best.iloc[0]["name"]
    alpha = float(best.iloc[0]["alpha"])
    max_iter = int(best.iloc[0]["max_iter"])
    print(bcolors.OK + "the best model is : " + name + bcolors.RESET)

    # 3-transformer les donnees de space avocado en fonction du nom du modele
    w = int(name[1])
    d = int(name[3])
    t = int(name[5])
    x_weight = add_polynomial_features(x_weight, w)
    x_dist = add_polynomial_features(x_dist, d)
    x_time = add_polynomial_features(x_time, t)
    x_all = np.concatenate((x_weight, x_dist, x_time), axis=1)
    x_all, mu, std = zscore_normalization(x_all)
    theta = np.ones(((w + d + t + 1), 1))
    x_train, x_test, y_train, y_test = data_spliter(x_all, y, 0.8)

    # 4-lancer le modele - confirmation des couts tests et train
    mylr = MyLR(theta, alpha=alpha, max_iter=max_iter).fit_(x_train, y_train)
    y_hat_train = mylr.predict_(x_train)
    y_hat_test = mylr.predict_(x_test)
    j_train = mylr.mse_(y_train, y_hat_train)
    j_test = mylr.mse_(y_test, y_hat_test)
    print(bcolors.OK + "j_train = " + str(j_train))
    print("j_test = " + str(j_test))
    print(bcolors.WARNING + "They should be "
          + str(best.iloc[0]["J_train"]) + " and "
          + str(best.iloc[0]["J_test"]) + bcolors.RESET)
    y_hat_all = mylr.predict_(x_all)

    # 5-extraire nom, Jtrain et J test
    costs = df[["name", "J_train", "J_test"]]

    # 7-mettre Jtrain en ordre decroissant
    sorted_costs = df.sort_values(by='J_test')
    # sorted_test_cost = df.sort_values(by="J_test")
    # print(sorted_test_cost.head(15))

    # 8-afficher courbe des J
    names = np.array(costs["name"])
    J_tests = np.array(costs["J_test"])
    J_tests_sorted = np.array(sorted_costs["J_test"])
    names_sorted = np.array(sorted_costs["name"])

    plt.scatter(names, J_tests)
    plt.ylabel("MSE of test set")
    plt.title("MSE vs models")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

    plt.scatter(names_sorted, J_tests_sorted)
    plt.ylabel("MSE of test set")
    plt.title("MSE vs models")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

    plt.scatter(names_sorted[:12], J_tests_sorted[:12])
    plt.ylabel("MSE of test set")
    plt.title("MSE vs models")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

    # 9-afficher prix reels et prix calcules
    data_weight_graph(x_weight[:, 0], y, y_hat_all)
    data_distance_graph(x_dist[:, 0], y, y_hat_all)
    data_time_graph(x_time[:, 0], y, y_hat_all)
