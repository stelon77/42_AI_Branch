import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_logistic_regression import MyLogisticRegression as MyLR
from utils import *

NB_LABELS = 4

try:
    dataX = pd.read_csv("solar_system_census.csv")
    dataY = pd.read_csv("solar_system_census_planets.csv")
    x = np.array(dataX[['weight', 'height', 'bone_density']])
    y = np.array(dataY[['Origin']])
except (FileNotFoundError, KeyError):
    print(bcolors.FAIL + "File or keys do not exist" + bcolors.RESET)
    exit(1)

m, n = x.shape
prediction = np.zeros((m, 1))

# split data
x_train, x_test, y_train, y_test = data_spliter(x, y, 0.7)  # prop 0.7

# normalize dataset x
x_train, mu, std = zscore_normalization(x_train)
x_test = normalize_others(x_test, mu, std)

all_theta = np.zeros((NB_LABELS, n + 1))
for K in range(NB_LABELS):
    thetas = np.ones((n + 1, 1))
    mylr = MyLR(thetas, alpha=0.5)
    y_zip_train = (y_train == K).astype(float)

    mylr = mylr.fit_(x_train, y_zip_train)
    all_theta[K, :] = mylr.theta.reshape((1, n + 1))

# avec x-test
X_test = mylr.add_intercept(x_test)
y_all_predict_test = sigmoid_predict_all(np.dot(X_test,
                                         np.transpose(all_theta)))
y_hat_test = np.argmax(y_all_predict_test, axis=1).reshape(x_test.shape[0], 1)
print(bcolors.OK + "\nOn the test dataset {:.2f}% of accuracy\n"
      .format(np.mean((y_hat_test == y_test).astype(float)) * 100)
      + bcolors.RESET)

#  prediction en probabilite de chaque label
x_norm = normalize_others(x, mu, std)
X_norm = mylr.add_intercept(x_norm)
y_all_predict = sigmoid_predict_all(np.dot(X_norm, np.transpose(all_theta)))
# on selectionne l'index de la prediction la plus haute
prediction = np.argmax(y_all_predict, axis=1).reshape(m, 1)
print(bcolors.OK + "\nOn the whole dataset {:.2f}% of accuracy\n"
      .format(np.mean((prediction == y).astype(float)) * 100)
      + bcolors.RESET)


venus = (prediction == 0).astype(int).nonzero()[0]
earth = (prediction == 1).astype(int).nonzero()[0]
mars = (prediction == 2).astype(int).nonzero()[0]
asteroid = (prediction == 3).astype(int).nonzero()[0]
true_venus = (y == 0).astype(int).nonzero()[0]
true_earth = (y == 1).astype(int).nonzero()[0]
true_mars = (y == 2).astype(int).nonzero()[0]
true_asteroid = (y == 3).astype(int).nonzero()[0]
plt.scatter(x[true_venus, 0], x[true_venus, 1], color="cyan",
            label="True Venus", marker="o")
plt.scatter(x[true_earth, 0], x[true_earth, 1], color="orange",
            label="True Earth", marker="o")
plt.scatter(x[true_mars, 0], x[true_mars, 1], color="red",
            label="True Mars", marker="o")
plt.scatter(x[true_asteroid, 0], x[true_asteroid, 1],
            color="blue", label="True Asteroid Belt", marker="o")
plt.scatter(x[venus, 0], x[venus, 1], color="cyan",
            label="Venus", marker=".")
plt.scatter(x[earth, 0], x[earth, 1], color="orange",
            label="Earth", marker=".")
plt.scatter(x[mars, 0], x[mars, 1], color="red",
            label="Mars", marker=".")
plt.scatter(x[asteroid, 0], x[asteroid, 1],
            color="blue", label="Asteroid Belt", marker=".")
plt.xlabel("Weight")
plt.ylabel("Height")
plt.legend()
plt.show()

plt.scatter(x[true_venus, 1], x[true_venus, 2], color="cyan",
            label="True Venus", marker="o")
plt.scatter(x[true_earth, 1], x[true_earth, 2], color="orange",
            label="True Earth", marker="o")
plt.scatter(x[true_mars, 1], x[true_mars, 2], color="red",
            label="True Mars", marker="o")
plt.scatter(x[true_asteroid, 1], x[true_asteroid, 2],
            color="blue", label="True Asteroid Belt", marker="o")
plt.scatter(x[venus, 1], x[venus, 2], color="cyan",
            label="Venus", marker=".")
plt.scatter(x[earth, 1], x[earth, 2], color="orange",
            label="Earth", marker=".")
plt.scatter(x[mars, 1], x[mars, 2], color="red",
            label="Mars", marker=".")
plt.scatter(x[asteroid, 1], x[asteroid, 2],
            color="blue", label="Asteroid Belt", marker=".")
plt.xlabel("Height")
plt.ylabel("Bone Density")
plt.legend()
plt.show()

plt.scatter(x[true_venus, 0], x[true_venus, 2], color="cyan",
            label="True Venus", marker="o")
plt.scatter(x[true_earth, 0], x[true_earth, 2], color="orange",
            label="True Earth", marker="o")
plt.scatter(x[true_mars, 0], x[true_mars, 2], color="red",
            label="True Mars", marker="o")
plt.scatter(x[true_asteroid, 0], x[true_asteroid, 2],
            color="blue", label="True Asteroid Belt", marker="o")
plt.scatter(x[venus, 0], x[venus, 2], color="cyan",
            label="Venus", marker=".")
plt.scatter(x[earth, 0], x[earth, 2], color="orange",
            label="Earth", marker=".")
plt.scatter(x[mars, 0], x[mars, 2], color="red",
            label="Mars", marker=".")
plt.scatter(x[asteroid, 0], x[asteroid, 2],
            color="blue", label="Asteroid Belt", marker=".")
plt.xlabel("Weight")
plt.ylabel("Bone Density")
plt.legend()
plt.show()
