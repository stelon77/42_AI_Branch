from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

FILENAME = "../are_blue_pills_magics.csv"
X_KEY = "Micrograms"
Y_KEY = "Score"
theta1 = np.array([[1.0], [1.0]]).reshape(-1, 1)
theta2 = np.array([[1.0], [1.0], [1.0]]).reshape(-1, 1)
theta3 = np.array([[1.0], [1.0], [1.0], [1.0]]).reshape(-1, 1)
theta4 = np.array([[-20], [160], [-80], [10], [-1]]).reshape(-1, 1)
theta5 = np.array([[1140], [-1850], [1110], [-305], [40], [-2]]).reshape(-1, 1)
theta6 = np.array([[9110], [-18015], [13400], [-4935],
                   [966], [-96.4], [3.86]]).reshape(-1, 1)

# read dataset from csv file
# and extract datas
try:
    data = pd.read_csv(FILENAME)
    x = np.array(data[X_KEY]).reshape(-1, 1)
    y = np.array(data[Y_KEY]).reshape(-1, 1)
except (FileNotFoundError, KeyError):
    print("File or keys do not exist")
    exit(1)


continuous_x = np.arange(1.1, 6.5, 0.01).reshape(-1, 1)
mse = np.zeros((6, 1))

try:
    # let's take rank 1 !
    x_1 = add_polynomial_features(x, 1)
    x_1_cont = add_polynomial_features(continuous_x, 1)
    my_lr = MyLR(theta1, alpha=1e-2, max_iter=10000).fit_(x_1, y)
    y_hat_cont_1 = my_lr.predict_(x_1_cont)
    y_hat_x_1 = my_lr.predict_(x_1)
    plt.plot(continuous_x, y_hat_cont_1, color='orange', label="degree 1")
    mse[0] = my_lr.mse_(y_hat_x_1, y)

    # let's take rank 2!
    x_2 = add_polynomial_features(x, 2)
    x_2_cont = add_polynomial_features(continuous_x, 2)
    my_lr = MyLR(theta2, alpha=2e-3, max_iter=100000).fit_(x_2, y)
    y_hat_cont_2 = my_lr.predict_(x_2_cont)
    y_hat_x_2 = my_lr.predict_(x_2)
    plt.plot(continuous_x, y_hat_cont_2, color='red', label="degree 2")
    mse[1] = my_lr.mse_(y_hat_x_2, y)

    # let's take rank 3!
    x_3 = add_polynomial_features(x, 3)
    x_3_cont = add_polynomial_features(continuous_x, 3)
    my_lr = MyLR(theta3, alpha=8.1e-5, max_iter=5000000).fit_(x_3, y)
    y_hat_cont_3 = my_lr.predict_(x_3_cont)
    y_hat_x_3 = my_lr.predict_(x_3)
    plt.plot(continuous_x, y_hat_cont_3, color='cyan', label="degree 3")
    mse[2] = my_lr.mse_(y_hat_x_3, y)

    # let's take rank 4!
    x_4 = add_polynomial_features(x, 4)
    x_4_cont = add_polynomial_features(continuous_x, 4)
    my_lr = MyLR(theta4, alpha=1e-6, max_iter=500000).fit_(x_4, y)
    y_hat_cont_4 = my_lr.predict_(x_4_cont)
    y_hat_x_4 = my_lr.predict_(x_4)
    plt.plot(continuous_x, y_hat_cont_4, color='blue', label="degree 4")
    mse[3] = my_lr.mse_(y_hat_x_4, y)

    # let's take rank 5!
    x_5 = add_polynomial_features(x, 5)
    x_5_cont = add_polynomial_features(continuous_x, 5)
    my_lr = MyLR(theta5, alpha=5e-8, max_iter=3000000).fit_(x_5, y)
    y_hat_cont_5 = my_lr.predict_(x_5_cont)
    y_hat_x_5 = my_lr.predict_(x_5)
    plt.plot(continuous_x, y_hat_cont_5, color='darkblue', label="degree 5")
    mse[4] = my_lr.mse_(y_hat_x_5, y)

    # let's take rank 6!
    x_6 = add_polynomial_features(x, 6)
    x_6_cont = add_polynomial_features(continuous_x, 6)
    my_lr = MyLR(theta6, alpha=1e-9, max_iter=1000000).fit_(x_6, y)
    y_hat_cont_6 = my_lr.predict_(x_6_cont)
    y_hat_x_6 = my_lr.predict_(x_6)
    plt.plot(continuous_x, y_hat_cont_6, color='darkred', label="degree 6")
    mse[5] = my_lr.mse_(y_hat_x_6, y)
except AttributeError:
    print("can't calculate mse as the linear regression is diverging")

# print mse score
print("\n MSE Scores :")
for index, value in enumerate(mse):
    print(" Polynomial degree {}, mse score : {}"
          .format(index + 1, float(value)))

plt.scatter(x, y)
plt.xlabel("Quantity of blue pills (in micrograms)")
plt.ylabel("Space driving score")
# plt.grid()
plt.legend()
plt.show()

list_mse = list(mse.flatten())
abciss = list(range(1, 7))

plt.bar(abciss, list_mse, color='maroon')
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE Score")
plt.title("Variation of MSE Score with polynomial degree")
plt.show()
