import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR


def data_thrust_graph(x, y, y_hat):
    plt.scatter(x, y, marker='o', color='green', label="Sell price")
    plt.scatter(x, y_hat, marker='.', color='lime',
                label="Predicted sell price")
    plt.legend(loc="upper left")
    plt.xlabel(r"$x_{2}:\ thrust\ power\ (in\ 10Km/s)$")
    plt.ylabel("y: sell price (in keuros)")
    plt.grid()
    plt.show()


def data_age_graph(x, y, y_hat):
    plt.scatter(x, y, marker='o', color='darkblue', label="Sell price")
    plt.scatter(x, y_hat, marker='.', color='cornflowerblue',
                label="Predicted sell price")
    plt.legend(loc="lower left")
    plt.xlabel(r"$x_{1}:\ age\ (in\ years)$")
    plt.ylabel("y: sell price (in keuros)")
    plt.grid()
    plt.show()


def data_distance_graph(x, y, y_hat):
    plt.scatter(x, y, marker='o', color='darkviolet', label="Sell price")
    plt.scatter(x, y_hat, marker='.', color='violet',
                label="Predicted sell price")
    plt.legend(loc="upper right")
    plt.xlabel(
        (r"$x_{3}:\ distance\ totalizer\ value\ of\ spacecraft(in\ Tmeters)$")
    )
    plt.ylabel("y: sell price (in keuros)")
    plt.grid()
    plt.show()


if __name__ == '__main__':

    data = pd.read_csv("../spacecraft_data.csv")

    # AGE
    X = np.array(data[['Age']])
    Y = np.array(data[['Sell_price']])
    myLR_age = MyLR(thetas=[[1000.0], [-1.0]], alpha=2.5e-5, max_iter=100000)
    myLR_age.fit_(X[:, 0].reshape(-1, 1), Y)
    Y_hat = myLR_age.predict_(X)
    print("\n#######################################\n")
    print("UNIVARIATE LINEAR REGRESSION : AGE")
    print("thetas = ", myLR_age.thetas.flatten())
    print("MSE = ", myLR_age.mse_(X[:, 0].reshape(-1, 1), Y))
    # Output
    """55736.86719..."""
    data_age_graph(X, Y, Y_hat)

    # THRUST POWER
    X = np.array(data[['Thrust_power']])
    Y = np.array(data[['Sell_price']])
    myLR_thrust = MyLR(thetas=[[0.0], [5.0]], alpha=1e-4, max_iter=50000)
    myLR_thrust.fit_(X[:, 0].reshape(-1, 1), Y)
    Y_hat = myLR_thrust.predict_(X)
    print("\n#######################################\n")
    print("UNIVARIATE LINEAR REGRESSION : THRUST POWER")
    print("thetas = ", myLR_thrust.thetas.flatten())
    print("MSE = ", myLR_thrust.mse_(X[:, 0].reshape(-1, 1), Y))
    data_thrust_graph(X, Y, Y_hat)

    # DISTANCE
    X = np.array(data[['Terameters']])
    Y = np.array(data[['Sell_price']])
    myLR_distance = MyLR(thetas=[[700.0], [-2.0]], alpha=1e-4,
                         max_iter=50000)
    myLR_distance.fit_(X[:, 0].reshape(-1, 1), Y)
    Y_hat = myLR_distance.predict_(X)
    print("\n#######################################\n")
    print("UNIVARIATE LINEAR REGRESSION : DISTANCE")
    print("thetas = ", myLR_distance.thetas.flatten())
    print("MSE = ", myLR_distance.mse_(X[:, 0].reshape(-1, 1), Y))
    data_distance_graph(X, Y, Y_hat)

    # NOW MULTIVARIATE
    X = np.array(data[['Age', 'Thrust_power', 'Terameters']])
    Y = np.array(data[['Sell_price']])
    my_lreg = MyLR(thetas=[1.0, 1.0, 1.0, 1.0], alpha=5e-5, max_iter=600000)

    # my_lreg = MyLR(thetas=[1.0, 1.0, 1.0, 1.0], alpha=6e-5, max_iter=500000)

    print("\n\n#######################################\n")
    print("MULTIVARIATE LINEAR REGRESSION : INITIAL STATE")
    print("thetas = ", my_lreg.thetas.flatten())
    print("MSE = ", my_lreg.mse_(X, Y))
    # Output:
    """144044.877..."""
    # Example 1:
    my_lreg.fit_(X, Y)
    print("\n\n#######################################\n")
    print("MULTIVARIATE LINEAR REGRESSION : AFTER FITTING")
    print("thetas = ", my_lreg.thetas.flatten())
    print("MSE = ", my_lreg.mse_(X, Y))
    # # Output:
    # """array([[334.994...],[-22.535...],[5.857...],[-2.586...]])"""
    # """586.896999..."""
    Y_hat = my_lreg.predict_(X)
    data_age_graph(X[:, 0], Y, Y_hat)
    data_thrust_graph(X[:, 1], Y, Y_hat)
    data_distance_graph(X[:, 2], Y, Y_hat)
