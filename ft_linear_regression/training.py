import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from random import random


class bcolors:
    OK = '\033[92m'  # green
    WARNING = '\033[93m'  # yellow
    FAIL = '\033[91m'  # red
    RESET = '\033[0m'  # reset color


class MyLinearRegression:
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        if not isinstance(alpha, (float, int)) or \
           not isinstance(max_iter, int) or \
           not isinstance(theta, (np.ndarray, list, tuple)):
            print(bcolors.FAIL + "bad arguments provided" + bcolors.RESET)
            exit(1)
        self.theta = np.array(theta)

    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
        x: has to be a numpy.array, a matrix of shape m * n:
        (number of training examples, number of features).
        y: has to be a numpy.array, a vector of shape m * 1:
        (number of training examples, 1).
        Return:
        new_theta: numpy.array, a vector of shape (number of features + 1, 1).
        None if there is a matching shape problem.
        None if x, y is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not MyLinearRegression.check_the_inputs(x, self.theta):
            return None
        self.theta = MyLinearRegression.reshape_if_needed(self.theta)
        if not MyLinearRegression.is_vector_not_empty_numpy(y):
            return None
        y = MyLinearRegression.reshape_if_needed(y)
        if x.shape[0] != y.shape[0]:
            return None
        if self.alpha <= 0 or self.max_iter <= 0:
            return None
        J_history = np.zeros((max_iter, 1))
        new_theta = self.theta
        for i in range(self.max_iter):
            new_theta = (new_theta -
                         (self.alpha *
                          MyLinearRegression.gradient(x, y, new_theta)))
            if not self.check_thetas(new_theta):
                print(bcolors.WARNING
                      + "thetas is NaN, you should try a \
smaller learning rate "
                      + bcolors.RESET)
                return None
            J_history[i] = self.compute_cost(x, y, new_theta)
        self.theta = new_theta
        return J_history

    def predict_(self, x):
        """Computes the prediction vector y_hat from two non-empty numpy.array.
        Args:
        x: has to be an numpy.array, a vector of shapes m * n.
        theta: has to be an numpy.array, a vector of shapes (n + 1) * 1.
        Return:
        y_hat as a numpy.array, a vector of shapes m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta shapes are not appropriate.
        None if x or theta is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not MyLinearRegression.check_the_inputs(x, self.theta):
            return None
        self.theta = MyLinearRegression.reshape_if_needed(self.theta)
        X = MyLinearRegression.add_intercept(x)
        return (np.dot(X, self.theta))

    def loss_elem_(self, x, y):
        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
        x: has to be an numpy.array, a matrix.
        y: has to be an numpy.array, a vector.
        Returns:
        J_elem: numpy.array, a vector of dimension
        (number of the training examples,1).
        None if there is a dimension matching problem between y and x.
        None if y or x is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not MyLinearRegression.check_the_inputs(x, self.theta):
            return None
        self.theta = MyLinearRegression.reshape_if_needed(self.theta)
        if not MyLinearRegression.is_vector_not_empty_numpy(y):
            return None
        y = MyLinearRegression.reshape_if_needed(y)
        if x.shape[0] != y.shape[0]:
            return None
        y_hat = MyLinearRegression.predict_(self, x)
        if y_hat is None:
            return None
        dif = y_hat - y
        return(np.square(dif))

    def loss_(self, x, y):
        """
        Description:
        Calculates the value of loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_value : has to be a float.
        None if there is a shape matching problem between y or y_hat.
        None if y or y_hat is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        J_elem = MyLinearRegression.loss_elem_(self, x, y)
        if J_elem is None or len(J_elem) == 0:
            return None
        J_value = np.sum(J_elem) / (2 * len(J_elem))
        return float(J_value)

    def mse_(self, x, y):
        """
        Description:
        Calculates the value of mse loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_value : has to be a float.
        None if there is a shape matching problem between y or y_hat.
        None if y or y_hat is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        return (MyLinearRegression.loss_(self, x, y) * 2)

    def data_hypothesis_graph(self, x, y, y_hat, J_history):
        ax1 = plt.subplot(121)
        ax1.scatter(x, y, marker='.')
        ax1.plot(x, y_hat, color='limegreen',
                 label=r"${\theta}_{0} = $" + str(self.theta[0][0])
                 + "\n" + r"${\theta}_{1} = $" + str(self.theta[1][0]))
        ax1.legend(loc="upper right")
        ax1.set_xlabel("Mileage(km)")
        ax1.set_ylabel("Price")

        ax2 = plt.subplot(122)
        X = np.array(range(self.max_iter))
        ax2.plot(X, J_history, color="blue")
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Cost")

        plt.show()

    @staticmethod
    def gradient(x, y, theta):
        """Computes a gradient vector from three
        non-empty numpy.array, without any for-loop.
        The three arrays must have the compatible shapes.
        Args:
        x: has to be an numpy.array, a matrix of shape m * n.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
        Return:
        The gradient as a numpy.array, a vector of shapes n * 1,
        containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of expected type.
        Raises:
        This function should not raise any Exception.
        """
        X = MyLinearRegression.add_intercept(x)
        m = x.shape[0]
        return np.dot(np.transpose(X), (np.dot(X, theta) - y)) / m

    @staticmethod
    def add_intercept(x):
        """Adds a column of 1's to the non-empty numpy.array x.
        Args:
        x: has to be an numpy.array, a vector of shape m * n.
        Returns:
        x as a numpy.array, a vector of shape m * (n + 1).
        None if x is not a numpy.array.
        None if x is a empty numpy.array.
        Raises:
        This function should not raise any Exception.
        """
        ones = np.ones((len(x), 1))
        return np.concatenate((ones, x), axis=1)

    @staticmethod
    def compute_cost(x, y, theta):
        with np.errstate(over='ignore'):
            X = MyLinearRegression.add_intercept(x)
            m = y.shape[0]
            y_hat = (np.dot(X, theta))

            dif = (y_hat - y)**2
            if np.isnan(dif).any() or np.isinf(dif).any():
                print(bcolors.WARNING
                      + "thetas is NaN, you should try a \
smaller learning rate "
                      + bcolors.RESET)
                exit(1)
            J = np.sum(dif) / (2 * m)
        return J

# --------------- TOOLS FUNCTIONS ----------------------------- #

    @staticmethod
    def check_thetas(thetas):
        for i in thetas:
            if np.isinf(i) or np.isnan(i):
                return False
        return True

    @staticmethod
    def is_numpy_array(x):
        return isinstance(x, np.ndarray)

    @staticmethod
    def is_not_empty(x):
        return x.size != 0

    @staticmethod
    def is_vertical_vector(x):
        if x.ndim > 2:
            return False
        if len(x.shape) == 1:
            return True
        if x.shape[1] == 1:
            return True
        return False

    @staticmethod
    def is_vector_same_size(x, y):
        return(len(x) == len(y))

    @staticmethod
    def reshape_vector(x):
        return x.reshape(len(x), 1)

    @staticmethod
    def reshape_if_needed(x):
        if len(x.shape) == 1:
            return x.reshape(len(x), 1)
        return x

    @staticmethod
    def is_matrix_corresponding(x, theta):
        return ((x.shape[1] + 1) == len(theta))

    @staticmethod
    def is_made_of_numbers(x):
        return (np.issubdtype(x.dtype, np.floating) or
                np.issubdtype(x.dtype, np.integer))

    @staticmethod
    def is_vector_not_empty_numpy(x):
        if MyLinearRegression.is_numpy_array(x) and \
           MyLinearRegression.is_not_empty(x) and \
           MyLinearRegression.is_made_of_numbers(x) and \
           MyLinearRegression.is_vertical_vector(x):
            return True
        return False

    @staticmethod
    def check_the_inputs(x, theta):
        if MyLinearRegression.is_numpy_array(x) and \
           MyLinearRegression.is_numpy_array(theta) and \
           MyLinearRegression.is_not_empty(x) and \
           MyLinearRegression.is_not_empty(theta) and \
           MyLinearRegression.is_made_of_numbers(x) and \
           MyLinearRegression.is_made_of_numbers(theta) and \
           MyLinearRegression.is_vertical_vector(theta) and \
           MyLinearRegression.is_matrix_corresponding(x, theta):
            return True
        return False

    @staticmethod
    def mean(x):
        if not MyLinearRegression.is_vector_not_empty_numpy(x):
            return None
        total = np.sum(x)
        length = len(x)

        return (float(total / length))

    @staticmethod
    def var(x):
        meanX = MyLinearRegression.mean(x)
        length = len(x)
        total = np.sum((x - meanX)**2) / length
        return (float(total))

    @staticmethod
    def std(x):
        return (float(MyLinearRegression.var(x)**0.5))

    @staticmethod
    def zscore(x):
        mu = MyLinearRegression.mean(x)
        if mu is None:
            return None
        std = MyLinearRegression.std(x)
        if std == 0:
            return None

        return(mu, std, ((x - mu) / std))


def ask_for_variables():
    ask = True
    default = True
    alpha, max_iter = 1e-1, 1000
    while ask:
        print(bcolors.OK
              + "Would you like to perform linear regression \
with your parameters ? (y/n)"
              + bcolors.RESET)
        print(bcolors.OK
              + "default : alpha = 0.1, max_iter =  1000"
              + bcolors.RESET)
        answer = input("\n>> ")
        if answer == 'y':
            default = False
            ask = False
        elif answer == 'n' or answer == '':
            ask = False
    if default:
        return alpha, max_iter
    ask = True
    while ask:
        try:
            alpha = float(input(bcolors.OK
                                + "Enter alpha (usually positive float \
between 0 and 1): "
                                + bcolors.RESET))
            max_iter = int(input(bcolors.OK
                                 + "Enter max_iter (int): "
                                 + bcolors.RESET))
            if alpha <= 0 or max_iter <= 0:
                raise ValueError
            ask = False
        except ValueError:
            print(bcolors.FAIL
                  + "wrong type of value, try again..."
                  + bcolors.RESET)
    return alpha, max_iter


if __name__ == "__main__":
    try:
        data = pd.read_csv("data.csv")
        Xkm = np.array(data['km']).reshape(-1, 1)
        Yprice = np.array(data['price']).reshape(-1, 1)
    except (FileNotFoundError, pd.errors.ParserError, KeyError):
        print(bcolors.FAIL
              + "csv datas corrupted or bad file name or bad data name"
              + bcolors.RESET)
        exit(1)

    thetas = [random(), random()]
    alpha, max_iter = ask_for_variables()
    # create instance of mylinear regression
    MyLR = MyLinearRegression(thetas, alpha, max_iter)
    mu, std, Xkm_normalized = MyLR.zscore(Xkm)
    if mu is None:
        print(bcolors.FAIL
              + "Can't normalize x : bad datas or empty datas"
              + bcolors.RESET)
        exit(1)
    J_history = MyLR.fit_(Xkm_normalized, Yprice)
    if J_history is None:
        exit(1)

    with open("thetas.csv", "w") as fitted:
        fitted.write("theta0,theta1,mu,std\n"
                     + str(MyLR.theta[0][0]) + ","
                     + str(MyLR.theta[1][0]) + ","
                     + str(mu) + "," + str(std))

    Y_hat = MyLR.predict_(Xkm_normalized)

    print("The thetas are : \n", MyLR.theta)
    MyLR.data_hypothesis_graph(Xkm, Yprice, Y_hat, J_history)
