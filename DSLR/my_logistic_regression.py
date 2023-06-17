#!/goinfre/lcoiffie/miniconda3/envs/42AI-lcoiffie/bin/python

import numpy as np
from utils import *


class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        if not isinstance(alpha, (float, int)) or \
           not isinstance(max_iter, int) or \
           not isinstance(theta, (np.ndarray, list, tuple)):
            print("bad arguments provided")
            exit(1)
        self.theta = np.array(theta)
        self.alpha = float(alpha)
        self.max_iter = max_iter

    @check_x_theta
    def predict_(self, x):
        """Computes the vector of prediction y_hat
        from two non-empty numpy.ndarray.
        Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
        Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exception.
        """
        # if not corresponding_size_matrix(x, self.theta):
        #     return None
        # x = reshape_if_needed(x)
        # self.theta = reshape_if_needed(self.theta)

        X = self.add_intercept(x)
        return self.sigmoid_(np.dot(X, self.theta))

    @staticmethod
    @check_x
    def sigmoid_(x):
        """
        Compute the sigmoid of a vector.
        Args:
        x: has to be a numpy.ndarray of shape (m, 1).
        Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
        Raises:
        This function should not raise any Exception.
        """
        return (1 / (1 + np.exp(-x)))

    @staticmethod
    def add_intercept(x):  # protection removed as x has been checked
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
        ones = np.ones((x.shape[0], 1))
        return np.concatenate((ones, x), axis=1)

    @check_y_yhat
    def loss_elem_(self, y, yhat):
        """
        Description:
        Calculates all the elements of the loss function.
        Args:
        y_hat: has to be an numpy.array, a vector.
        y: has to be an numpy.array, a vector.
        Returns:
        J_elem: numpy.array, a vector of dimension
        (number of the training examples,1).
        None if there is a dimension matching problem between y and y_hat.
        None if y or y_hat is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        m = y.shape[0]
        one = np.ones((m, 1))
        epsilon = np.full((m, 1), 1e-15)
        term_1 = np.log(yhat + epsilon)
        term_2 = np.log(one - yhat + epsilon)
        loss_elem = np.multiply(y, term_1) + np.multiply(one - y, term_2)
        loss_elem = - loss_elem / m
        return loss_elem

    @check_y_yhat
    def loss_(self, y, yhat):
        """
        Computes the logistic loss value.
        Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
        Returns:
        The logistic loss value as a float.
        None on any error.
        Raises:
        This function should not raise any Exception.
        """
        m = y.shape[0]
        one = np.ones((m, 1))
        epsilon = np.full((m, 1), 1e-15)
        loss = float(np.dot(np.transpose(y),
                     np.log(yhat + epsilon)) +
                     np.dot(np.transpose(one - y),
                     np.log(one - yhat + epsilon)))
        loss = - loss / m
        return loss

    @check_x_y_theta
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
        J_history = np.zeros((self.max_iter, 1))  # #####
        m = x.shape[0]
        for i in range(self.max_iter):
            self.theta = self.theta - (self.alpha * self.log_gradient(x, y))
            if np.isinf(self.theta).any() or np.isnan(self.theta).any():
                print("gradient is diverging, choose a smaller alpha")
                return None
            y_hat = self.predict_(x)  # #######
            loss = self.loss_(y, y_hat)  # #######
            J_history[i] = loss  # ########
        visual(J_history)  # ##########
        return self

    # @check_x_y_theta
    def log_gradient(self, x, y):  # protections removed to optimize algorithm
        """Computes a gradient vector from three non-empty numpy.ndarray,
        with a for-loop. The three arrays must have compatiblArgs:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
        Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1,
        containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
        Raises:
        This function should not raise any Exception.
        """
        m = x.shape[0]
        y_hat = self.predict_(x)
        X = self.add_intercept(x)

        return np.dot(np.transpose(X), (y_hat - y)) / m


if __name__ == "__main__":
    from my_logistic_regression import MyLogisticRegression as MyLR

    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLR(thetas)
    # Example 0:
    y_hat = mylr.predict_(X)
    print(y_hat)
    # Output:
    """array([[0.99930437],
    [1. ],
    [1. ]])"""
    # Example 1:
    # loss_elem = mylr.loss_elem_(Y, y_hat)
    # print("loss_elem  = ", loss_elem)
    # print("sum = ", np.sum(loss_elem))
    print(mylr.loss_(Y, y_hat))
    # Output:
    """11.513157421577004"""
    # Example 2:
    mylr.fit_(X, Y)
    print(mylr.theta)
    # Output:
    """array([[ 2.11826435]
    [ 0.10154334]
    [ 6.43942899]
    [-5.10817488]
    [ 0.6212541 ]])"""
    # Example 3:
    print(mylr.predict_(X))
    # Output:
    """array([[0.57606717]
    [0.68599807]
    [0.06562156]])"""
    # Example 4:
    y_hat = mylr.predict_(X)
    # loss_elem = mylr.loss_elem_(Y, y_hat)
    # print("loss_elem  = ", loss_elem)
    # print("sum = ", np.sum(loss_elem))
    print(mylr.loss_(Y, y_hat))
    # Output:
    """1.4779126923052268"""
