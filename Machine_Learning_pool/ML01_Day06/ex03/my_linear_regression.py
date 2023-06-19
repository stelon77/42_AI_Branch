#!/usr/bin/python
import numpy as np


class MyLinearRegression:
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        if not isinstance(alpha, (float, int)) or \
           not isinstance(max_iter, int) or \
           not isinstance(thetas, (np.ndarray, list, tuple)):
            print("bad arguments provided")
            exit(1)
        self.thetas = np.array(thetas).reshape(len(thetas), 1)

    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
        x: has to be a numpy.array, a vector of shape m * 1:
        (number of training examples, 1).
        y: has to be a numpy.array, a vector of shape m * 1:
        (number of training examples, 1).
        Return:
        new_theta: numpy.array, a vector of shape 2 * 1.
        None if there is a matching shape problem.
        None if x, y, theta, alpha or max_iter is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not MyLinearRegression.is_two_vector_not_empty_same_size(x, y):
            return None
        x = MyLinearRegression.reshape_if_needed(x)
        y = MyLinearRegression.reshape_if_needed(y)
        if not MyLinearRegression.is_not_empty_numpy_vector_2_1(self.thetas):
            return None
        self.thetas = MyLinearRegression.reshape_if_needed(self.thetas)
        if (not isinstance(self.alpha, float) or
           not isinstance(self.max_iter, int) or
           self.max_iter <= 0):
            return None

        X = MyLinearRegression.add_intercept(x)
        new_thetas = self.thetas
        for i in range(self.max_iter):
            nabla_J = (np.dot(np.transpose(X), (np.dot(X, new_thetas) - y))
                       / len(x))
            new_thetas = new_thetas - (nabla_J * self.alpha)
        self.thetas = new_thetas
        return new_thetas

    def predict_(self, x):
        """"
        Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
        Returns:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exceptions.
        """
        if not MyLinearRegression.is_not_empty_numpy_vector_2_1(self.thetas):
            return None
        self.thetas = MyLinearRegression.reshape_if_needed(self.thetas)
        if not MyLinearRegression.is_vector_not_empty_numpy(x):
            return None
        x = MyLinearRegression.reshape_if_needed(x)

        X = MyLinearRegression.add_intercept(x)
        if X.shape[1] != self.thetas.shape[0]:
            return None
        y_hat = np.dot(X, self.thetas)
        return y_hat

    def loss_elem_(self, y, y_hat):

        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_elem: numpy.array, a vector of dimension
        (number of the training examples,1).
        None if there is a dimension matching problem between y and y_hat.
        None if y or y_hat is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        if not MyLinearRegression.is_two_vector_not_empty_same_size(y, y_hat):
            return None
        x = MyLinearRegression.reshape_if_needed(y_hat)
        y = MyLinearRegression.reshape_if_needed(y)
        dif = y_hat - y
        return(np.square(dif))

    def loss_(self, y, y_hat):
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
        J_elem = MyLinearRegression.loss_elem_(self, y, y_hat)
        if J_elem is None or len(J_elem) == 0:
            return None
        J_value = np.sum(J_elem) / (2 * len(J_elem))
        return float(J_value)

    def mse_(self, y, y_hat):
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
        return (MyLinearRegression.loss_(self, y, y_hat) * 2)

    # --------------- TOOLS FUNCTIONS ----------------------------- #

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
    def is_made_of_numbers(x):
        return (np.issubdtype(x.dtype, np.floating) or
                np.issubdtype(x.dtype, np.integer))

    @staticmethod
    def is_two_vector_not_empty_same_size(x, y):
        if MyLinearRegression.is_numpy_array(x) and \
           MyLinearRegression.is_numpy_array(y) and \
           MyLinearRegression.is_not_empty(x) and \
           MyLinearRegression.is_not_empty(y) and \
           MyLinearRegression.is_made_of_numbers(x) and \
           MyLinearRegression.is_made_of_numbers(y) and \
           MyLinearRegression.is_vertical_vector(x) and \
           MyLinearRegression.is_vertical_vector(y) and \
           MyLinearRegression.is_vector_same_size(x, y):
            return True
        return False

    @staticmethod
    def is_not_empty_numpy_vector_2_1(x):
        if MyLinearRegression.is_numpy_array(x) and \
           MyLinearRegression.is_not_empty(x) and \
           MyLinearRegression.is_made_of_numbers(x) and \
           MyLinearRegression.is_vertical_vector and \
           len(x) == 2:
            return True
        return False

    @staticmethod
    def is_vector_not_empty_numpy(x):
        if MyLinearRegression.is_numpy_array(x) and \
           MyLinearRegression.is_not_empty(x) and \
           MyLinearRegression.is_made_of_numbers(x) and \
           MyLinearRegression.is_vertical_vector(x):
            return True
        return False

    @staticmethod
    def is_matrix_not_empty_numpy(x):
        if MyLinearRegression.is_numpy_array(x) and \
           MyLinearRegression.is_not_empty(x) and \
           MyLinearRegression.is_made_of_numbers(x):
            return True
        return False

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
        if not MyLinearRegression.is_matrix_not_empty_numpy(x):
            return None
        if len(x.shape) == 1:
            x = x.reshape((x.shape[0], 1))
        ones = np.ones((len(x), 1))
        return np.concatenate((ones, x), axis=1)


if __name__ == "__main__":
    import numpy as np
    from my_linear_regression import MyLinearRegression as MyLR

    x = np.array([[12.4956442], [21.5007972],
                  [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236],
                  [45.7655287], [46.6793434], [59.5585554]])
    lr1 = MyLR([2, 0.7])
    # Example 0.0:
    print(lr1.predict_(x))
    # Output:
    """array([[10.74695094],
    [17.05055804],
    [24.08691674],
    [36.24020866],
    [42.25621131]])"""
    # Example 0.1:
    print(lr1.loss_elem_(lr1.predict_(x), y))
    # Output:
    """array([[710.45867381],
    [364.68645485],
    [469.96221651],
    [108.97553412],
    [299.37111101]])"""
    # Example 0.2:
    print(lr1.loss_(lr1.predict_(x), y))
    # Output:
    """195.34539903032385"""
    # Example 1.0:
    lr2 = MyLR([1, 1], 5e-8, 1500000)
    lr2.fit_(x, y)
    print(lr2.thetas)
    # Output:
    """array([[1.40709365],
    [1.1150909 ]])"""
    # Example 1.1:
    print(lr2.predict_(x))
    # Output:
    """array([[15.3408728 ],
    [25.38243697],
    [36.59126492],
    [55.95130097],
    [65.53471499]])"""
    # Example 1.2:
    print(lr2.loss_elem_(lr2.predict_(x), y))
    # Output:
    """array([[486.66604863],
    [115.88278416],
    [ 84.16711596],
    [ 85.96919719],
    [ 35.71448348]])"""
    # Example 1.3:
    print(lr2.loss_(lr2.predict_(x), y))
    # Output:
    """80.83996294128525"""
