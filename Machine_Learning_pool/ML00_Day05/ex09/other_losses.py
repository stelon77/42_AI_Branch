import numpy as np
from math import fabs
# from .tools import *


def mse_elem(y, y_hat):
    """
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_elem: numpy.array, a vector of
    dimension (number of the training examples,1).
    None if there is a dimension matching problem between y and y_hat.
    None if y or y_hat is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    tool = tools()
    if not tool.is_two_vector_not_empty_same_size_numpy(y, y_hat):
        return None
    y = tool.reshape_if_needed(y)
    y_hat = tool.reshape_if_needed(y_hat)
    return np.square(y_hat - y)


def rmse_elem(y, y_hat):
    """
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_elem: numpy.array, a vector of
    dimension (number of the training examples,1).
    None if there is a dimension matching problem between y and y_hat.
    None if y or y_hat is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    tool = tools()
    if not tool.is_two_vector_not_empty_same_size_numpy(y, y_hat):
        return None
    y = tool.reshape_if_needed(y)
    y_hat = tool.reshape_if_needed(y_hat)
    return np.square(y_hat - y)


def mae_elem(y, y_hat):
    """
    Description:
    Calculates all the elements |y_pred - y| of the loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_elem: numpy.array, a vector of
    dimension (number of the training examples,1).
    None if there is a dimension matching problem between y and y_hat.
    None if y or y_hat is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    tool = tools()
    if not tool.is_two_vector_not_empty_same_size_numpy(y, y_hat):
        return None
    y = tool.reshape_if_needed(y)
    y_hat = tool.reshape_if_needed(y_hat)

    return np.abs(y_hat - y)


def r2score_elem(y, y_hat):
    """
    Description:
    Calculates all the elements (y_hat - y)^2 and (y - mean_y)^2
    of the loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_elem:  2 numpy.array, vectors of
    dimension (number of the training examples,1).
    None if there is a dimension matching problem between y and y_hat.
    None if y or y_hat is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    tool = tools()
    if not tool.is_two_vector_not_empty_same_size_numpy(y, y_hat):
        return None
    y = tool.reshape_if_needed(y)
    y_hat = tool.reshape_if_needed(y_hat)
    y_mean = np.sum(y) / len(y)
    return np.square(y_hat - y), np.square(y - y_mean)


def mse_(y, y_hat):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of shape m * 1.
    y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
    mse: has to be a float.
    None if there is a matching shape problem.
    Raises:
    This function should not raise any Exception.
    """
    J_elem = mse_elem(y, y_hat)
    J_value = np.sum(J_elem) / len(J_elem)
    return float(J_value)


def rmse_(y, y_hat):
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of shape m * 1.
    y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
    rmse: has to be a float.
    None if there is a matching shape problem.
    Raises:
    This function should not raise any Exception.
    """
    J_elem = rmse_elem(y, y_hat)
    J_value = np.sqrt((np.sum(J_elem) / len(J_elem)))
    return float(J_value)


def mae_(y, y_hat):
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of shape m * 1.
    y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
    mae: has to be a float.
    None if there is a matching shape problem.
    Raises:
    This function should not raise any Exception.
    """
    J_elem = mae_elem(y, y_hat)
    J_value = np.sum(J_elem) / len(J_elem)
    return float(J_value)


def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.array, a vector of shape m * 1.
    y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching shape problem.
    Raises:
    This function should not raise any Exception.
    """
    J_elem_num, J_elem_denom = r2score_elem(y, y_hat)
    denom = np.sum(J_elem_denom)
    if denom == 0:
        print("division by zero in R2score")
        return 1.0

    J_value = 1 - (np.sum(J_elem_num) / denom)
    return float(J_value)

# -----------------------tools---------------------#


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
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
    tool = tools()
    if not tool.is_vector_not_empty_numpy(theta):
        return None
    if not tool.is_numpy_array(x) or \
       not tool.is_not_empty(x) or \
       not tool.is_made_of_numbers(x):
        return None
    if len(x.shape) != 1:
        if (len(x[0]) + 1) != len(theta):
            return None
    else:
        if theta.shape not in ((2,), (2, 1)):
            return None

    X = tool.add_intercept(x)
    if X.shape[1] != theta.shape[0]:
        return None
    y_hat = np.dot(X, theta)
    return y_hat


class tools:
    """ All the tools we need """
    def __init__(self):
        pass

    @staticmethod
    def add_intercept(x):
        """Adds a column of 1’s to the non-empty numpy.array x.
        Args:
        x: has to be an numpy.array, a vector of shape m * n.
        Returns:
        x as a numpy.array, a vector of shape m * (n + 1).
        None if x is not a numpy.array.
        None if x is a empty numpy.array.
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or x.size == 0:
            return None
        if len(x.shape) == 1:
            x = x.reshape((x.shape[0], 1))
        ones = np.ones((len(x), 1))
        return np.concatenate((ones, x), axis=1)

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
        if len(x.shape) == 1 or x.shape[1] == 1:
            return True
        return False

    @staticmethod
    def is_made_of_numbers(x):
        return (np.issubdtype(x.dtype, np.floating) or
                np.issubdtype(x.dtype, np.integer))

    @staticmethod
    def is_vector_same_size(x, y):
        return(len(x) == len(y))

    def is_two_vector_not_empty_same_size_numpy(self, x, y):
        if self.is_numpy_array(x) and \
           self.is_numpy_array(y) and \
           self.is_not_empty(x) and \
           self.is_not_empty(y) and \
           self.is_made_of_numbers(x) and \
           self.is_made_of_numbers(y) and \
           self.is_vertical_vector and \
           self.is_vertical_vector(y) and \
           self.is_vector_same_size(x, y):
            return True
        return False

    def is_vector_not_empty_numpy(self, x):
        if self.is_numpy_array(x) and \
           self.is_not_empty(x) and \
           self.is_made_of_numbers(x) and \
           self.is_vertical_vector(x):
            return True
        return False

    def is_vector_not_empty_numpy_2_x_1(self, x):
        if not self.is_vector_not_empty_numpy(x):
            return False
        if x.shape not in ((2,), (2, 1)):
            return False
        return True

    @staticmethod
    def reshape_if_needed(x):
        if len(x.shape) == 1:
            return x.reshape(len(x), 1)
        return x


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error, \
         mean_absolute_error, r2_score
    from math import sqrt
    x = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    # Mean squared error
    # your implementation
    print(mse_(x, y))
    # Output:
    """4.285714285714286"""
    # sklearn implementation
    print(mean_squared_error(x, y))
    # Output:
    """4.285714285714286"""
    # Root mean squared error
    # your implementation
    print(rmse_(x, y))
    # Output:
    """2.0701966780270626"""
    # sklearn implementation not available: take the square root of MSE
    print(sqrt(mean_squared_error(x, y)))
    # Output:
    """2.0701966780270626"""
    # Mean absolute error
    # your implementation
    print(mae_(x, y))
    # Output:
    """1.7142857142857142"""
    # sklearn implementation
    print(mean_absolute_error(x, y))
    # Output:
    """1.7142857142857142"""
    # R2-score
    # your implementation
    print(r2score_(x, y))
    # Output:
    """0.9681721733858745"""
    # sklearn implementation
    print(r2_score(x, y))
    # Output:
    """0.9681721733858745"""
