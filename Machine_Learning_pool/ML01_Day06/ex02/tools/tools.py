import numpy as np


def is_numpy_array(x):
    return isinstance(x, np.ndarray)


def is_not_empty(x):
    return x.size != 0


def is_vertical_vector(x):
    if x.ndim > 2:
        return False
    if len(x.shape) == 1 or x.shape[1] == 1:
        return True
    return False


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


def is_vector_same_size(x, y):
    return(len(x) == len(y))


def reshape_if_needed(x):
    if len(x.shape) == 1:
        return x.reshape(len(x), 1)
    return x


def is_two_vector_not_empty_same_size_numpy(x, y):
    if is_numpy_array(x) and \
       is_numpy_array(y) and \
       is_not_empty(x) and \
       is_not_empty(y) and \
       is_made_of_numbers(x) and \
       is_made_of_numbers(y) and \
       is_vertical_vector and \
       is_vertical_vector(y) and \
       is_vector_same_size(x, y):
        return True
    return False


def is_vector_not_empty_numpy(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_made_of_numbers(x) and \
       is_vertical_vector(x):
        return True
    return False


def is_vector_not_empty_numpy_2_x_1(x):
    if not is_vector_not_empty_numpy(x):
        return False
    if x.shape not in ((2,), (2, 1)):
        return False
    return True


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
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None

    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    ones = np.ones((len(x), 1))
    return np.concatenate((ones, x), axis=1)


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
    if not is_vector_not_empty_numpy(x) or \
       not is_vector_not_empty_numpy_2_x_1(theta):
        return None
    x = reshape_if_needed(x)
    theta = reshape_if_needed(theta)

    X = add_intercept(x)
    if X.shape[1] != theta.shape[0]:
        return None
    y_hat = np.dot(X, theta)
    return y_hat
