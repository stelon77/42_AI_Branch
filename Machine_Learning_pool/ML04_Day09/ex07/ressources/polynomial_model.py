import numpy as np
import math


def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values
    up to the power given in argument.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    power: has to be an int, the power up to which the components
    of vector x are going to be raised.
    Return:
    The matrix of polynomial features as a numpy.array, of shape m * n,
    containing the polynomial feature values for all training examples.
    None if x is an empty numpy.array.
    None if x or power is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not is_vector_not_empty_numpy(x):
        return None
    x = reshape_if_needed(x)
    if not isinstance(power, int) or power < 1:
        return None
    if power >= 2:
        for n in range(2, power + 1):
            x_power = np.power(x[:, [0]], n)
            x = np.concatenate((x, x_power), axis=1)
    return x


def is_numpy_array(x):
    return isinstance(x, np.ndarray)


def is_not_empty(x):
    return x.size != 0


def is_vertical_vector(x):
    if x.ndim > 2:
        return False
    if len(x.shape) == 1:
        return True
    if x.shape[1] == 1:
        return True
    return False


def reshape_vector(x):
    return x.reshape(len(x), 1)


def reshape_if_needed(x):
    if len(x.shape) == 1:
        return x.reshape(len(x), 1)
    return x


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


def is_vector_not_empty_numpy(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_made_of_numbers(x) and \
       is_vertical_vector(x):
        return True
    return False


if __name__ == '__main__':
    x = np.arange(1, 6).reshape(-1, 1)
    # Example 0:
    print(add_polynomial_features(x, 3))
    # Output:
    """array([[ 1, 1, 1],
    [ 2, 4, 8],
    [ 3, 9, 27],
    [ 4, 16, 64],
    [ 5, 25, 125]])"""
    # Example 1:
    print(add_polynomial_features(x, 6))
    # Output:
    """array([[ 1, 1, 1, 1, 1, 1],
    [ 2, 4, 8, 16, 32, 64],
    [ 3, 9, 27, 81, 243, 729],
    [ 4, 16, 64, 256, 1024, 4096],
    [ 5, 25, 125, 625, 3125, 15625]])"""
