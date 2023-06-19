import numpy as np


def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising
    its columns to every power in the range of 1 up to the power giveArgs:
    x: has to be an numpy.ndarray, a matrix of shape m * n.
    power: has to be an int, the power up to which the columns
    of matrix x are going to be raised.
    Returns:
    The matrix of polynomial features as a numpy.ndarray,
    of shape m * (np), containg the polynomial feature
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if not is_matrix_not_empty_numpy(x):
        return None
    x = reshape_if_needed(x)
    if not isinstance(power, int) or power < 1:
        return None
    m, n = x.shape
    return_matrix = np.zeros((m, n * power), dtype=x.dtype)
    for pow in range(power):
        for col in range(n):
            return_matrix[:, (pow * n) + col] = np.power(x[:, col], pow + 1)
    return return_matrix


def is_matrix_not_empty_numpy(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_max_2_dimensions(x) and \
       is_made_of_numbers(x):
        return True
    return False


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


def is_max_2_dimensions(x):
    return x.ndim <= 2


def is_numpy_array(x):
    return isinstance(x, np.ndarray)


def is_not_empty(x):
    return x.size != 0


def reshape_if_needed(x):
    if len(x.shape) == 1:
        return x.reshape(len(x), 1)
    return x


if __name__ == "__main__":
    import numpy as np
    x = np.arange(1, 11).reshape(5, 2)
    # Example 1:
    print(x.dtype)
    print(add_polynomial_features(x, 3))
    # Output:
    """array([[ 1, 2, 1, 4, 1, 8],
    [ 3, 4, 9, 16, 27, 64],
    [ 5, 6, 25, 36, 125, 216],
    [ 7, 8, 49, 64, 343, 512],
    [ 9, 10, 81, 100, 729, 1000]])"""
    # Example 2:
    print(add_polynomial_features(x, 4))
    # Output:
    """array([[ 1, 2, 1, 4, 1, 8, 1, 16],
    [ 3, 4, 9, 16, 27, 64, 81, 256],
    [ 5, 6, 25, 36, 125, 216, 625, 1296],
    [ 7, 8, 49, 64, 343, 512, 2401, 4096],
    [ 9, 10, 81, 100, 729, 1000, 6561, 10000]])"""
