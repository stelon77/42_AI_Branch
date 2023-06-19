import numpy as np


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
    if not check_x(x):
        return None
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    ones = np.ones((len(x), 1))
    return np.concatenate((ones, x), axis=1)


def check_x(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_made_of_numbers(x):
        return True
    return False


def is_numpy_array(x):
    return isinstance(x, np.ndarray)


def is_not_empty(x):
    return x.size != 0


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


if __name__ == '__main__':
    # Example 1:
    x = np.arange(1, 6).reshape((5, 1))
    print(add_intercept(x))
    # # Output:
    # array([[1., 1.],
    # [1., 2.],
    # [1., 3.],
    # [1., 4.],
    # [1., 5.]])

    # # Example 2:
    y = np.arange(1, 10).reshape((3, 3))
    print(add_intercept(y))
    # # Output:
    # array([[1., 1., 2., 3.],
    # [1., 4., 5., 6.],
    # [1., 7., 8., 9.]])
