import numpy as np


def loss_(y, y_hat):
    """Computes the mean squared error of two
    non-empty numpy.array, without any for loop.
    The two arrays must have the same shapes.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Return:
    The mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same shapes.
    None if y or y_hat is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not is_two_vector_not_empty_same_size_numpy(y, y_hat):
        return None
    y = reshape_if_needed(y)
    y_hat = reshape_if_needed(y_hat)
    m = len(y)
    dif = y_hat - y
    return float((np.dot(np.transpose(dif), dif) / (2 * m)))


# --------------- TOOLS FUNCTIONS ----------------------------- #


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


def is_vector_same_size(x, y):
    return(len(x) == len(y))


def reshape_vector(x):
    return x.reshape(len(x), 1)


def reshape_if_needed(x):
    if len(x.shape) == 1:
        return x.reshape(len(x), 1)
    return x


def is_matrix_size_corresponding(x, theta):
    return ((x.shape[1] + 1) == len(theta))


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


def is_two_vector_not_empty_same_size_numpy(x, y):
    if is_numpy_array(x) and \
       is_numpy_array(y) and \
       is_not_empty(x) and \
       is_not_empty(y) and \
       is_made_of_numbers(x) and \
       is_made_of_numbers(y) and \
       is_vertical_vector(x) and \
       is_vertical_vector(y) and \
       is_vector_same_size(x, y):
        return True
    return False


if __name__ == '__main__':
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    # Example 0:
    print(loss_(X, Y))
    # Output:
    """2.142857142857143"""
    # Example 1:
    print(loss_(X, X))
    # Output:
    """0.0"""
