import numpy as np


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
    if not corresponding_size_matrix(x, theta):
        return None
    theta = reshape_if_needed(theta)
    if not is_vector_not_empty_numpy(y):
        return None
    y = reshape_if_needed(y)
    if x.shape[0] != y.shape[0]:
        return None

    X = add_intercept(x)
    m = x.shape[0]
    return np.dot(np.transpose(X), (np.dot(X, theta) - y)) / m


def add_intercept(x):  # les checks ont ete pratiques avant
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


def reshape_if_needed(x):
    if len(x.shape) == 1:
        return x.reshape(len(x), 1)
    return x


def is_matrix_size_corresponding(x, theta):
    return ((x.shape[1] + 1) == len(theta))


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


def corresponding_size_matrix(x, theta):
    if is_numpy_array(x) and \
       is_numpy_array(theta) and \
       is_not_empty(x) and \
       is_not_empty(theta) and \
       is_made_of_numbers(x) and \
       is_made_of_numbers(theta) and \
       is_vertical_vector(theta) and \
       is_matrix_size_corresponding(x, theta):
        return True
    return False


def is_vector_not_empty_numpy(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_made_of_numbers(x) and \
       is_vertical_vector(x):
        return True
    return False


if __name__ == '__main__':
    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta1 = np.array([[0], [3], [0.5], [-6]])
    # Example 0:
    print(gradient(x, y, theta1))
    # Output:
    """array([[ -33.71428571],[ -37.35714286],[ 183.14285714],[ -393.]])"""
    # Example 1:
    theta2 = np.array([[0], [0], [0], [0]])
    print(gradient(x, y, theta2))
    # Output:
    """array([[ -0.71428571],[ 0.85714286],[ 23.28571429],[ -26.42857143]])"""
