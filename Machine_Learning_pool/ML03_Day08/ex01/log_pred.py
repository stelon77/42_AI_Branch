import numpy as np


def check_x_theta(fonction):
    """
    decorator used to check x and theta
    """
    def wrap_x_theta(*args, **kwargs):
        if not corresponding_size_matrix(args[0], args[1]):
            return None
        x = reshape_if_needed(args[0])
        theta = reshape_if_needed(args[1])
        ret = fonction(x, theta)
        return ret
    return wrap_x_theta


def check_x(fonction):
    """
    decorator used to check x
    """
    def wrap_x(*args, **kwargs):
        if not is_vector_not_empty_numpy_or_scalar(args[0]):
            return None
        x = reshape_if_needed(args[0])
        ret = fonction(x)
        return ret
    return wrap_x


@check_x_theta
def logistic_predict_(x, theta):
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
    X = add_intercept(x)
    return sigmoid_(np.dot(X, theta))


# --------------- TOOLS FUNCTIONS ----------------------------- #


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


def reshape_if_needed(x):
    if len(x.shape) == 1:
        return x.reshape(len(x), 1)
    return x


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


def is_scalar(x):
    return x.ndim == 0


def is_max_2_dimensions(x):
    return x.ndim <= 2


def is_matrix_size_corresponding(x, theta):
    return ((x.shape[1] + 1) == len(theta))


def is_vector_not_empty_numpy_or_scalar(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_made_of_numbers(x) and \
       (is_scalar(x) or
       is_vertical_vector(x)):
        return True
    return False


def is_vector_not_empty_numpy(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_made_of_numbers(x) and \
       is_vertical_vector(x):
        return True
    return False


def is_matrix_not_empty_numpy(x):
    if is_numpy_array(x) and \
       is_not_empty(x) and \
       is_max_2_dimensions(x) and \
       is_made_of_numbers(x):
        return True
    return False


def corresponding_size_matrix(x, theta):
    if is_matrix_not_empty_numpy(x) and \
       is_vector_not_empty_numpy(theta) and \
       is_matrix_size_corresponding(x, theta):
        return True
    return False


if __name__ == "__main__":
    # Example 1
    x = np.array([4]).reshape((-1, 1))
    theta = np.array([[2], [0.5]])
    print(logistic_predict_(x, theta))
    # Output:
    """array([[0.98201379]])"""
    # Example 1
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(logistic_predict_(x2, theta2))
    # Output:
    """array([[0.98201379],
    [0.99624161],
    [0.97340301],
    [0.99875204],
    [0.90720705]])"""
    # Example 3
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(logistic_predict_(x3, theta3))
    # Output:
    """array([[0.03916572],
    [0.00045262],
    [0.2890505 ]])"""
