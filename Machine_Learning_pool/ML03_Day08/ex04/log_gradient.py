import numpy as np


##########################################################
#                         DECORATORS                     #
##########################################################
def check_x_y_theta(fonction):
    """
    decorator used to check x and y and theta
    """
    def wrap_x_y_theta(*args, **kwargs):
        if not corresponding_size_matrix(args[0], args[2]) or\
           not is_vector_not_empty_numpy(args[1]) or \
           not is_same_number_of_lines(args[0], args[1]):
            return None
        y = reshape_if_needed(args[1])
        theta = reshape_if_needed(args[2])
        ret = fonction(args[0], y, theta)
        return ret
    return wrap_x_y_theta


def check_y_yhat_eps(fonction):
    """
    decorator used to check y and y_hat and eps
    """
    def wrap_y_yhat_eps(*args, **kwargs):
        if not is_vector_not_empty_numpy_same_size(args[0], args[1]):
            return None
        y = reshape_if_needed(args[0])
        y_hat = reshape_if_needed(args[1])
        new_args = [y, y_hat]
        for cle in kwargs:
            if cle != "eps":
                return None
            if not isinstance(kwargs[cle], (int, float)):
                return None
        ret = fonction(*new_args, **kwargs)
        return ret
    return wrap_y_yhat_eps


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

#############################################################


@check_x_y_theta
def log_gradient(x, y, theta):
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
    y_hat = logistic_predict_(x, theta)
    X = add_intercept(x)
    grad = np.zeros((X.shape[1], 1))
    for j in range(X.shape[1]):
        for i in range(m):
            grad[j] += (float(y_hat[i]) - float(y[i])) * float(X[i][j])
        grad[j] /= m
    return grad

# --------------- TOOLS FUNCTIONS ----------------------------- #


@check_y_yhat_eps
def vec_log_loss_(y, y_hat, eps=1e-15):
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
    epsilon = np.full((m, 1), eps)
    loss_vector = float(np.dot(np.transpose(y),
                        np.log(y_hat + epsilon)) +
                        np.dot(np.transpose(one - y),
                        np.log(one - y_hat + epsilon)))
    loss = - loss_vector / m
    return loss


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
    # if not is_matrix_not_empty_numpy(x):
    #     return None
    # if len(x.shape) == 1:
    #     x = x.reshape((x.shape[0], 1))
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


def is_vector_same_size(x, y):
    return(len(x) == len(y))


def is_same_number_of_lines(x, y):
    return(x.shape[0] == y.shape[0])


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


def is_vector_not_empty_numpy_same_size(x, y):
    if is_vector_not_empty_numpy(x) and \
       is_vector_not_empty_numpy(y) and \
       is_vector_same_size(x, y):
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
    # Example 1:
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    print(log_gradient(x1, y1, theta1))
    # Output:
    """array([[-0.01798621],
    [-0.07194484]])"""
    # Example 2:
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(log_gradient(x2, y2, theta2))
    # Output:
    """array([[0.3715235 ],
    [3.25647547]])"""
    # Example 3:
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(log_gradient(x3, y3, theta3))
    # Output:
    """array([[-0.55711039],
    [-0.90334809],
    [-2.01756886],
    [-2.10071291],
    [-3.27257351]])"""
