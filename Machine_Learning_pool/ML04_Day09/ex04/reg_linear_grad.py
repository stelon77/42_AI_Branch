import numpy as np


##########################################################
#                         DECORATORS                     #
##########################################################


def check_y_x_theta_lambda(fonction):
    """
    decorator used to check y, x, theta and lambda
    """
    def wrap_y_x_theta_lambda(*args, **kwargs):
        y = args[0]
        x = args[1]
        theta = args[2]
        lambda_ = args[3]

        if not corresponding_size_matrix(x, theta):
            return None
        if not is_vector_not_empty_numpy(y):
            return None
        y = reshape_if_needed(y)
        theta = reshape_if_needed(theta)
        x = reshape_if_needed(x)
        if not is_same_number_of_lines(x, y):
            return None
        if not isinstance(lambda_, (float, int)):
            return None
        lambda_ = float(lambda_)
        new_args = [y, x, theta, lambda_]
        ret = fonction(*new_args, **kwargs)
        return ret
    return wrap_y_x_theta_lambda


def grad_regularization(fonction):
    """
    decorator used to include the regularisation factor L2
    """
    def wrap_grad_regularization(*args, **kwargs):
        ret = fonction(*args, **kwargs)
        x = args[1]
        thetas = np.copy(args[2])
        lambda_ = args[3]
        m, n = x.shape
        thetas[0] = 0
        ret += (lambda_ * thetas) / m  # lambda * L2
        return ret
    return wrap_grad_regularization


##########################################################
#                   MAIN FUNCTION                        #
##########################################################


@check_y_x_theta_lambda
@grad_regularization
def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of
    three non-empty numpy.ndarray, with two for-loop.
    The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1,
    containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    m, n = x.shape
    gradient = np.zeros((n + 1, 1))
    y_hat = np.zeros((m, 1))
    for i in range(m):
        y_hat[i] = theta[0]
        for j in range(n):
            y_hat[i] += theta[j + 1] * x[i, j]
    dif = y_hat - y
    gradient[0] = np.sum(dif) / m
    for j in range(1, n + 1):
        sum = 0
        for a, b in zip(dif, x[:, j - 1]):
            sum += a * b
        gradient[j] = sum / m
    return gradient


@check_y_x_theta_lambda
@grad_regularization
def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of
    three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1,
    containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    m, n = x.shape
    X = add_intercept(x)
    return np.dot(np.transpose(X), (np.dot(X, theta) - y)) / m


##########################################################
#              UTILITY FUNCTIONS                         #
##########################################################


def add_intercept(x):  # protection removed as x has been checked
    """Adds a column of 1's to the non-empty numpy.array x of shape m * n..
    """
    ones = np.ones((x.shape[0], 1))
    return np.concatenate((ones, x), axis=1)


def corresponding_size_matrix(x, theta):
    if is_numpy_array(x) and \
       is_numpy_array(theta) and \
       is_not_empty(x) and \
       is_not_empty(theta) and \
       is_made_of_numbers(x) and \
       is_made_of_numbers(theta) and \
       is_vertical_vector(theta) and \
       is_max_2_dimensions(x) and \
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


def is_max_2_dimensions(x):
    return x.ndim <= 2


def is_matrix_size_corresponding(x, theta):
    return ((x.shape[1] + 1) == len(theta))


def is_same_number_of_lines(x, y):
    return(x.shape[0] == y.shape[0])


if __name__ == '__main__':
    x = np.array([[-6, -7, -9],
                  [13, -2, 14],
                  [-7, 14, -1],
                  [-8, -4, 6],
                  [-5, -9, 6],
                  [1, -5, 11],
                  [9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])
    # Example 1.1:
    print(reg_linear_grad(y, x, theta, 1))
    # Output:
    """array([[ -60.99 ],
    [-195.64714286],
    [ 863.46571429],
    [-644.52142857]])"""
    # Example 1.2:
    print(vec_reg_linear_grad(y, x, theta, 1))
    # Output:
    """array([[ -60.99 ],
    [-195.64714286],
    [ 863.46571429],
    [-644.52142857]])"""
    # Example 2.1:
    print(reg_linear_grad(y, x, theta, 0.5))
    # Output:
    """array([[ -60.99 ],
    [-195.86142857],
    [ 862.71571429],
    [-644.09285714]])"""
    # Example 2.2:
    print(vec_reg_linear_grad(y, x, theta, 0.5))
    # Output:
    """array([[ -60.99 ],
    [-195.86142857],
    [ 862.71571429],
    [-644.09285714]])"""
    # Example 3.1:
    print(reg_linear_grad(y, x, theta, 0.0))
    # Output:
    """array([[ -60.99 ],
    [-196.07571429],
    [ 861.96571429],
    [-643.66428571]])"""
    # Example 3.2:
    print(vec_reg_linear_grad(y, x, theta, 0.0))
    # Output:
    """array([[ -60.99 ],
    [-196.07571429],
    [ 861.96571429],
    [-643.66428571]])"""
