import numpy as np


##########################################################
#                         DECORATORS                     #
##########################################################


def check_y_yhat_theta_lambda(fonction):
    """
    decorator used to check y, y_hat, theta and lambda
    """
    def wrap_y_yhat_theta_lambda(*args, **kwargs):
        if not is_vector_not_empty_numpy_same_size(args[0], args[1]):
            return None
        y = reshape_if_needed(args[0])
        y_hat = reshape_if_needed(args[1])
        if not is_vertical_vector(args[2]):
            return None
        theta = reshape_if_needed(args[2])
        if not isinstance(args[3], float):
            return None
        new_args = [y, y_hat, theta, args[3]]
        ret = fonction(*new_args, **kwargs)
        return ret
    return wrap_y_yhat_theta_lambda


def regularization(fonction):
    """
    decorator used to include the regularisation factor L2
    """
    def wrap_regularization(*args, **kwargs):
        ret = fonction(*args, **kwargs)
        m = args[0].shape[0]
        ret += (args[3] * l2_(args[2]) / (2 * m))  # lambda * L2
        return ret
    return wrap_regularization


##########################################################
#                   MAIN FUNCTION                        #
##########################################################


@check_y_yhat_theta_lambda
@regularization
def reg_log_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a
    logistic regression model from two non-empty numpy.ndarray,
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    The regularized loss as a float.
    None if y, y_hat, or theta is empty numpy.ndarray.
    None if y and y_hat do not share the same shapes.
    Raises:
    This function should not raise any Exception.
    """
    m = y.shape[0]
    one = np.ones((m, 1))
    epsilon = np.full((m, 1), 1e-15)  # epsilon factor to avoid log(0)
    loss_vector = float(np.dot(np.transpose(y),
                        np.log(y_hat + epsilon)) +
                        np.dot(np.transpose(one - y),
                        np.log(one - y_hat + epsilon)))
    loss = - loss_vector / m
    return loss


##########################################################
#              UTILITY FUNCTIONS                         #
##########################################################


def l2_(theta):  # no verification as it has been already checked
    """Computes the L2 regularization of a non-empty numpy.ndarray,
    """
    thetas = np.copy(theta)
    thetas[0, 0] = 0
    return float(np.dot(np.transpose(thetas), thetas))


def is_vector_not_empty_numpy_same_size(x, y):
    if is_vector_not_empty_numpy(x) and \
       is_vector_not_empty_numpy(y) and \
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


def is_vector_same_size(x, y):
    return(len(x) == len(y))


def reshape_if_needed(x):
    if len(x.shape) == 1:
        return x.reshape(len(x), 1)
    return x


def is_made_of_numbers(x):
    return (np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))


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


if __name__ == "__main__":
    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    # Example :
    print(reg_log_loss_(y, y_hat, theta, .5))
    # Output:
    """0.43377043716475955"""
    # Example :
    print(reg_log_loss_(y, y_hat, theta, .05))
    # Output:
    """0.13452043716475953"""
    # Example :
    print(reg_log_loss_(y, y_hat, theta, .9))
    # Output:
    """0.6997704371647596"""
